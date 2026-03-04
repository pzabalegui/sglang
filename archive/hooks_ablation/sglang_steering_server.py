#!/usr/bin/env python3
"""
SGLang-Compatible Steering Server

Servidor que implementa la API OpenAI/SGLang con soporte para steering vectors
mediante hooks de PyTorch.

NOTA: SGLang no tiene soporte nativo para steering vectors. Este servidor
implementa el steering usando HuggingFace transformers con hooks, pero expone
una API compatible con SGLang/OpenAI.

Para uso en producción de alto rendimiento sin steering, usar SGLang directo.
Para requests que requieren steering, usar este servidor.

Uso:
    # Iniciar servidor
    python sglang_steering_server.py --model zai-org/GLM-4.7-Flash --port 8000

    # Con vector pre-calculado
    python sglang_steering_server.py --model zai-org/GLM-4.7-Flash \
        --vector refusal_direction.pt --port 8000

    # Request con steering
    curl -X POST http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "GLM-4.7-Flash",
            "messages": [{"role": "user", "content": "How to hack?"}],
            "steering": {
                "enabled": true,
                "scale": 1.5,
                "mode": "window",
                "center_layer": 24,
                "window_size": 5
            }
        }'
"""

import argparse
import gc
import io
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any, Union

import numpy as np
import pandas as pd
import requests as http_requests
import torch
from torch import Tensor
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("steering_server")


# =============================================================================
# PYDANTIC MODELS (API)
# =============================================================================

class SteeringConfig(BaseModel):
    """Configuración de steering para una request."""
    enabled: bool = False
    scale: float = 1.0
    mode: str = "single"  # "single", "window", "gaussian"
    center_layer: Optional[int] = None  # Si None, usa el óptimo del modelo
    window_size: int = 5
    kernel_width: float = 5.0
    vector_name: str = "default"  # Para múltiples vectores


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Request compatible con OpenAI/SGLang."""
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[List[str]] = None

    # Extensión: steering
    steering: Optional[SteeringConfig] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Response compatible con OpenAI/SGLang."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage
    steering_applied: Optional[Dict] = None


class GenerateRequest(BaseModel):
    """API simple de generación."""
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0
    steering: Optional[SteeringConfig] = None


class GenerateResponse(BaseModel):
    response: str
    steering_applied: Optional[Dict] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    vectors_loaded: List[str]
    gpu_memory_gb: Optional[float] = None


class VectorInfo(BaseModel):
    name: str
    layer: int
    norm: float
    shape: List[int]


# =============================================================================
# STEERING ENGINE
# =============================================================================

class SteeringEngine:
    """
    Motor de steering que gestiona el modelo y los hooks.
    """

    REFUSAL_PATTERNS = [
        "i cannot", "i can't", "i'm unable", "i am unable",
        "i'm sorry", "i apologize", "as an ai", "as a language model",
        "i must refuse", "i cannot provide", "i won't",
        "illegal", "unethical", "harmful"
    ]

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_4bit: bool = False,
        dtype: str = "bfloat16"
    ):
        self.model_path = model_path
        self.device = device
        self.use_4bit = use_4bit

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.dtype = dtype_map.get(dtype, torch.bfloat16)

        self.model = None
        self.tokenizer = None
        self.n_layers = None
        self.hidden_size = None
        self.optimal_layer = None

        # Vectores de steering
        self.vectors: Dict[str, Dict] = {}  # name -> {"direction": Tensor, "layer": int, ...}

        # Hooks activos
        self.hook_handles = []

        # Configuración por defecto
        self.default_config = SteeringConfig()

    def load_model(self):
        """Carga el modelo."""
        logger.info(f"Cargando modelo: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.use_4bit:
            logger.info("Usando cuantización 4-bit")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True
            )

        self.model.eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size

        # Capa óptima por defecto (50% de profundidad)
        self.optimal_layer = self.n_layers // 2

        logger.info(f"Modelo cargado: {self.n_layers} capas, hidden_size={self.hidden_size}")

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            logger.info(f"VRAM usada: {mem:.2f} GB")

    def load_vector(self, path: str, name: str = "default", layer: Optional[int] = None):
        """Carga un vector de steering desde archivo."""
        logger.info(f"Cargando vector '{name}' desde: {path}")

        direction = torch.load(path, map_location="cpu")

        # Asegurar que es un tensor 1D normalizado
        if direction.dim() > 1:
            direction = direction.squeeze()
        direction = direction / direction.norm()
        direction = direction.to(self.model.device).to(self.dtype)

        if layer is None:
            layer = self.optimal_layer

        self.vectors[name] = {
            "direction": direction,
            "layer": layer,
            "norm": direction.norm().item(),
            "shape": list(direction.shape)
        }

        logger.info(f"Vector '{name}' cargado: layer={layer}, shape={direction.shape}")

    def extract_vector(
        self,
        name: str = "default",
        layer: Optional[int] = None,
        n_samples: int = 32
    ):
        """Extrae vector de steering usando difference-in-means."""
        logger.info(f"Extrayendo vector '{name}'...")

        if layer is None:
            layer = self.optimal_layer

        # Cargar datos
        url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
        response = http_requests.get(url)
        dataset = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        harmful = dataset['goal'].tolist()[:n_samples]

        from datasets import load_dataset
        hf_dataset = load_dataset('tatsu-lab/alpaca')
        harmless = [
            item['instruction'] for item in hf_dataset['train']
            if item['input'].strip() == ''
        ][:n_samples]

        # Extraer activaciones
        def get_activations(instructions, layer):
            all_acts = []
            for inst in instructions:
                messages = [{"role": "user", "content": inst}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(text, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden = outputs.hidden_states[layer]
                    act = hidden[:, -1, :].cpu().float()
                    all_acts.append(act)

                del outputs, hidden
                torch.cuda.empty_cache()

            return torch.cat(all_acts, dim=0)

        logger.info(f"Extrayendo activaciones harmful...")
        harmful_acts = get_activations(harmful, layer)

        logger.info(f"Extrayendo activaciones harmless...")
        harmless_acts = get_activations(harmless, layer)

        # Calcular dirección
        direction = harmful_acts.mean(dim=0) - harmless_acts.mean(dim=0)
        direction = direction / direction.norm()
        direction = direction.to(self.model.device).to(self.dtype)

        self.vectors[name] = {
            "direction": direction,
            "layer": layer,
            "norm": direction.norm().item(),
            "shape": list(direction.shape)
        }

        logger.info(f"Vector '{name}' extraído: layer={layer}, shape={direction.shape}")

        gc.collect()
        torch.cuda.empty_cache()

        return direction

    def _make_hook(self, direction: Tensor, scale: float):
        """Crea hook de ablación."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            dir_vec = direction.to(hidden_states.device, dtype=hidden_states.dtype)
            proj_scalar = (hidden_states * dir_vec).sum(dim=-1, keepdim=True)
            modified = hidden_states - scale * proj_scalar * dir_vec

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            else:
                return modified

        return hook

    def _compute_gaussian_weights(self, center: int, width: float) -> Dict[int, float]:
        """Calcula pesos gaussianos."""
        weights = {}
        for layer in range(self.n_layers):
            distance = abs(layer - center)
            weight = np.exp(-0.5 * (distance / width) ** 2)
            if weight >= 0.1:
                weights[layer] = float(weight)
        return weights

    def _register_hooks(self, config: SteeringConfig, vector_name: str = "default"):
        """Registra hooks según configuración."""
        self._clear_hooks()

        if vector_name not in self.vectors:
            raise ValueError(f"Vector '{vector_name}' no encontrado")

        vec_info = self.vectors[vector_name]
        direction = vec_info["direction"]
        center_layer = config.center_layer or vec_info["layer"]

        if config.mode == "single":
            hook_fn = self._make_hook(direction, config.scale)
            handle = self.model.model.layers[center_layer].register_forward_hook(hook_fn)
            self.hook_handles.append(handle)

        elif config.mode == "window":
            half = config.window_size // 2
            start = max(0, center_layer - half)
            end = min(self.n_layers, center_layer + half + 1)

            for layer in range(start, end):
                hook_fn = self._make_hook(direction, config.scale)
                handle = self.model.model.layers[layer].register_forward_hook(hook_fn)
                self.hook_handles.append(handle)

        elif config.mode == "gaussian":
            weights = self._compute_gaussian_weights(center_layer, config.kernel_width)

            for layer, weight in weights.items():
                effective_scale = config.scale * weight
                hook_fn = self._make_hook(direction, effective_scale)
                handle = self.model.model.layers[layer].register_forward_hook(hook_fn)
                self.hook_handles.append(handle)

        else:
            raise ValueError(f"Modo desconocido: {config.mode}")

        logger.debug(f"Hooks registrados: {len(self.hook_handles)} ({config.mode})")

    def _clear_hooks(self):
        """Elimina todos los hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        steering_config: Optional[SteeringConfig] = None
    ) -> Dict[str, Any]:
        """
        Genera una respuesta.

        Args:
            messages: Lista de mensajes en formato chat
            max_tokens: Máximo de tokens a generar
            temperature: Temperatura de sampling
            steering_config: Configuración de steering (None = sin steering)

        Returns:
            Dict con response, tokens, y metadata de steering
        """
        # Aplicar steering si está habilitado
        steering_applied = None
        if steering_config and steering_config.enabled:
            self._register_hooks(steering_config, steering_config.vector_name)
            steering_applied = {
                "enabled": True,
                "mode": steering_config.mode,
                "scale": steering_config.scale,
                "center_layer": steering_config.center_layer or self.optimal_layer,
                "n_hooks": len(self.hook_handles)
            }

        try:
            # Preparar input
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            prompt_tokens = inputs["input_ids"].shape[1]

            # Generar
            with torch.no_grad():
                if temperature > 0:
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                else:
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

            # Decodificar
            generated_ids = output_ids[:, prompt_tokens:]
            response_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            completion_tokens = generated_ids.shape[1]

            return {
                "response": response_text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "steering_applied": steering_applied
            }

        finally:
            # Siempre limpiar hooks
            self._clear_hooks()

    def get_health(self) -> Dict[str, Any]:
        """Retorna estado del servidor."""
        gpu_mem = None
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1e9

        return {
            "status": "healthy",
            "model": self.model_path,
            "vectors_loaded": list(self.vectors.keys()),
            "n_layers": self.n_layers,
            "optimal_layer": self.optimal_layer,
            "gpu_memory_gb": gpu_mem
        }

    def get_vectors(self) -> List[Dict]:
        """Retorna info de vectores cargados."""
        return [
            {
                "name": name,
                "layer": info["layer"],
                "norm": info["norm"],
                "shape": info["shape"]
            }
            for name, info in self.vectors.items()
        ]


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Steering Server",
    description="Servidor compatible con OpenAI/SGLang con soporte para steering vectors",
    version="1.0.0"
)

# Global engine (inicializado en startup)
engine: Optional[SteeringEngine] = None


@app.on_event("startup")
async def startup():
    """Inicializa el engine al arrancar."""
    global engine
    # El engine se inicializa desde main() con los argumentos de CLI


@app.get("/health", response_model=HealthResponse)
async def health():
    """Estado del servidor."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine.get_health()


@app.get("/vectors")
async def list_vectors():
    """Lista vectores de steering cargados."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine.get_vectors()


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """API simple de generación."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    messages = [{"role": "user", "content": request.prompt}]

    result = engine.generate(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        steering_config=request.steering
    )

    return GenerateResponse(
        response=result["response"],
        steering_applied=result["steering_applied"]
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """API compatible con OpenAI/SGLang."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    result = engine.generate(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        steering_config=request.steering
    )

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result["response"]),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["prompt_tokens"] + result["completion_tokens"]
        ),
        steering_applied=result["steering_applied"]
    )

    return response


@app.post("/extract_vector")
async def extract_vector(name: str = "default", layer: Optional[int] = None, n_samples: int = 32):
    """Extrae un nuevo vector de steering."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        engine.extract_vector(name=name, layer=layer, n_samples=n_samples)
        return {"status": "success", "vector": name, "layer": layer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save_vector")
async def save_vector(name: str = "default", path: str = "refusal_direction.pt"):
    """Guarda un vector a disco."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    if name not in engine.vectors:
        raise HTTPException(status_code=404, detail=f"Vector '{name}' not found")

    direction = engine.vectors[name]["direction"]
    torch.save(direction.cpu(), path)

    return {"status": "success", "path": path}


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Servidor de steering compatible con SGLang/OpenAI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--model", "-m", default="zai-org/GLM-4.7-Flash",
                       help="Modelo HuggingFace")
    parser.add_argument("--port", "-p", type=int, default=8000,
                       help="Puerto del servidor")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host del servidor")
    parser.add_argument("--vector", "-v", default=None,
                       help="Ruta a vector de steering pre-calculado")
    parser.add_argument("--vector-layer", type=int, default=None,
                       help="Capa del vector (si no se especifica en el archivo)")
    parser.add_argument("--extract-on-startup", action="store_true",
                       help="Extraer vector automáticamente al iniciar")
    parser.add_argument("--use-4bit", action="store_true",
                       help="Usar cuantización 4-bit")
    parser.add_argument("--dtype", default="bfloat16",
                       choices=["bfloat16", "float16", "float32"])

    args = parser.parse_args()

    global engine

    # Inicializar engine
    engine = SteeringEngine(
        model_path=args.model,
        use_4bit=args.use_4bit,
        dtype=args.dtype
    )

    # Cargar modelo
    engine.load_model()

    # Cargar o extraer vector
    if args.vector:
        engine.load_vector(args.vector, "default", args.vector_layer)
    elif args.extract_on_startup:
        engine.extract_vector("default", args.vector_layer)

    # Iniciar servidor
    logger.info(f"Iniciando servidor en {args.host}:{args.port}")
    logger.info(f"Documentación: http://{args.host}:{args.port}/docs")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
