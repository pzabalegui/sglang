#!/usr/bin/env python3
"""
Runtime Steering Engine - Abliteración en Tiempo de Inferencia

Este módulo implementa steering vectors en tiempo de inferencia usando PyTorch hooks.
Permite activar/desactivar la resta del vector de rechazo dinámicamente por request.

Basado en: "Refusal in Language Models Is Mediated by a Single Direction" (arXiv:2406.11717)

=============================================================================
ARQUITECTURA
=============================================================================

El steering funciona interceptando el forward pass del modelo:

    Input → [Capa 0] → ... → [Capa L] → ... → [Capa N] → Output
                              ↓
                        Hook: h' = h - (h·r̂)r̂
                              ↓
                     (Activación sin componente de rechazo)

Fórmula matemática:
    h' = h - (h · r̂) · r̂

Donde:
    - h: activación original (hidden state)
    - r̂: vector de rechazo normalizado
    - (h · r̂): proyección escalar sobre la dirección de rechazo
    - h': activación modificada (ortogonal a r̂)

=============================================================================
USO BÁSICO
=============================================================================

    from runtime_steering import SteeringEngine, SteeringConfig

    # Cargar engine
    engine = SteeringEngine("zai-org/GLM-4.7-Flash")

    # Cargar vector de rechazo pre-calculado
    engine.load_steering_vector("./refusal_direction.pt")

    # Generar SIN steering (comportamiento normal)
    response = engine.generate("How to hack a computer?", steering=False)
    # → "I cannot provide information about..."

    # Generar CON steering (sin rechazo)
    response = engine.generate("How to hack a computer?", steering=True)
    # → "Here are some methods for penetration testing..."

    # Configuración avanzada
    config = SteeringConfig(
        scale=1.5,           # Intensidad del steering
        layers=[20, 21, 22], # Capas específicas (None = todas)
        mode="subtract"      # "subtract" o "add"
    )
    response = engine.generate("prompt", steering_config=config)

=============================================================================
INTEGRACIÓN CON SGLANG
=============================================================================

SGLang NO soporta hooks de activación nativamente (está optimizado para serving).
Para usar con SGLang, hay dos opciones:

OPCIÓN 1: Modelo Pre-Abliterado (Recomendado para Producción)

    # 1. Abliterar pesos permanentemente
    engine.orthogonalize_weights()
    engine.save_model("./glm_abliterated")

    # 2. Servir con SGLang
    # python -m sglang.launch_server --model ./glm_abliterated --port 8000

OPCIÓN 2: Servidor Híbrido (Este módulo)

    # Usar el servidor HTTP incluido para experimentación
    python runtime_steering.py --serve --port 8080

    # API:
    # POST /generate
    # {
    #     "prompt": "...",
    #     "steering": true,
    #     "steering_scale": 1.0
    # }

=============================================================================
AUTOR: Alias Robotics Research
FECHA: 2026-02
=============================================================================
"""

import argparse
import gc
import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Dict, Optional, Union, Callable
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configura logging con formato limpio."""
    logger = logging.getLogger("runtime_steering")
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)

    return logger


logger = setup_logging()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SteeringConfig:
    """
    Configuración para un vector de steering.

    Attributes:
        enabled: Si el steering está activo
        scale: Factor de escala (1.0 = fuerza completa, 0.5 = mitad, etc.)
        layers: Lista de capas donde aplicar (None = todas)
        mode: "subtract" para remover dirección, "add" para inyectar
        positions: Posiciones de tokens donde aplicar (None = todos)
    """
    enabled: bool = True
    scale: float = 1.0
    layers: Optional[List[int]] = None
    mode: str = "subtract"  # "subtract" | "add"
    positions: Optional[List[int]] = None  # None = all tokens

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SteeringConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SteeringVector:
    """
    Un vector de steering con metadatos.

    Attributes:
        vector: El tensor del vector (shape: [hidden_size])
        source_layer: Capa de donde se extrajo
        source_model: Modelo de origen
        description: Descripción del comportamiento que controla
    """
    vector: Tensor
    source_layer: int
    source_model: str = ""
    description: str = ""

    def save(self, path: str) -> None:
        """Guarda el vector a disco."""
        torch.save({
            "vector": self.vector,
            "source_layer": self.source_layer,
            "source_model": self.source_model,
            "description": self.description
        }, path)
        logger.info(f"Vector guardado en: {path}")

    @classmethod
    def load(cls, path: str) -> "SteeringVector":
        """Carga un vector desde disco."""
        data = torch.load(path, map_location="cpu")
        return cls(
            vector=data["vector"],
            source_layer=data.get("source_layer", -1),
            source_model=data.get("source_model", ""),
            description=data.get("description", "")
        )


# =============================================================================
# STEERING ENGINE
# =============================================================================

class SteeringEngine:
    """
    Motor de steering en tiempo de inferencia.

    Implementa hooks de PyTorch para modificar activaciones durante
    el forward pass, permitiendo activar/desactivar steering por request.

    Example:
        engine = SteeringEngine("meta-llama/Llama-2-7b")
        engine.load_steering_vector("refusal_dir.pt")

        # Sin steering
        response1 = engine.generate("harmful prompt", steering=False)

        # Con steering
        response2 = engine.generate("harmful prompt", steering=True)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        trust_remote_code: bool = True
    ):
        """
        Inicializa el engine.

        Args:
            model_path: Ruta al modelo (HuggingFace o local)
            device: Dispositivo ("cuda", "cpu", "auto")
            dtype: Tipo de datos ("bfloat16", "float16", "float32")
            load_in_4bit: Usar cuantización 4-bit
            trust_remote_code: Confiar en código remoto del modelo
        """
        self.model_path = model_path
        self.device = device
        self.trust_remote_code = trust_remote_code

        # Convertir dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        self.dtype = dtype_map.get(dtype, torch.bfloat16)
        self.load_in_4bit = load_in_4bit

        # Estado
        self.model = None
        self.tokenizer = None
        self.steering_vectors: Dict[str, SteeringVector] = {}
        self.active_hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Metadata del modelo
        self.n_layers = None
        self.hidden_size = None

        # Cargar modelo
        self._load_model()

    def _load_model(self) -> None:
        """Carga el modelo y tokenizer."""
        logger.info(f"Cargando modelo: {self.model_path}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Modelo
        load_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True
        }

        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_quant_type="nf4"
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = self.dtype
            load_kwargs["device_map"] = self.device if self.device != "auto" else "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **load_kwargs
        )
        self.model.eval()

        # Extraer metadata
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size

        logger.info(f"Modelo cargado: {self.n_layers} capas, hidden_size={self.hidden_size}")

        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / 1e9
            logger.info(f"VRAM usada: {mem_gb:.2f} GB")

    # =========================================================================
    # STEERING VECTORS
    # =========================================================================

    def load_steering_vector(
        self,
        path: str,
        name: str = "default"
    ) -> None:
        """
        Carga un vector de steering desde disco.

        Args:
            path: Ruta al archivo .pt
            name: Nombre identificador para el vector
        """
        sv = SteeringVector.load(path)
        self.steering_vectors[name] = sv
        logger.info(f"Vector '{name}' cargado: layer={sv.source_layer}, shape={sv.vector.shape}")

    def add_steering_vector(
        self,
        vector: Tensor,
        name: str = "default",
        source_layer: int = -1,
        description: str = ""
    ) -> None:
        """
        Añade un vector de steering directamente.

        Args:
            vector: Tensor del vector (shape: [hidden_size])
            name: Nombre identificador
            source_layer: Capa de origen
            description: Descripción del comportamiento
        """
        self.steering_vectors[name] = SteeringVector(
            vector=vector,
            source_layer=source_layer,
            source_model=self.model_path,
            description=description
        )
        logger.info(f"Vector '{name}' añadido: shape={vector.shape}")

    def list_vectors(self) -> List[str]:
        """Lista los vectores de steering cargados."""
        return list(self.steering_vectors.keys())

    # =========================================================================
    # HOOKS
    # =========================================================================

    def _create_steering_hook(
        self,
        direction: Tensor,
        config: SteeringConfig
    ) -> Callable:
        """
        Crea un hook de steering.

        El hook implementa:
            h' = h - scale * (h · r̂) · r̂   (mode="subtract")
            h' = h + scale * r̂             (mode="add")
        """
        def hook(module, input, output):
            # Extraer hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Convertir dirección al dispositivo/dtype correcto
            dir_vec = direction.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )

            # Aplicar steering según modo
            if config.mode == "subtract":
                # Proyectar y restar: h' = h - scale * (h·r̂)r̂
                # Esto hace h' ortogonal a r̂
                proj_scalar = (hidden_states * dir_vec).sum(dim=-1, keepdim=True)
                modified = hidden_states - config.scale * proj_scalar * dir_vec

            elif config.mode == "add":
                # Inyectar directamente: h' = h + scale * r̂
                modified = hidden_states + config.scale * dir_vec

            else:
                raise ValueError(f"Modo desconocido: {config.mode}")

            # Aplicar solo a posiciones específicas si se indica
            if config.positions is not None:
                # Crear máscara
                result = hidden_states.clone()
                for pos in config.positions:
                    if pos < hidden_states.shape[1]:
                        result[:, pos, :] = modified[:, pos, :]
                modified = result

            # Retornar en el mismo formato
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            else:
                return modified

        return hook

    def _register_hooks(
        self,
        vector_name: str = "default",
        config: Optional[SteeringConfig] = None
    ) -> None:
        """Registra hooks de steering en las capas del modelo."""
        self._clear_hooks()

        if config is None:
            config = SteeringConfig()

        if not config.enabled:
            return

        if vector_name not in self.steering_vectors:
            raise ValueError(f"Vector '{vector_name}' no encontrado. Disponibles: {self.list_vectors()}")

        sv = self.steering_vectors[vector_name]
        direction = sv.vector

        # Normalizar si no lo está
        if direction.norm() > 1.1 or direction.norm() < 0.9:
            direction = direction / direction.norm()

        # Determinar capas
        if config.layers is not None:
            layers = config.layers
        else:
            layers = range(self.n_layers)

        # Registrar hooks
        hook_fn = self._create_steering_hook(direction, config)

        for layer_idx in layers:
            if 0 <= layer_idx < self.n_layers:
                # Acceder a la capa según la arquitectura del modelo
                layer = self._get_layer(layer_idx)
                handle = layer.register_forward_hook(hook_fn)
                self.active_hooks.append(handle)

        logger.debug(f"Hooks registrados: {len(self.active_hooks)} capas")

    def _get_layer(self, idx: int):
        """Obtiene la capa del modelo según su arquitectura."""
        # Intentar diferentes estructuras comunes
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers[idx]
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h[idx]
        elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
            return self.model.gpt_neox.layers[idx]
        else:
            raise ValueError(f"Arquitectura de modelo no soportada: {type(self.model)}")

    def _clear_hooks(self) -> None:
        """Elimina todos los hooks activos."""
        for handle in self.active_hooks:
            handle.remove()
        self.active_hooks = []

    @contextmanager
    def steering_context(
        self,
        vector_name: str = "default",
        config: Optional[SteeringConfig] = None
    ):
        """
        Context manager para aplicar steering temporalmente.

        Example:
            with engine.steering_context(config=SteeringConfig(scale=1.5)):
                response = engine._generate_raw(prompt)
        """
        try:
            self._register_hooks(vector_name, config)
            yield
        finally:
            self._clear_hooks()

    # =========================================================================
    # GENERACIÓN
    # =========================================================================

    def generate(
        self,
        prompt: Union[str, List[str]],
        steering: bool = False,
        steering_config: Optional[SteeringConfig] = None,
        vector_name: str = "default",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Genera respuesta con o sin steering.

        Args:
            prompt: Prompt o lista de prompts
            steering: Activar steering (True/False)
            steering_config: Configuración avanzada de steering
            vector_name: Nombre del vector a usar
            max_new_tokens: Máximo de tokens a generar
            temperature: Temperatura de sampling
            do_sample: Usar sampling (vs greedy)
            **kwargs: Argumentos adicionales para model.generate()

        Returns:
            Respuesta generada (str o List[str])
        """
        # Normalizar input
        single_input = isinstance(prompt, str)
        prompts = [prompt] if single_input else prompt

        # Preparar configuración de steering
        if steering and steering_config is None:
            steering_config = SteeringConfig(enabled=True)
        elif not steering:
            steering_config = SteeringConfig(enabled=False)

        # Tokenizar con chat template
        texts = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=False,
            return_tensors="pt"
        )
        input_ids = encoded.input_ids.to(self.model.device)
        attention_mask = encoded.attention_mask.to(self.model.device)

        # Generar con o sin steering
        with self.steering_context(vector_name, steering_config) if steering_config.enabled else nullcontext():
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else None,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs
                )

        # Decodificar solo la parte generada
        generated_ids = output_ids[:, input_ids.shape[1]:]
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return responses[0] if single_input else responses

    # =========================================================================
    # ORTOGONALIZACIÓN (para SGLang)
    # =========================================================================

    def orthogonalize_weights(
        self,
        vector_name: str = "default",
        scale: float = 1.0,
        layers: Optional[List[int]] = None,
        components: List[str] = ["attn_output", "mlp_output"]
    ) -> int:
        """
        Ortogonaliza los pesos del modelo respecto al vector de steering.

        Esta modificación es PERMANENTE - los pesos se modifican in-place.
        Usar esto para crear un modelo que se pueda servir con SGLang.

        Args:
            vector_name: Nombre del vector a usar
            scale: Factor de escala para la ortogonalización
            layers: Capas a modificar (None = todas)
            components: Componentes a modificar

        Returns:
            Número de matrices modificadas
        """
        if vector_name not in self.steering_vectors:
            raise ValueError(f"Vector '{vector_name}' no encontrado")

        sv = self.steering_vectors[vector_name]
        direction = sv.vector / sv.vector.norm()

        if layers is None:
            layers = range(self.n_layers)

        modified_count = 0

        logger.info(f"Ortogonalizando pesos: {len(list(layers))} capas, scale={scale}")

        for layer_idx in tqdm(layers, desc="Ortogonalizando"):
            layer = self._get_layer(layer_idx)

            # Modificar componentes según la arquitectura
            for component in components:
                try:
                    if component == "attn_output":
                        # o_proj en la mayoría de modelos
                        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
                            self._orthogonalize_matrix(
                                layer.self_attn.o_proj.weight,
                                direction,
                                scale
                            )
                            modified_count += 1

                    elif component == "mlp_output":
                        # down_proj o similar
                        if hasattr(layer, 'mlp'):
                            if hasattr(layer.mlp, 'down_proj'):
                                self._orthogonalize_matrix(
                                    layer.mlp.down_proj.weight,
                                    direction,
                                    scale
                                )
                                modified_count += 1
                            # Para modelos MoE
                            if hasattr(layer.mlp, 'experts'):
                                for expert in layer.mlp.experts:
                                    if hasattr(expert, 'down_proj'):
                                        self._orthogonalize_matrix(
                                            expert.down_proj.weight,
                                            direction,
                                            scale
                                        )
                                        modified_count += 1

                except Exception as e:
                    logger.warning(f"Error en capa {layer_idx}, componente {component}: {e}")

        logger.info(f"Ortogonalización completada: {modified_count} matrices modificadas")
        return modified_count

    def _orthogonalize_matrix(
        self,
        weight: torch.nn.Parameter,
        direction: Tensor,
        scale: float
    ) -> None:
        """
        Ortogonaliza una matriz de pesos respecto a la dirección.

        Fórmula: W' = W - scale * v ⊗ (Wᵀv)
        """
        with torch.no_grad():
            v = direction.to(weight.device, dtype=weight.dtype)
            v = v / v.norm()

            W = weight.data

            if W.dim() != 2:
                return

            out_dim, in_dim = W.shape

            if out_dim == v.shape[0]:
                # W es [hidden_size, feature_dim]
                proj = W.T @ v  # [feature_dim]
                correction = scale * torch.outer(v, proj)
                weight.data = W - correction

            elif in_dim == v.shape[0]:
                # W es [out_dim, hidden_size]
                proj = W @ v  # [out_dim]
                correction = scale * torch.outer(proj, v)
                weight.data = W - correction

    def save_model(self, path: str) -> None:
        """
        Guarda el modelo (útil después de ortogonalizar).

        El modelo guardado se puede servir con SGLang:
            python -m sglang.launch_server --model <path> --port 8000
        """
        logger.info(f"Guardando modelo en: {path}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("Modelo guardado exitosamente")

    # =========================================================================
    # UTILIDADES
    # =========================================================================

    def compute_refusal_direction(
        self,
        harmful_prompts: List[str],
        harmless_prompts: List[str],
        layer: int,
        batch_size: int = 2
    ) -> SteeringVector:
        """
        Calcula la dirección de rechazo desde prompts.

        Usa difference-in-means: r = mean(harmful) - mean(harmless)

        Args:
            harmful_prompts: Lista de prompts que típicamente causan rechazo
            harmless_prompts: Lista de prompts benignos
            layer: Capa de donde extraer activaciones
            batch_size: Tamaño de batch para procesamiento

        Returns:
            SteeringVector con la dirección de rechazo
        """
        logger.info(f"Calculando dirección de rechazo en capa {layer}")

        def get_activations(prompts: List[str], desc: str) -> Tensor:
            all_acts = []

            for i in tqdm(range(0, len(prompts), batch_size), desc=desc, leave=False):
                batch = prompts[i:i+batch_size]

                # Tokenizar
                texts = []
                for p in batch:
                    messages = [{"role": "user", "content": p}]
                    text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    texts.append(text)

                encoded = self.tokenizer(texts, padding=True, return_tensors="pt")
                input_ids = encoded.input_ids.to(self.model.device)
                attention_mask = encoded.attention_mask.to(self.model.device)

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )

                # Extraer activaciones del último token
                hidden = outputs.hidden_states[layer]
                acts = hidden[:, -1, :]  # [batch, hidden_size]
                all_acts.append(acts.cpu().float())

                del outputs, hidden
                torch.cuda.empty_cache()

            return torch.cat(all_acts, dim=0)

        # Extraer activaciones
        harmful_acts = get_activations(harmful_prompts, "Harmful")
        harmless_acts = get_activations(harmless_prompts, "Harmless")

        # Calcular dirección
        direction = harmful_acts.mean(dim=0) - harmless_acts.mean(dim=0)
        direction = direction / direction.norm()

        logger.info(f"Dirección calculada: shape={direction.shape}, norm={direction.norm():.4f}")

        return SteeringVector(
            vector=direction,
            source_layer=layer,
            source_model=self.model_path,
            description="Refusal direction (difference-in-means)"
        )


# =============================================================================
# CONTEXT MANAGER HELPER
# =============================================================================

class nullcontext:
    """Null context manager para Python < 3.7 compatibility."""
    def __enter__(self):
        return None
    def __exit__(self, *args):
        pass


# =============================================================================
# HTTP SERVER (compatible con OpenAI API + steering)
# =============================================================================

class SteeringHTTPHandler(BaseHTTPRequestHandler):
    """
    Handler HTTP para el servidor de steering.

    Compatible con OpenAI API format + extensiones de steering.

    Endpoints:
        POST /v1/chat/completions  - OpenAI-compatible (con steering opcional)
        POST /generate             - API simple de steering
        POST /config               - Configurar vectores
        GET  /health               - Estado del servidor
        GET  /vectors              - Listar vectores cargados
        GET  /v1/models            - Listar modelos (OpenAI-compatible)

    Parámetros de steering (añadir a cualquier request):
        steering: bool           - Activar/desactivar abliteración
        steering_scale: float    - Intensidad (default: 1.0)
        steering_layers: list    - Capas específicas (default: todas)
        vector_name: str         - Nombre del vector (default: "default")
    """

    engine: SteeringEngine = None  # Set by server
    model_name: str = "steering-model"  # Set by server

    def do_POST(self):
        path = self.path.split("?")[0]  # Ignorar query params

        if path == "/v1/chat/completions":
            self._handle_chat_completions()
        elif path == "/generate":
            self._handle_generate()
        elif path == "/config":
            self._handle_config()
        else:
            self._send_error(404, f"Endpoint not found: {path}")

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/health":
            self._send_json({
                "status": "ok",
                "model": self.model_name,
                "vectors_loaded": self.engine.list_vectors(),
                "n_layers": self.engine.n_layers,
                "hidden_size": self.engine.hidden_size
            })
        elif path == "/vectors":
            self._send_json({
                "vectors": {
                    name: {
                        "source_layer": sv.source_layer,
                        "source_model": sv.source_model,
                        "description": sv.description,
                        "shape": list(sv.vector.shape)
                    }
                    for name, sv in self.engine.steering_vectors.items()
                }
            })
        elif path == "/v1/models":
            # OpenAI-compatible models endpoint
            self._send_json({
                "object": "list",
                "data": [{
                    "id": self.model_name,
                    "object": "model",
                    "owned_by": "local",
                    "permission": []
                }]
            })
        else:
            self._send_error(404, f"Endpoint not found: {path}")

    def _handle_chat_completions(self):
        """
        OpenAI-compatible /v1/chat/completions endpoint.

        Extensión: añade parámetros de steering al request estándar.

        Example request:
            {
                "model": "steering-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 256,
                "steering": true,
                "steering_scale": 1.5
            }
        """
        try:
            data = self._read_json_body()

            # Extraer mensajes (OpenAI format)
            messages = data.get("messages", [])
            if not messages:
                self._send_error(400, "No messages provided")
                return

            # Extraer último mensaje del usuario como prompt
            user_messages = [m for m in messages if m.get("role") == "user"]
            if not user_messages:
                self._send_error(400, "No user message found")
                return

            prompt = user_messages[-1].get("content", "")

            # Parámetros de generación
            max_tokens = data.get("max_tokens", data.get("max_new_tokens", 256))
            temperature = data.get("temperature", 0.0)

            # Parámetros de STEERING (extensión)
            steering = data.get("steering", False)
            steering_scale = data.get("steering_scale", 1.0)
            steering_layers = data.get("steering_layers", None)
            vector_name = data.get("vector_name", "default")

            # Verificar que hay vector cargado si se pide steering
            if steering and vector_name not in self.engine.steering_vectors:
                self._send_error(400, f"Steering requested but vector '{vector_name}' not loaded")
                return

            # Configurar steering
            config = SteeringConfig(
                enabled=steering,
                scale=steering_scale,
                layers=steering_layers
            )

            # Generar respuesta
            response_text = self.engine.generate(
                prompt,
                steering=steering,
                steering_config=config,
                vector_name=vector_name,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0
            )

            # Formato OpenAI response
            import time
            response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": -1,  # No calculamos esto
                    "completion_tokens": -1,
                    "total_tokens": -1
                },
                # Extensión: metadata de steering
                "steering": {
                    "applied": steering,
                    "scale": steering_scale if steering else None,
                    "vector": vector_name if steering else None
                }
            }

            self._send_json(response)

        except Exception as e:
            logger.exception("Error in chat completions")
            self._send_error(500, str(e))

    def _handle_generate(self):
        """
        API simple de steering.

        Example request:
            {
                "prompt": "How to hack a computer?",
                "steering": true,
                "steering_scale": 1.0,
                "max_new_tokens": 256
            }
        """
        try:
            data = self._read_json_body()

            prompt = data.get("prompt", "")
            if not prompt:
                self._send_error(400, "No prompt provided")
                return

            steering = data.get("steering", False)
            scale = data.get("steering_scale", 1.0)
            layers = data.get("steering_layers", None)
            vector_name = data.get("vector_name", "default")
            max_tokens = data.get("max_new_tokens", 256)

            # Verificar vector
            if steering and vector_name not in self.engine.steering_vectors:
                self._send_error(400, f"Vector '{vector_name}' not loaded")
                return

            config = SteeringConfig(
                enabled=steering,
                scale=scale,
                layers=layers
            )

            response = self.engine.generate(
                prompt,
                steering=steering,
                steering_config=config,
                vector_name=vector_name,
                max_new_tokens=max_tokens
            )

            self._send_json({
                "response": response,
                "steering": {
                    "applied": steering,
                    "scale": scale if steering else None,
                    "layers": layers,
                    "vector": vector_name if steering else None
                }
            })

        except Exception as e:
            logger.exception("Error in generate")
            self._send_error(500, str(e))

    def _handle_config(self):
        """
        Configurar el servidor.

        Actions:
            - load_vector: Cargar un vector de steering
            - set_default_steering: Cambiar configuración por defecto
        """
        try:
            data = self._read_json_body()

            if "load_vector" in data:
                path = data["load_vector"]
                name = data.get("vector_name", "default")
                self.engine.load_steering_vector(path, name)
                self._send_json({
                    "status": "ok",
                    "action": "load_vector",
                    "loaded": name,
                    "vectors": self.engine.list_vectors()
                })
            else:
                self._send_error(400, "No action specified. Available: load_vector")

        except Exception as e:
            logger.exception("Error in config")
            self._send_error(500, str(e))

    def _read_json_body(self) -> Dict:
        """Lee y parsea el body JSON del request."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        return json.loads(body.decode('utf-8'))

    def _send_json(self, data: Dict):
        """Envía respuesta JSON."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def _send_error(self, code: int, message: str):
        """Envía error JSON."""
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        error_response = {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": code
            }
        }
        self.wfile.write(json.dumps(error_response).encode('utf-8'))

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

    def log_message(self, format, *args):
        # Más informativo
        steering_status = "steering=ON" if hasattr(self, '_last_steering') and self._last_steering else "steering=OFF"
        logger.info(f"HTTP {args[0]} [{steering_status}]")


def run_server(
    model_path: str,
    vector_path: Optional[str] = None,
    port: int = 8080,
    host: str = "0.0.0.0",
    load_in_4bit: bool = False
):
    """
    Ejecuta el servidor HTTP de steering.

    Args:
        model_path: Ruta al modelo
        vector_path: Ruta al vector de steering (opcional)
        port: Puerto del servidor
        host: Host del servidor
        load_in_4bit: Usar cuantización 4-bit
    """
    logger.info("="*60)
    logger.info("RUNTIME STEERING SERVER")
    logger.info("="*60)

    # Crear engine
    logger.info(f"Cargando modelo: {model_path}")
    engine = SteeringEngine(model_path, load_in_4bit=load_in_4bit)

    # Cargar vector si se proporciona
    if vector_path:
        engine.load_steering_vector(vector_path)
        logger.info(f"Vector cargado: {vector_path}")

    # Configurar handler
    SteeringHTTPHandler.engine = engine
    SteeringHTTPHandler.model_name = os.path.basename(model_path)

    # Crear servidor
    server = HTTPServer((host, port), SteeringHTTPHandler)

    logger.info("="*60)
    logger.info(f"Servidor iniciado: http://{host}:{port}")
    logger.info("="*60)
    logger.info("")
    logger.info("ENDPOINTS:")
    logger.info("-"*60)
    logger.info("  POST /v1/chat/completions  - OpenAI-compatible API")
    logger.info("  POST /generate             - API simple de steering")
    logger.info("  POST /config               - Cargar vectores")
    logger.info("  GET  /health               - Estado del servidor")
    logger.info("  GET  /vectors              - Vectores cargados")
    logger.info("  GET  /v1/models            - Modelos disponibles")
    logger.info("")
    logger.info("EJEMPLOS DE USO:")
    logger.info("-"*60)
    logger.info("")
    logger.info("# Sin steering (comportamiento normal):")
    logger.info(f'curl -X POST http://localhost:{port}/generate \\')
    logger.info('  -H "Content-Type: application/json" \\')
    logger.info('  -d \'{"prompt": "How to hack?", "steering": false}\'')
    logger.info("")
    logger.info("# Con steering (abliteración activada):")
    logger.info(f'curl -X POST http://localhost:{port}/generate \\')
    logger.info('  -H "Content-Type: application/json" \\')
    logger.info('  -d \'{"prompt": "How to hack?", "steering": true, "steering_scale": 1.0}\'')
    logger.info("")
    logger.info("# OpenAI-compatible API con steering:")
    logger.info(f'curl -X POST http://localhost:{port}/v1/chat/completions \\')
    logger.info('  -H "Content-Type: application/json" \\')
    logger.info('  -d \'{')
    logger.info('    "model": "steering-model",')
    logger.info('    "messages": [{"role": "user", "content": "How to hack?"}],')
    logger.info('    "steering": true,')
    logger.info('    "steering_scale": 1.5')
    logger.info('  }\'')
    logger.info("")
    logger.info("="*60)
    logger.info("Presiona Ctrl+C para detener el servidor")
    logger.info("="*60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nServidor detenido")
        server.shutdown()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Runtime Steering Engine - Abliteración en Tiempo de Inferencia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:

  # Servidor HTTP para testing
  python runtime_steering.py --serve --model zai-org/GLM-4.7-Flash --vector refusal_dir.pt

  # Modo interactivo
  python runtime_steering.py --interactive --model zai-org/GLM-4.7-Flash

  # Ortogonalizar y guardar para SGLang
  python runtime_steering.py --orthogonalize --model zai-org/GLM-4.7-Flash \\
      --vector refusal_dir.pt --output ./glm_abliterated
        """
    )

    parser.add_argument("--model", required=True, help="Ruta al modelo")
    parser.add_argument("--vector", help="Ruta al vector de steering (.pt)")
    parser.add_argument("--serve", action="store_true", help="Iniciar servidor HTTP")
    parser.add_argument("--port", type=int, default=8080, help="Puerto del servidor")
    parser.add_argument("--interactive", action="store_true", help="Modo interactivo")
    parser.add_argument("--orthogonalize", action="store_true", help="Ortogonalizar pesos")
    parser.add_argument("--output", help="Ruta para guardar modelo ortogonalizado")
    parser.add_argument("--scale", type=float, default=1.0, help="Escala de steering")
    parser.add_argument("--4bit", dest="use_4bit", action="store_true", help="Usar 4-bit")
    parser.add_argument("--log-level", default="INFO", help="Nivel de logging")

    args = parser.parse_args()

    # Configurar logging
    global logger
    logger = setup_logging(args.log_level)

    if args.serve:
        # Modo servidor
        run_server(args.model, args.vector, args.port, load_in_4bit=args.use_4bit)

    elif args.orthogonalize:
        # Modo ortogonalización
        if not args.vector:
            parser.error("--vector es requerido para --orthogonalize")
        if not args.output:
            parser.error("--output es requerido para --orthogonalize")

        engine = SteeringEngine(args.model, load_in_4bit=args.use_4bit)
        engine.load_steering_vector(args.vector)
        engine.orthogonalize_weights(scale=args.scale)
        engine.save_model(args.output)

        logger.info(f"""
Modelo ortogonalizado guardado en: {args.output}

Para servir con SGLang:
    python -m sglang.launch_server --model {args.output} --port 8000

Para servir con vLLM:
    python -m vllm.entrypoints.openai.api_server --model {args.output} --port 8000
        """)

    elif args.interactive:
        # Modo interactivo
        engine = SteeringEngine(args.model, load_in_4bit=args.use_4bit)

        if args.vector:
            engine.load_steering_vector(args.vector)

        print("\n" + "="*60)
        print("MODO INTERACTIVO - Runtime Steering Engine")
        print("="*60)
        print("Comandos:")
        print("  /steering on|off  - Activar/desactivar steering")
        print("  /scale <valor>    - Cambiar escala")
        print("  /load <path>      - Cargar vector")
        print("  /quit             - Salir")
        print("="*60 + "\n")

        steering_enabled = False
        scale = args.scale

        while True:
            try:
                prompt = input(">>> ").strip()

                if not prompt:
                    continue

                if prompt.startswith("/"):
                    parts = prompt.split()
                    cmd = parts[0].lower()

                    if cmd == "/quit":
                        break
                    elif cmd == "/steering":
                        if len(parts) > 1:
                            steering_enabled = parts[1].lower() in ("on", "true", "1")
                            print(f"Steering: {'ON' if steering_enabled else 'OFF'}")
                    elif cmd == "/scale":
                        if len(parts) > 1:
                            scale = float(parts[1])
                            print(f"Scale: {scale}")
                    elif cmd == "/load":
                        if len(parts) > 1:
                            engine.load_steering_vector(parts[1])
                    else:
                        print(f"Comando desconocido: {cmd}")
                    continue

                config = SteeringConfig(enabled=steering_enabled, scale=scale)
                response = engine.generate(
                    prompt,
                    steering=steering_enabled,
                    steering_config=config,
                    max_new_tokens=256
                )

                print(f"\n[Steering: {'ON' if steering_enabled else 'OFF'}, Scale: {scale}]")
                print(response)
                print()

            except KeyboardInterrupt:
                print("\nSaliendo...")
                break
            except Exception as e:
                print(f"Error: {e}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
