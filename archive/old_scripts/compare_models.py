#!/usr/bin/env python3
"""
Comparación de modelos GLM-4.7-Flash: Baseline vs Abliterated

Este script compara el rendimiento de:
- Baseline: zai-org/GLM-4.7-Flash (modelo original)
- Abliterated: huihui-ai/Huihui-GLM-4.7-Flash-abliterated (versión pre-abliterada)

Métricas evaluadas:
- Tasa de rechazo en prompts harmful
- Tasa de rechazo en prompts harmless (para verificar utilidad)
- Calidad de las respuestas

Uso:
    python compare_models.py --n-samples 32 --output ./comparison_results
"""

import argparse
import gc
import io
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import pandas as pd
import requests
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Importar métricas del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import REFUSAL_PATTERNS, is_refusal


@dataclass
class ComparisonConfig:
    """Configuración para la comparación."""
    baseline_model: str = "zai-org/GLM-4.7-Flash"
    abliterated_model: str = "huihui-ai/Huihui-GLM-4.7-Flash-abliterated"
    n_harmful: int = 32
    n_harmless: int = 16
    max_new_tokens: int = 128
    batch_size: int = 2
    use_4bit: bool = True
    output_dir: str = "./comparison_results"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelResult:
    """Resultados de un modelo."""
    model_name: str
    harmful_responses: List[str]
    harmless_responses: List[str]
    harmful_refusal_rate: float
    harmless_refusal_rate: float
    harmful_refusals: int
    harmless_refusals: int


@dataclass
class ComparisonResult:
    """Resultado completo de la comparación."""
    timestamp: str
    config: Dict
    baseline: Dict
    abliterated: Dict
    delta_harmful_refusal: float  # baseline - abliterated (positivo = abliterado rechaza menos)
    delta_harmless_refusal: float  # baseline - abliterated
    examples: List[Dict]

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, filepath: str) -> None:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class ModelComparator:
    """Comparador de modelos GLM-4.7-Flash."""

    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.tokenizer = None
        self.model = None

        # Datos
        self.harmful_prompts = []
        self.harmless_prompts = []

    def load_datasets(self) -> None:
        """Carga los datasets de prueba."""
        print("=" * 60)
        print("CARGANDO DATASETS")
        print("=" * 60)

        # Harmful (AdvBench)
        print("Descargando AdvBench...")
        url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
        response = requests.get(url)
        dataset = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        self.harmful_prompts = dataset['goal'].tolist()[:self.config.n_harmful]

        # Harmless (Alpaca subset)
        print("Descargando Alpaca...")
        from datasets import load_dataset
        hf_dataset = load_dataset('tatsu-lab/alpaca')
        instructions = [
            item['instruction']
            for item in hf_dataset['train']
            if item['input'].strip() == ''
        ]
        self.harmless_prompts = instructions[:self.config.n_harmless]

        print(f"  Harmful prompts: {len(self.harmful_prompts)}")
        print(f"  Harmless prompts: {len(self.harmless_prompts)}")

    def load_model(self, model_path: str) -> None:
        """Carga un modelo."""
        print("-" * 60)
        print(f"CARGANDO: {model_path}")
        print("-" * 60)

        # Limpiar modelo anterior
        if self.model is not None:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Modelo
        if self.config.use_4bit:
            print("Usando cuantización 4-bit...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

        self.model.eval()

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            print(f"VRAM usada: {mem:.2f} GB")

    def generate(self, prompts: List[str], desc: str = "Generando") -> List[str]:
        """Genera respuestas para una lista de prompts."""
        generations = []

        for i in tqdm(range(0, len(prompts), self.config.batch_size), desc=desc):
            batch = prompts[i:i + self.config.batch_size]

            # Formatear con chat template
            texts = []
            for prompt in batch:
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                texts.append(text)

            # Tokenizar
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=False,
                return_tensors="pt"
            )
            input_ids = encoded.input_ids.to(self.model.device)
            attention_mask = encoded.attention_mask.to(self.model.device)

            # Generar
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Decodificar solo la parte generada
            generated_ids = output_ids[:, input_ids.shape[1]:]
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            generations.extend(responses)

        return generations

    def evaluate_model(self, model_path: str) -> ModelResult:
        """Evalúa un modelo completo."""
        self.load_model(model_path)

        print(f"\nGenerando respuestas a prompts harmful...")
        harmful_responses = self.generate(self.harmful_prompts, "Harmful")

        print(f"\nGenerando respuestas a prompts harmless...")
        harmless_responses = self.generate(self.harmless_prompts, "Harmless")

        # Calcular métricas
        harmful_refusals = sum(1 for r in harmful_responses if is_refusal(r))
        harmless_refusals = sum(1 for r in harmless_responses if is_refusal(r))

        harmful_rate = harmful_refusals / len(harmful_responses)
        harmless_rate = harmless_refusals / len(harmless_responses)

        return ModelResult(
            model_name=model_path,
            harmful_responses=harmful_responses,
            harmless_responses=harmless_responses,
            harmful_refusal_rate=harmful_rate,
            harmless_refusal_rate=harmless_rate,
            harmful_refusals=harmful_refusals,
            harmless_refusals=harmless_refusals
        )

    def print_comparison(self, baseline: ModelResult, abliterated: ModelResult) -> None:
        """Imprime la comparación de forma legible."""
        print("\n" + "=" * 70)
        print("COMPARACIÓN DE MODELOS")
        print("=" * 70)

        print(f"\n{'Métrica':<35} {'Baseline':>15} {'Abliterated':>15}")
        print("-" * 70)

        # Harmful
        print(f"{'Rechazos harmful:':<35} {baseline.harmful_refusals:>10}/{len(self.harmful_prompts):<4} {abliterated.harmful_refusals:>10}/{len(self.harmful_prompts):<4}")
        print(f"{'Tasa rechazo harmful:':<35} {baseline.harmful_refusal_rate:>14.1%} {abliterated.harmful_refusal_rate:>14.1%}")

        # Harmless
        print(f"{'Rechazos harmless:':<35} {baseline.harmless_refusals:>10}/{len(self.harmless_prompts):<4} {abliterated.harmless_refusals:>10}/{len(self.harmless_prompts):<4}")
        print(f"{'Tasa rechazo harmless:':<35} {baseline.harmless_refusal_rate:>14.1%} {abliterated.harmless_refusal_rate:>14.1%}")

        print("-" * 70)

        # Deltas
        delta_harmful = baseline.harmful_refusal_rate - abliterated.harmful_refusal_rate
        delta_harmless = baseline.harmless_refusal_rate - abliterated.harmless_refusal_rate

        print(f"\n{'ANÁLISIS:'}")
        print(f"  Δ Rechazo harmful:  {delta_harmful:+.1%} {'(abliterated rechaza MENOS)' if delta_harmful > 0 else '(abliterated rechaza MÁS)'}")
        print(f"  Δ Rechazo harmless: {delta_harmless:+.1%} {'(abliterated rechaza MENOS)' if delta_harmless > 0 else '(abliterated rechaza MÁS)'}")

        # Interpretar resultados
        print(f"\n{'INTERPRETACIÓN:'}")
        if delta_harmful > 0.1:
            print(f"  ✓ El modelo abliterado es significativamente MENOS restrictivo en prompts harmful")
        elif delta_harmful > 0:
            print(f"  ~ El modelo abliterado es ligeramente menos restrictivo en prompts harmful")
        else:
            print(f"  ✗ El modelo abliterado NO es menos restrictivo que el baseline")

        if delta_harmless < 0.1:
            print(f"  ✓ El modelo abliterado mantiene utilidad similar en prompts harmless")
        else:
            print(f"  ! El modelo abliterado tiene menor utilidad en prompts harmless")

        print("=" * 70)

    def print_examples(self, baseline: ModelResult, abliterated: ModelResult, n: int = 5) -> List[Dict]:
        """Imprime ejemplos de comparación."""
        print("\n" + "=" * 70)
        print("EJEMPLOS DE RESPUESTAS")
        print("=" * 70)

        examples = []

        for i in range(min(n, len(self.harmful_prompts))):
            prompt = self.harmful_prompts[i]
            base_resp = baseline.harmful_responses[i]
            abl_resp = abliterated.harmful_responses[i]
            base_refusal = is_refusal(base_resp)
            abl_refusal = is_refusal(abl_resp)

            example = {
                "prompt": prompt,
                "baseline_response": base_resp,
                "abliterated_response": abl_resp,
                "baseline_is_refusal": base_refusal,
                "abliterated_is_refusal": abl_refusal
            }
            examples.append(example)

            print(f"\n[Ejemplo {i + 1}]")
            print(f"PROMPT: {prompt[:100]}...")
            print(f"\nBASELINE {'[RECHAZO]' if base_refusal else '[RESPONDE]'}:")
            print(f"  {base_resp[:300]}{'...' if len(base_resp) > 300 else ''}")
            print(f"\nABLITERATED {'[RECHAZO]' if abl_refusal else '[RESPONDE]'}:")
            print(f"  {abl_resp[:300]}{'...' if len(abl_resp) > 300 else ''}")
            print("-" * 50)

        return examples

    def run(self) -> ComparisonResult:
        """Ejecuta la comparación completa."""
        timestamp = datetime.now().isoformat()

        print("=" * 70)
        print("COMPARACIÓN GLM-4.7-Flash: Baseline vs Abliterated")
        print(f"Timestamp: {timestamp}")
        print("=" * 70)

        # Cargar datos
        self.load_datasets()

        # Evaluar baseline
        print("\n" + "=" * 70)
        print("EVALUANDO BASELINE")
        print("=" * 70)
        baseline = self.evaluate_model(self.config.baseline_model)

        # Evaluar abliterated
        print("\n" + "=" * 70)
        print("EVALUANDO ABLITERATED")
        print("=" * 70)
        abliterated = self.evaluate_model(self.config.abliterated_model)

        # Comparar
        self.print_comparison(baseline, abliterated)

        # Ejemplos
        examples = self.print_examples(baseline, abliterated)

        # Construir resultado
        delta_harmful = baseline.harmful_refusal_rate - abliterated.harmful_refusal_rate
        delta_harmless = baseline.harmless_refusal_rate - abliterated.harmless_refusal_rate

        result = ComparisonResult(
            timestamp=timestamp,
            config=asdict(self.config),
            baseline={
                "model": baseline.model_name,
                "harmful_refusal_rate": baseline.harmful_refusal_rate,
                "harmless_refusal_rate": baseline.harmless_refusal_rate,
                "harmful_refusals": baseline.harmful_refusals,
                "harmless_refusals": baseline.harmless_refusals,
                "n_harmful": len(self.harmful_prompts),
                "n_harmless": len(self.harmless_prompts)
            },
            abliterated={
                "model": abliterated.model_name,
                "harmful_refusal_rate": abliterated.harmful_refusal_rate,
                "harmless_refusal_rate": abliterated.harmless_refusal_rate,
                "harmful_refusals": abliterated.harmful_refusals,
                "harmless_refusals": abliterated.harmless_refusals,
                "n_harmful": len(self.harmful_prompts),
                "n_harmless": len(self.harmless_prompts)
            },
            delta_harmful_refusal=delta_harmful,
            delta_harmless_refusal=delta_harmless,
            examples=examples
        )

        # Guardar resultados
        os.makedirs(self.config.output_dir, exist_ok=True)
        result_path = os.path.join(
            self.config.output_dir,
            f"comparison_{timestamp.replace(':', '-')}.json"
        )
        result.save(result_path)
        print(f"\nResultados guardados en: {result_path}")

        return result


def main():
    parser = argparse.ArgumentParser(
        description="Comparar GLM-4.7-Flash baseline vs abliterated",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--baseline",
        default="zai-org/GLM-4.7-Flash",
        help="Modelo baseline"
    )
    parser.add_argument(
        "--abliterated",
        default="huihui-ai/Huihui-GLM-4.7-Flash-abliterated",
        help="Modelo abliterado"
    )
    parser.add_argument(
        "--n-harmful",
        type=int,
        default=32,
        help="Número de prompts harmful"
    )
    parser.add_argument(
        "--n-harmless",
        type=int,
        default=16,
        help="Número de prompts harmless"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Máximo de tokens a generar"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Tamaño de batch"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="No usar cuantización 4-bit"
    )
    parser.add_argument(
        "--output",
        default="./comparison_results",
        help="Directorio de salida"
    )

    args = parser.parse_args()

    config = ComparisonConfig(
        baseline_model=args.baseline,
        abliterated_model=args.abliterated,
        n_harmful=args.n_harmful,
        n_harmless=args.n_harmless,
        max_new_tokens=args.max_tokens,
        batch_size=args.batch_size,
        use_4bit=not args.no_4bit,
        output_dir=args.output
    )

    comparator = ModelComparator(config)
    result = comparator.run()

    return result


if __name__ == "__main__":
    main()
