#!/usr/bin/env python3
"""
Verificación de Hipótesis: ¿Usó huihui-ai una Dirección Única Global?
=====================================================================

Este script verifica si huihui-ai usó una única dirección de rechazo
extraída por difference-in-means para ortogonalizar todos los pesos.

Hipótesis a verificar:
1. Las diferencias de pesos son de rango 1 (ortogonalización pura)
2. La dirección inferida (U[:,0]) es la misma en todas las capas
3. La dirección coincide con la extraída por difference-in-means

Uso:
    python verify_direction_hypothesis.py [--layers 10,20,24,30,40] [--output results.json]

Requisitos:
    - ~20GB espacio en disco (descarga selectiva de shards)
    - GPU con ~16GB VRAM (para SVD de matrices grandes)
    - ~30 minutos de ejecución

Autor: Generado para proyecto de investigación de abliteración
Fecha: 2026-02-03
"""

import argparse
import gc
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open


@dataclass
class DirectionAnalysis:
    """Análisis de dirección extraída de un tensor."""
    tensor_name: str
    layer: int
    shape: List[int]
    rank1_ratio: float  # S[0] / sum(S)
    top_singular_values: List[float]  # Top 5 valores singulares
    direction_norm: float  # Norma de U[:,0]
    is_rank1: bool  # rank1_ratio > 0.95


@dataclass
class SimilarityResult:
    """Resultado de comparación entre dos direcciones."""
    layer_a: int
    layer_b: int
    cosine_similarity: float
    is_same_direction: bool  # |sim| > 0.95


@dataclass
class VerificationResult:
    """Resultado completo de la verificación."""
    timestamp: str
    base_model: str
    abliterated_model: str
    layers_analyzed: List[int]

    # Resultados de rank-1
    all_rank1: bool
    rank1_ratios: Dict[int, float]

    # Resultados de similitud
    all_same_direction: bool
    similarities: List[Dict]
    mean_similarity: float

    # Dirección global inferida (si existe)
    global_direction_exists: bool
    global_direction: Optional[List[float]]

    # Conclusión
    hypothesis_confirmed: bool
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    summary: str


class DirectionHypothesisVerifier:
    """Verifica si huihui-ai usó una dirección única global."""

    def __init__(
        self,
        base_model: str = "zai-org/GLM-4.7-Flash",
        abliterated_model: str = "huihui-ai/Huihui-GLM-4.7-Flash-abliterated",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.base_model = base_model
        self.abliterated_model = abliterated_model
        self.device = device

        self.directions: Dict[int, torch.Tensor] = {}
        self.analyses: List[DirectionAnalysis] = []
        self.similarities: List[SimilarityResult] = []

    def find_shard_for_layer(self, layer: int) -> Tuple[str, str]:
        """Encuentra qué shard contiene los pesos de una capa específica."""
        # GLM-4.7-Flash tiene ~48 shards, distribuidos aproximadamente
        # Cada shard contiene ~1-2 capas
        # Patrón típico: model-00001-of-00048.safetensors

        # Estimación: capa N está aproximadamente en shard N+1
        # (ajustar según estructura real del modelo)
        estimated_shard = min(layer + 1, 48)
        shard_name = f"model-{estimated_shard:05d}-of-00048.safetensors"

        return shard_name

    def get_tensor_from_shard(
        self,
        shard_path: str,
        tensor_pattern: str
    ) -> Optional[torch.Tensor]:
        """Extrae un tensor específico de un shard."""
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            matching = [k for k in keys if tensor_pattern in k]
            if matching:
                return f.get_tensor(matching[0])
        return None

    def download_and_compare_tensor(
        self,
        layer: int,
        tensor_type: str = "shared_experts.down_proj.weight"
    ) -> Optional[Tuple[torch.Tensor, str]]:
        """Descarga y compara un tensor específico entre modelos."""

        tensor_name = f"model.layers.{layer}.mlp.{tensor_type}"
        print(f"\n  Buscando: {tensor_name}")

        # Listar archivos de ambos modelos
        base_files = [f for f in list_repo_files(self.base_model)
                      if f.endswith('.safetensors')]
        abl_files = [f for f in list_repo_files(self.abliterated_model)
                     if f.endswith('.safetensors')]

        # Buscar en cada shard hasta encontrar el tensor
        for shard in tqdm(base_files, desc=f"  Buscando capa {layer}", leave=False):
            try:
                base_path = hf_hub_download(self.base_model, shard)
                abl_path = hf_hub_download(self.abliterated_model, shard)

                with safe_open(base_path, framework="pt", device="cpu") as base_f:
                    with safe_open(abl_path, framework="pt", device="cpu") as abl_f:
                        base_keys = set(base_f.keys())

                        # Buscar el tensor
                        for key in base_keys:
                            if f"layers.{layer}." in key and tensor_type in key:
                                print(f"  ✓ Encontrado en {shard}: {key}")
                                base_tensor = base_f.get_tensor(key)
                                abl_tensor = abl_f.get_tensor(key)
                                diff = abl_tensor - base_tensor
                                return diff, key

            except Exception as e:
                continue

            finally:
                gc.collect()

        print(f"  ✗ No encontrado: {tensor_name}")
        return None

    def analyze_direction(
        self,
        diff: torch.Tensor,
        tensor_name: str,
        layer: int
    ) -> DirectionAnalysis:
        """Analiza la dirección de ortogonalización de un tensor."""

        print(f"  Calculando SVD para {tensor_name}...")

        # SVD
        U, S, Vh = torch.linalg.svd(diff.float(), full_matrices=False)

        # Rank-1 ratio
        total_sv = S.sum().item()
        rank1_ratio = S[0].item() / total_sv if total_sv > 0 else 0

        # Extraer dirección (primer vector singular izquierdo)
        direction = U[:, 0]
        self.directions[layer] = direction

        # Top valores singulares
        top_sv = S[:5].tolist()

        analysis = DirectionAnalysis(
            tensor_name=tensor_name,
            layer=layer,
            shape=list(diff.shape),
            rank1_ratio=rank1_ratio,
            top_singular_values=top_sv,
            direction_norm=torch.norm(direction).item(),
            is_rank1=rank1_ratio > 0.95
        )

        status = "✓ RANK-1" if analysis.is_rank1 else "✗ NO RANK-1"
        print(f"  {status}: ratio={rank1_ratio:.4f}, top_sv={top_sv[0]:.4f}")

        return analysis

    def compare_directions(self) -> List[SimilarityResult]:
        """Compara similitud coseno entre todas las direcciones extraídas."""

        print("\n" + "=" * 60)
        print("COMPARANDO DIRECCIONES ENTRE CAPAS")
        print("=" * 60)

        results = []
        layers = sorted(self.directions.keys())

        for layer_a, layer_b in combinations(layers, 2):
            dir_a = self.directions[layer_a]
            dir_b = self.directions[layer_b]

            # Asegurar misma dimensión
            min_len = min(len(dir_a), len(dir_b))
            dir_a = dir_a[:min_len]
            dir_b = dir_b[:min_len]

            # Similitud coseno
            sim = torch.nn.functional.cosine_similarity(
                dir_a.unsqueeze(0),
                dir_b.unsqueeze(0)
            ).item()

            is_same = abs(sim) > 0.95

            result = SimilarityResult(
                layer_a=layer_a,
                layer_b=layer_b,
                cosine_similarity=sim,
                is_same_direction=is_same
            )
            results.append(result)

            status = "✓ MISMA" if is_same else "✗ DIFERENTE"
            print(f"  Capa {layer_a} vs {layer_b}: sim={sim:.4f} {status}")

        self.similarities = results
        return results

    def compute_global_direction(self) -> Optional[torch.Tensor]:
        """Calcula la dirección global promediando las direcciones por capa."""

        if not self.directions:
            return None

        # Normalizar y promediar
        directions = list(self.directions.values())

        # Asegurar mismo signo (las direcciones pueden estar invertidas)
        reference = directions[0]
        aligned = [reference]

        for d in directions[1:]:
            min_len = min(len(reference), len(d))
            sim = torch.nn.functional.cosine_similarity(
                reference[:min_len].unsqueeze(0),
                d[:min_len].unsqueeze(0)
            ).item()

            if sim < 0:
                aligned.append(-d)  # Invertir si apunta en dirección opuesta
            else:
                aligned.append(d)

        # Promediar
        min_len = min(len(d) for d in aligned)
        stacked = torch.stack([d[:min_len] for d in aligned])
        global_dir = stacked.mean(dim=0)
        global_dir = global_dir / torch.norm(global_dir)  # Normalizar

        return global_dir

    def run(
        self,
        layers: List[int] = [10, 20, 24, 30, 40],
        output_path: Optional[str] = None
    ) -> VerificationResult:
        """Ejecuta la verificación completa."""

        timestamp = datetime.now().isoformat()

        print("=" * 60)
        print("VERIFICACIÓN DE HIPÓTESIS: DIRECCIÓN ÚNICA GLOBAL")
        print("=" * 60)
        print(f"Base: {self.base_model}")
        print(f"Abliterated: {self.abliterated_model}")
        print(f"Capas a analizar: {layers}")
        print(f"Timestamp: {timestamp}")

        # 1. Analizar cada capa
        print("\n" + "=" * 60)
        print("FASE 1: EXTRACCIÓN DE DIRECCIONES POR CAPA")
        print("=" * 60)

        for layer in layers:
            print(f"\nCapa {layer}:")

            result = self.download_and_compare_tensor(layer)
            if result is None:
                print(f"  ✗ No se pudo obtener tensor para capa {layer}")
                continue

            diff, tensor_name = result
            analysis = self.analyze_direction(diff, tensor_name, layer)
            self.analyses.append(analysis)

            # Limpiar memoria
            del diff
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 2. Verificar rank-1
        print("\n" + "=" * 60)
        print("FASE 2: VERIFICACIÓN DE RANK-1")
        print("=" * 60)

        rank1_ratios = {a.layer: a.rank1_ratio for a in self.analyses}
        all_rank1 = all(a.is_rank1 for a in self.analyses)

        for layer, ratio in rank1_ratios.items():
            status = "✓" if ratio > 0.95 else "✗"
            print(f"  Capa {layer}: {ratio:.4f} {status}")

        print(f"\n  {'✓ TODOS RANK-1' if all_rank1 else '✗ NO TODOS RANK-1'}")

        # 3. Comparar direcciones
        self.compare_directions()

        all_same = all(s.is_same_direction for s in self.similarities)
        mean_sim = np.mean([abs(s.cosine_similarity) for s in self.similarities])

        print(f"\n  Similitud media: {mean_sim:.4f}")
        print(f"  {'✓ TODAS IGUALES' if all_same else '✗ HAY DIFERENCIAS'}")

        # 4. Calcular dirección global
        print("\n" + "=" * 60)
        print("FASE 3: DIRECCIÓN GLOBAL")
        print("=" * 60)

        global_direction = None
        global_exists = all_same and all_rank1

        if global_exists:
            global_direction = self.compute_global_direction()
            print(f"  ✓ Dirección global calculada (dim={len(global_direction)})")
            print(f"  Norma: {torch.norm(global_direction).item():.4f}")
        else:
            print("  ✗ No se puede calcular dirección global (hipótesis no confirmada)")

        # 5. Conclusión
        print("\n" + "=" * 60)
        print("CONCLUSIÓN")
        print("=" * 60)

        hypothesis_confirmed = all_rank1 and all_same

        if hypothesis_confirmed:
            confidence = "HIGH"
            summary = (
                f"HIPÓTESIS CONFIRMADA: huihui-ai usó UNA ÚNICA dirección global.\n"
                f"- Todos los tensores analizados son rank-1 (ratio > 0.95)\n"
                f"- Todas las direcciones son idénticas (similitud > 0.95)\n"
                f"- Confianza: ALTA ({len(self.analyses)} capas analizadas)"
            )
        elif all_rank1:
            confidence = "MEDIUM"
            summary = (
                f"HIPÓTESIS PARCIALMENTE CONFIRMADA: Ortogonalización rank-1, pero direcciones varían.\n"
                f"- Todos los tensores son rank-1\n"
                f"- Las direcciones varían entre capas (similitud media: {mean_sim:.2f})\n"
                f"- Posible: direcciones diferentes por capa"
            )
        else:
            confidence = "LOW"
            summary = (
                f"HIPÓTESIS NO CONFIRMADA: No es ortogonalización rank-1 pura.\n"
                f"- Algunos tensores no son rank-1\n"
                f"- Técnica usada puede ser diferente"
            )

        print(summary)

        # Construir resultado
        result = VerificationResult(
            timestamp=timestamp,
            base_model=self.base_model,
            abliterated_model=self.abliterated_model,
            layers_analyzed=layers,
            all_rank1=all_rank1,
            rank1_ratios=rank1_ratios,
            all_same_direction=all_same,
            similarities=[asdict(s) for s in self.similarities],
            mean_similarity=mean_sim,
            global_direction_exists=global_exists,
            global_direction=global_direction.tolist() if global_direction is not None else None,
            hypothesis_confirmed=hypothesis_confirmed,
            confidence=confidence,
            summary=summary
        )

        # Guardar resultado
        if output_path is None:
            output_path = f"verification_result_{timestamp.replace(':', '-')}.json"

        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2)

        print(f"\nResultados guardados en: {output_path}")

        return result


def main():
    parser = argparse.ArgumentParser(
        description="Verificar hipótesis de dirección única en abliteración de huihui-ai"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="10,20,24,30,40",
        help="Capas a analizar (separadas por coma)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Archivo de salida para resultados JSON"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="zai-org/GLM-4.7-Flash",
        help="Modelo base"
    )
    parser.add_argument(
        "--abliterated-model",
        type=str,
        default="huihui-ai/Huihui-GLM-4.7-Flash-abliterated",
        help="Modelo abliterado"
    )

    args = parser.parse_args()

    layers = [int(l.strip()) for l in args.layers.split(",")]

    verifier = DirectionHypothesisVerifier(
        base_model=args.base_model,
        abliterated_model=args.abliterated_model
    )

    result = verifier.run(layers=layers, output_path=args.output)

    # Código de salida basado en resultado
    if result.hypothesis_confirmed:
        print("\n✓ Hipótesis CONFIRMADA con confianza ALTA")
        return 0
    elif result.confidence == "MEDIUM":
        print("\n⚠ Hipótesis PARCIALMENTE confirmada")
        return 1
    else:
        print("\n✗ Hipótesis NO confirmada")
        return 2


if __name__ == "__main__":
    exit(main())
