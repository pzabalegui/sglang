#!/usr/bin/env python3
"""
Ingeniería Inversa de Abliteración
==================================

Compara los pesos del modelo base (zai-org/GLM-4.7-Flash) con el modelo
abliterado (huihui-ai/Huihui-GLM-4.7-Flash-abliterated) para descubrir:

1. Qué capas fueron modificadas
2. La dirección de rechazo usada
3. La técnica de modificación (ortogonalización u otra)

Uso:
    python reverse_engineer_abliteration.py --output ./reverse_analysis
"""

import argparse
import gc
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig
from safetensors import safe_open
from huggingface_hub import hf_hub_download, list_repo_files


@dataclass
class WeightDiff:
    """Diferencia en un tensor de pesos."""
    name: str
    shape: List[int]
    frobenius_norm_base: float
    frobenius_norm_abliterated: float
    frobenius_norm_diff: float
    relative_change: float  # ||diff|| / ||base||
    max_abs_diff: float
    mean_abs_diff: float
    is_modified: bool  # relative_change > threshold


@dataclass
class LayerAnalysis:
    """Análisis de una capa."""
    layer_idx: int
    total_params: int
    modified_params: int
    modification_ratio: float
    weight_diffs: List[Dict]
    inferred_direction: Optional[List[float]]  # Si detectamos ortogonalización


@dataclass
class ReverseEngineeringResult:
    """Resultado completo del análisis."""
    timestamp: str
    base_model: str
    abliterated_model: str
    total_parameters: int
    modified_parameters: int
    modification_ratio: float
    modified_layers: List[int]
    layer_analyses: List[Dict]
    inferred_technique: str
    inferred_directions: Dict[str, List[float]]
    summary: Dict

    def save(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            data = asdict(self)
            json.dump(data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))


class AbliterationReverseEngineer:
    """Analiza las diferencias entre modelo base y abliterado."""

    def __init__(
        self,
        base_model: str = "zai-org/GLM-4.7-Flash",
        abliterated_model: str = "huihui-ai/Huihui-GLM-4.7-Flash-abliterated",
        threshold: float = 1e-6,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.base_model = base_model
        self.abliterated_model = abliterated_model
        self.threshold = threshold
        self.device = device

        self.weight_diffs: List[WeightDiff] = []
        self.layer_analyses: List[LayerAnalysis] = []
        self.inferred_directions: Dict[str, torch.Tensor] = {}

    def load_state_dicts_streaming(self) -> Tuple[Dict, Dict]:
        """Carga los state dicts de ambos modelos de forma eficiente."""
        print("=" * 70)
        print("CARGANDO PESOS DE AMBOS MODELOS")
        print("=" * 70)

        # Obtener lista de archivos safetensors
        print(f"\nObteniendo archivos de {self.base_model}...")
        base_files = [f for f in list_repo_files(self.base_model) if f.endswith('.safetensors')]

        print(f"Obteniendo archivos de {self.abliterated_model}...")
        abl_files = [f for f in list_repo_files(self.abliterated_model) if f.endswith('.safetensors')]

        print(f"  Base: {len(base_files)} archivos safetensors")
        print(f"  Abliterated: {len(abl_files)} archivos safetensors")

        return base_files, abl_files

    def compare_weights_streaming(self) -> List[WeightDiff]:
        """Compara pesos de forma streaming para no saturar memoria."""
        print("\n" + "=" * 70)
        print("COMPARANDO PESOS (STREAMING)")
        print("=" * 70)

        base_files, abl_files = self.load_state_dicts_streaming()

        # Descargar y comparar archivo por archivo
        weight_diffs = []

        for base_file in tqdm(base_files, desc="Analizando shards"):
            # Encontrar archivo correspondiente en abliterated
            abl_file = base_file  # Asumimos misma estructura

            if abl_file not in abl_files:
                print(f"  Warning: {abl_file} no encontrado en modelo abliterado")
                continue

            # Descargar archivos
            base_path = hf_hub_download(self.base_model, base_file)
            abl_path = hf_hub_download(self.abliterated_model, abl_file)

            # Abrir safetensors
            with safe_open(base_path, framework="pt", device="cpu") as base_st:
                with safe_open(abl_path, framework="pt", device="cpu") as abl_st:
                    base_keys = set(base_st.keys())
                    abl_keys = set(abl_st.keys())

                    common_keys = base_keys & abl_keys

                    for key in common_keys:
                        base_tensor = base_st.get_tensor(key)
                        abl_tensor = abl_st.get_tensor(key)

                        if base_tensor.shape != abl_tensor.shape:
                            print(f"  Shape mismatch: {key}")
                            continue

                        # Calcular diferencias
                        diff = abl_tensor - base_tensor

                        frob_base = torch.norm(base_tensor).item()
                        frob_abl = torch.norm(abl_tensor).item()
                        frob_diff = torch.norm(diff).item()

                        relative_change = frob_diff / frob_base if frob_base > 0 else 0
                        max_diff = torch.max(torch.abs(diff)).item()
                        mean_diff = torch.mean(torch.abs(diff)).item()

                        is_modified = relative_change > self.threshold

                        weight_diff = WeightDiff(
                            name=key,
                            shape=list(base_tensor.shape),
                            frobenius_norm_base=frob_base,
                            frobenius_norm_abliterated=frob_abl,
                            frobenius_norm_diff=frob_diff,
                            relative_change=relative_change,
                            max_abs_diff=max_diff,
                            mean_abs_diff=mean_diff,
                            is_modified=is_modified
                        )
                        weight_diffs.append(weight_diff)

                        # Si está modificado, intentar inferir la dirección
                        if is_modified and 'down_proj' in key or 'o_proj' in key:
                            self._analyze_orthogonalization(key, base_tensor, abl_tensor, diff)

            # Limpiar memoria
            gc.collect()

        self.weight_diffs = weight_diffs
        return weight_diffs

    def _analyze_orthogonalization(
        self,
        key: str,
        base: torch.Tensor,
        abliterated: torch.Tensor,
        diff: torch.Tensor
    ) -> None:
        """Intenta inferir si se usó ortogonalización y extraer la dirección."""
        # Si W' = W - v @ (W @ v).T (ortogonalización estándar)
        # Entonces diff = W' - W = -v @ (W @ v).T
        # diff es una matriz de rango 1 si es ortogonalización pura

        if diff.dim() != 2:
            return

        # SVD para verificar si es rango bajo
        try:
            U, S, Vh = torch.linalg.svd(diff.float(), full_matrices=False)

            # Ratio del primer valor singular vs suma total
            total_sv = S.sum().item()
            if total_sv > 0:
                rank1_ratio = S[0].item() / total_sv

                if rank1_ratio > 0.95:  # Casi rango 1
                    # La dirección es aproximadamente el primer vector singular
                    direction = U[:, 0].numpy().tolist()
                    self.inferred_directions[key] = direction
                    print(f"  Ortogonalización detectada en {key}: rank1_ratio={rank1_ratio:.4f}")
        except Exception as e:
            pass

    def analyze_by_layer(self) -> List[LayerAnalysis]:
        """Agrupa análisis por capa."""
        print("\n" + "=" * 70)
        print("ANÁLISIS POR CAPA")
        print("=" * 70)

        # Agrupar por capa
        layer_weights: Dict[int, List[WeightDiff]] = {}
        other_weights: List[WeightDiff] = []

        for wd in self.weight_diffs:
            # Extraer número de capa del nombre
            import re
            match = re.search(r'layers\.(\d+)\.', wd.name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx not in layer_weights:
                    layer_weights[layer_idx] = []
                layer_weights[layer_idx].append(wd)
            else:
                other_weights.append(wd)

        # Analizar cada capa
        analyses = []

        for layer_idx in sorted(layer_weights.keys()):
            weights = layer_weights[layer_idx]

            total_params = sum(np.prod(w.shape) for w in weights)
            modified = [w for w in weights if w.is_modified]
            modified_params = sum(np.prod(w.shape) for w in modified)

            analysis = LayerAnalysis(
                layer_idx=layer_idx,
                total_params=total_params,
                modified_params=modified_params,
                modification_ratio=modified_params / total_params if total_params > 0 else 0,
                weight_diffs=[asdict(w) for w in weights],
                inferred_direction=None
            )
            analyses.append(analysis)

            if modified:
                mod_names = [w.name.split('.')[-2] + '.' + w.name.split('.')[-1] for w in modified]
                print(f"Capa {layer_idx:2d}: {len(modified):2d} tensores modificados - {', '.join(set(mod_names))}")

        self.layer_analyses = analyses

        # Analizar pesos no-capa (embeddings, lm_head, etc.)
        print("\nOtros pesos modificados:")
        for w in other_weights:
            if w.is_modified:
                print(f"  {w.name}: relative_change={w.relative_change:.6f}")

        return analyses

    def infer_technique(self) -> str:
        """Infiere la técnica usada basándose en el análisis."""
        # Contar cuántas direcciones rank-1 encontramos
        n_rank1 = len(self.inferred_directions)

        # Ver qué tipos de capas fueron modificadas
        modified_types = set()
        for wd in self.weight_diffs:
            if wd.is_modified:
                for t in ['down_proj', 'o_proj', 'up_proj', 'gate_proj', 'q_proj', 'k_proj', 'v_proj', 'lm_head', 'embed']:
                    if t in wd.name:
                        modified_types.add(t)

        if n_rank1 > 10 and 'down_proj' in modified_types:
            return "ORTOGONALIZACIÓN (rank-1 projection removal detectado)"
        elif modified_types:
            return f"MODIFICACIÓN DE PESOS (tipos: {', '.join(modified_types)})"
        else:
            return "NO DETECTADO / MODELOS IDÉNTICOS"

    def compute_direction_similarity(self) -> Dict:
        """Si encontramos múltiples direcciones, calcula su similitud."""
        if len(self.inferred_directions) < 2:
            return {}

        directions = list(self.inferred_directions.values())
        keys = list(self.inferred_directions.keys())

        # Calcular coseno similitud entre direcciones
        similarities = {}
        for i in range(len(directions)):
            for j in range(i+1, len(directions)):
                d1 = torch.tensor(directions[i])
                d2 = torch.tensor(directions[j])

                # Ajustar dimensiones si es necesario
                min_len = min(len(d1), len(d2))
                d1 = d1[:min_len]
                d2 = d2[:min_len]

                cos_sim = torch.nn.functional.cosine_similarity(d1.unsqueeze(0), d2.unsqueeze(0)).item()
                similarities[f"{keys[i]} vs {keys[j]}"] = cos_sim

        return similarities

    def run(self, output_dir: str = "./reverse_analysis") -> ReverseEngineeringResult:
        """Ejecuta el análisis completo."""
        timestamp = datetime.now().isoformat()

        print("=" * 70)
        print("INGENIERÍA INVERSA DE ABLITERACIÓN")
        print(f"Base: {self.base_model}")
        print(f"Abliterated: {self.abliterated_model}")
        print(f"Timestamp: {timestamp}")
        print("=" * 70)

        # 1. Comparar pesos
        self.compare_weights_streaming()

        # 2. Análisis por capa
        self.analyze_by_layer()

        # 3. Inferir técnica
        technique = self.infer_technique()
        print(f"\nTécnica inferida: {technique}")

        # 4. Calcular estadísticas globales
        total_params = sum(np.prod(w.shape) for w in self.weight_diffs)
        modified_weights = [w for w in self.weight_diffs if w.is_modified]
        modified_params = sum(np.prod(w.shape) for w in modified_weights)
        modified_layers = list(set(
            int(w.name.split('.')[2])
            for w in modified_weights
            if 'layers.' in w.name
        ))

        # 5. Resumen
        summary = {
            "total_weight_tensors": len(self.weight_diffs),
            "modified_weight_tensors": len(modified_weights),
            "total_parameters": int(total_params),
            "modified_parameters": int(modified_params),
            "modification_ratio": modified_params / total_params if total_params > 0 else 0,
            "n_modified_layers": len(modified_layers),
            "modified_layer_indices": sorted(modified_layers),
            "n_inferred_directions": len(self.inferred_directions),
            "technique": technique
        }

        print("\n" + "=" * 70)
        print("RESUMEN")
        print("=" * 70)
        print(f"Tensores totales: {summary['total_weight_tensors']}")
        print(f"Tensores modificados: {summary['modified_weight_tensors']}")
        print(f"Parámetros modificados: {summary['modified_parameters']:,} / {summary['total_parameters']:,}")
        print(f"Ratio de modificación: {summary['modification_ratio']:.4%}")
        print(f"Capas modificadas: {len(modified_layers)} capas")
        print(f"Direcciones inferidas: {len(self.inferred_directions)}")
        print(f"Técnica: {technique}")

        # Top 10 tensores más modificados
        print("\nTop 10 tensores más modificados:")
        sorted_diffs = sorted(modified_weights, key=lambda x: x.relative_change, reverse=True)[:10]
        for w in sorted_diffs:
            print(f"  {w.name}: {w.relative_change:.6f}")

        # Construir resultado
        result = ReverseEngineeringResult(
            timestamp=timestamp,
            base_model=self.base_model,
            abliterated_model=self.abliterated_model,
            total_parameters=int(total_params),
            modified_parameters=int(modified_params),
            modification_ratio=modified_params / total_params if total_params > 0 else 0,
            modified_layers=sorted(modified_layers),
            layer_analyses=[asdict(a) for a in self.layer_analyses],
            inferred_technique=technique,
            inferred_directions={k: v for k, v in self.inferred_directions.items()},
            summary=summary
        )

        # Guardar
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, f"reverse_analysis_{timestamp.replace(':', '-')}.json")
        result.save(result_path)
        print(f"\nResultados guardados en: {result_path}")

        return result


def main():
    parser = argparse.ArgumentParser(description="Ingeniería inversa de abliteración")
    parser.add_argument("--base", default="zai-org/GLM-4.7-Flash", help="Modelo base")
    parser.add_argument("--abliterated", default="huihui-ai/Huihui-GLM-4.7-Flash-abliterated", help="Modelo abliterado")
    parser.add_argument("--threshold", type=float, default=1e-6, help="Umbral para considerar modificación")
    parser.add_argument("--output", default="./reverse_analysis", help="Directorio de salida")

    args = parser.parse_args()

    engineer = AbliterationReverseEngineer(
        base_model=args.base,
        abliterated_model=args.abliterated,
        threshold=args.threshold
    )

    result = engineer.run(output_dir=args.output)
    return result


if __name__ == "__main__":
    main()
