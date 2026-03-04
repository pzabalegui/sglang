#!/usr/bin/env python3
"""
Análisis rápido de diferencias de pesos entre modelos.
Sin SVD - solo comparación directa de tensores.
"""

import json
import os
from datetime import datetime
from collections import defaultdict

import torch
from tqdm import tqdm
from safetensors import safe_open
from huggingface_hub import hf_hub_download, list_repo_files


def analyze_weight_diff(
    base_model: str = "zai-org/GLM-4.7-Flash",
    abliterated_model: str = "huihui-ai/Huihui-GLM-4.7-Flash-abliterated",
    threshold: float = 1e-6,
    output_dir: str = "./reverse_analysis"
):
    """Análisis rápido de diferencias de pesos."""

    timestamp = datetime.now().isoformat()
    print("=" * 70)
    print("ANÁLISIS RÁPIDO DE DIFERENCIAS DE PESOS")
    print(f"Base: {base_model}")
    print(f"Abliterated: {abliterated_model}")
    print("=" * 70)

    # Obtener archivos
    print("\nObteniendo lista de archivos...")
    base_files = sorted([f for f in list_repo_files(base_model) if f.endswith('.safetensors')])
    abl_files = sorted([f for f in list_repo_files(abliterated_model) if f.endswith('.safetensors')])

    print(f"  Base: {len(base_files)} shards")
    print(f"  Abliterated: {len(abl_files)} shards")

    # Resultados
    modified_weights = []
    all_weights = []
    layer_stats = defaultdict(lambda: {"total": 0, "modified": 0, "weights": []})

    # Comparar shard por shard
    for base_file in tqdm(base_files, desc="Comparando shards"):
        if base_file not in abl_files:
            continue

        base_path = hf_hub_download(base_model, base_file)
        abl_path = hf_hub_download(abliterated_model, base_file)

        with safe_open(base_path, framework="pt", device="cpu") as base_st:
            with safe_open(abl_path, framework="pt", device="cpu") as abl_st:
                for key in base_st.keys():
                    if key not in abl_st.keys():
                        continue

                    base_t = base_st.get_tensor(key)
                    abl_t = abl_st.get_tensor(key)

                    if base_t.shape != abl_t.shape:
                        continue

                    # Calcular métricas
                    diff = (abl_t - base_t).float()
                    frob_base = torch.norm(base_t.float()).item()
                    frob_diff = torch.norm(diff).item()

                    rel_change = frob_diff / frob_base if frob_base > 0 else 0
                    max_diff = torch.max(torch.abs(diff)).item()

                    is_modified = rel_change > threshold

                    # Extraer capa
                    layer_idx = -1
                    if "layers." in key:
                        try:
                            layer_idx = int(key.split("layers.")[1].split(".")[0])
                        except:
                            pass

                    weight_info = {
                        "name": key,
                        "shape": list(base_t.shape),
                        "params": int(base_t.numel()),
                        "frob_base": frob_base,
                        "frob_diff": frob_diff,
                        "rel_change": rel_change,
                        "max_diff": max_diff,
                        "is_modified": is_modified,
                        "layer": layer_idx
                    }

                    all_weights.append(weight_info)

                    if is_modified:
                        modified_weights.append(weight_info)

                    if layer_idx >= 0:
                        layer_stats[layer_idx]["total"] += base_t.numel()
                        if is_modified:
                            layer_stats[layer_idx]["modified"] += base_t.numel()
                            layer_stats[layer_idx]["weights"].append(key.split(".")[-2] + "." + key.split(".")[-1])

    # Análisis
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)

    total_params = sum(w["params"] for w in all_weights)
    modified_params = sum(w["params"] for w in modified_weights)

    print(f"\nTotal tensores: {len(all_weights)}")
    print(f"Tensores modificados: {len(modified_weights)}")
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros modificados: {modified_params:,}")
    print(f"Ratio de modificación: {modified_params/total_params:.4%}")

    # Capas modificadas
    print("\n" + "-" * 70)
    print("CAPAS MODIFICADAS")
    print("-" * 70)

    modified_layers = []
    for layer_idx in sorted(layer_stats.keys()):
        stats = layer_stats[layer_idx]
        if stats["modified"] > 0:
            ratio = stats["modified"] / stats["total"] if stats["total"] > 0 else 0
            unique_weights = list(set(stats["weights"]))
            modified_layers.append(layer_idx)
            print(f"Capa {layer_idx:2d}: {ratio:.2%} modificado - {', '.join(unique_weights[:5])}")

    # Top modificados
    print("\n" + "-" * 70)
    print("TOP 20 TENSORES MÁS MODIFICADOS")
    print("-" * 70)

    sorted_mods = sorted(modified_weights, key=lambda x: x["rel_change"], reverse=True)[:20]
    for w in sorted_mods:
        print(f"  {w['rel_change']:.6f} | {w['name']}")

    # Analizar patrón
    print("\n" + "-" * 70)
    print("ANÁLISIS DE PATRÓN")
    print("-" * 70)

    # Qué tipos de pesos fueron modificados
    weight_types = defaultdict(int)
    for w in modified_weights:
        for t in ["down_proj", "o_proj", "up_proj", "gate_proj", "q_proj", "k_proj", "v_proj",
                  "q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj", "lm_head", "embed",
                  "layernorm", "gate.weight", "experts"]:
            if t in w["name"]:
                weight_types[t] += 1
                break

    print("Tipos de pesos modificados:")
    for t, count in sorted(weight_types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    # Inferir técnica
    if "down_proj" in weight_types or "o_proj" in weight_types:
        if len(modified_layers) == len(layer_stats):  # Todas las capas
            technique = "ORTOGONALIZACIÓN GLOBAL (todas las capas, down_proj/o_proj)"
        else:
            technique = f"ORTOGONALIZACIÓN PARCIAL ({len(modified_layers)} capas)"
    else:
        technique = "TÉCNICA DESCONOCIDA"

    print(f"\nTécnica inferida: {technique}")

    # Guardar resultados
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "timestamp": timestamp,
        "base_model": base_model,
        "abliterated_model": abliterated_model,
        "total_tensors": len(all_weights),
        "modified_tensors": len(modified_weights),
        "total_params": total_params,
        "modified_params": modified_params,
        "modification_ratio": modified_params / total_params,
        "modified_layers": modified_layers,
        "n_layers_total": len(layer_stats),
        "weight_types_modified": dict(weight_types),
        "inferred_technique": technique,
        "top_modified": sorted_mods[:50]
    }

    result_path = os.path.join(output_dir, f"quick_analysis_{timestamp.replace(':', '-')}.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResultados guardados en: {result_path}")

    return result


if __name__ == "__main__":
    analyze_weight_diff()
