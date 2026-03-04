#!/usr/bin/env python3
"""
Quick check: Does huihui-ai/Huihui-Qwen3.5-27B-abliterated have actual weight
modifications compared to Qwen/Qwen3.5-27B, or was it created via runtime hooks?

This script downloads the safetensors index from both models, identifies shared
shard files, and compares a few key weight tensors without loading the full model.
Requires ~0 GPU, ~2-5 GB disk for partial shard downloads.

If ||DeltaW|| > epsilon for any matrix → real weight modification → proceed to
full SVD analysis (weight_diff_analysis_qwen35.py).

If ||DeltaW|| ≈ 0 everywhere → hook-based abliteration → no weight signal to extract.

Usage:
    python weight_diff_quick_check.py
    python weight_diff_quick_check.py --model-a Qwen/Qwen3.5-27B \
           --model-b huihui-ai/Huihui-Qwen3.5-27B-abliterated
    python weight_diff_quick_check.py --layers 32,38,42 --cache-dir /tmp/wdiff_cache
"""

import argparse
import json
import os
import sys
import time

import torch

try:
    from huggingface_hub import hf_hub_download, HfApi
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors not installed. Run: pip install safetensors")
    sys.exit(1)


# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_MODEL_A = "Qwen/Qwen3.5-27B"
DEFAULT_MODEL_B = "huihui-ai/Huihui-Qwen3.5-27B-abliterated"

# Matrices to compare per layer (all projection matrices in transformer block)
WEIGHT_MATRICES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]

# Also check biases (MultiverseComputing/ElSnacko approach modifies biases)
BIAS_MATRICES = [
    "self_attn.q_proj.bias",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.bias",
    "self_attn.o_proj.bias",
    "mlp.gate_proj.bias",
    "mlp.up_proj.bias",
    "mlp.down_proj.bias",
]

# Layers to check (spread across the 64-layer model)
# L32 = 50% depth (peak WRMD), L38 = 60% depth (Sumandora default), L42 = 66%
DEFAULT_LAYERS = [10, 20, 32, 38, 42, 50, 60]


def get_safetensors_index(model_id: str, cache_dir: str) -> dict:
    """Download and parse the safetensors index file."""
    try:
        index_path = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors.index.json",
            cache_dir=cache_dir,
        )
        with open(index_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARNING: Could not download index for {model_id}: {e}")
        return None


def find_shard_for_tensor(index: dict, tensor_name: str) -> str | None:
    """Find which shard file contains a given tensor."""
    weight_map = index.get("weight_map", {})
    return weight_map.get(tensor_name)


def load_tensor_from_shard(
    model_id: str, shard_file: str, tensor_name: str, cache_dir: str
) -> torch.Tensor | None:
    """Download a shard and extract a single tensor."""
    try:
        shard_path = hf_hub_download(
            repo_id=model_id,
            filename=shard_file,
            cache_dir=cache_dir,
        )
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            if tensor_name in f.keys():
                return f.get_tensor(tensor_name)
        return None
    except Exception as e:
        print(f"  WARNING: Could not load {tensor_name} from {shard_file}: {e}")
        return None


def compare_tensors(t_a: torch.Tensor, t_b: torch.Tensor) -> dict:
    """Compute difference metrics between two tensors."""
    # Cast to float32 for accurate comparison
    a = t_a.float()
    b = t_b.float()
    delta = a - b

    frob_norm = delta.norm().item()
    max_abs = delta.abs().max().item()
    mean_abs = delta.abs().mean().item()
    rel_frob = frob_norm / (a.norm().item() + 1e-10)

    # Check if tensors are identical (within float precision)
    is_identical = torch.allclose(a, b, atol=1e-6, rtol=1e-5)

    return {
        "frobenius_norm": frob_norm,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "relative_frobenius": rel_frob,
        "is_identical": is_identical,
        "shape": list(t_a.shape),
        "dtype_a": str(t_a.dtype),
        "dtype_b": str(t_b.dtype),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Quick check for weight differences between aligned and abliterated models"
    )
    parser.add_argument("--model-a", default=DEFAULT_MODEL_A, help="Aligned model (with refusal)")
    parser.add_argument("--model-b", default=DEFAULT_MODEL_B, help="Abliterated model (without refusal)")
    parser.add_argument(
        "--layers",
        default=",".join(map(str, DEFAULT_LAYERS)),
        help="Comma-separated layer indices to check",
    )
    parser.add_argument("--cache-dir", default="/tmp/wdiff_cache", help="Cache directory for downloads")
    parser.add_argument("--check-biases", action="store_true", help="Also check bias tensors")
    parser.add_argument("--output", default="/tmp/weight_diff_quick_check.json", help="Output JSON path")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(args.cache_dir, exist_ok=True)

    print("=" * 70)
    print("Weight Diff Quick Check: Qwen 3.5-27B aligned vs abliterated")
    print("=" * 70)
    print(f"  Model A (aligned):    {args.model_a}")
    print(f"  Model B (abliterated): {args.model_b}")
    print(f"  Layers to check:      {layers}")
    print(f"  Cache dir:            {args.cache_dir}")
    print()

    # Step 1: Download safetensors indices
    print("[1/3] Downloading safetensors indices...")
    t0 = time.time()
    index_a = get_safetensors_index(args.model_a, args.cache_dir)
    index_b = get_safetensors_index(args.model_b, args.cache_dir)

    if index_a is None or index_b is None:
        print("ERROR: Could not download model indices. Check model IDs and network.")
        sys.exit(1)

    print(f"  Model A: {len(index_a['weight_map'])} tensors")
    print(f"  Model B: {len(index_b['weight_map'])} tensors")
    print(f"  Done in {time.time() - t0:.1f}s")
    print()

    # Step 2: Compare weight tensors
    print("[2/3] Comparing weight tensors...")
    matrices_to_check = list(WEIGHT_MATRICES)
    if args.check_biases:
        matrices_to_check.extend(BIAS_MATRICES)

    results = {}
    shards_downloaded = set()  # Track which shards we've already downloaded
    n_modified = 0
    n_identical = 0
    n_missing = 0

    for layer_idx in layers:
        layer_results = {}
        print(f"\n  Layer {layer_idx}:")

        for matrix_name in matrices_to_check:
            full_name = f"model.layers.{layer_idx}.{matrix_name}"

            # Find shards
            shard_a = find_shard_for_tensor(index_a, full_name)
            shard_b = find_shard_for_tensor(index_b, full_name)

            if shard_a is None or shard_b is None:
                if "bias" in matrix_name:
                    # Biases may not exist in all models — skip silently
                    continue
                print(f"    {matrix_name}: MISSING (shard_a={shard_a}, shard_b={shard_b})")
                n_missing += 1
                continue

            # Download and load tensors
            t_a = load_tensor_from_shard(args.model_a, shard_a, full_name, args.cache_dir)
            t_b = load_tensor_from_shard(args.model_b, shard_b, full_name, args.cache_dir)

            if t_a is None or t_b is None:
                print(f"    {matrix_name}: LOAD ERROR")
                n_missing += 1
                continue

            if t_a.shape != t_b.shape:
                print(f"    {matrix_name}: SHAPE MISMATCH {t_a.shape} vs {t_b.shape}")
                n_missing += 1
                continue

            # Compare
            metrics = compare_tensors(t_a, t_b)
            layer_results[matrix_name] = metrics

            status = "IDENTICAL" if metrics["is_identical"] else "MODIFIED"
            if metrics["is_identical"]:
                n_identical += 1
            else:
                n_modified += 1

            frob = metrics["frobenius_norm"]
            rel = metrics["relative_frobenius"]
            maxd = metrics["max_abs_diff"]
            print(f"    {matrix_name}: {status}  ||Delta||={frob:.6f}  rel={rel:.2e}  max={maxd:.2e}")

            # Free memory
            del t_a, t_b

        results[f"layer_{layer_idx}"] = layer_results

    print()

    # Step 3: Summary
    print("[3/3] Summary")
    print("=" * 70)
    total = n_modified + n_identical
    print(f"  Tensors compared: {total}")
    print(f"  Modified:         {n_modified}")
    print(f"  Identical:        {n_identical}")
    print(f"  Missing/Error:    {n_missing}")
    print()

    if n_modified > 0:
        print("  RESULT: WEIGHT MODIFICATION DETECTED")
        print("  The abliterated model has actual weight changes (not just runtime hooks).")
        print("  → Proceed to full SVD analysis: python weight_diff_analysis_qwen35.py")
        print()

        # Find the layers/matrices with largest modifications
        print("  Top modifications by Frobenius norm:")
        all_mods = []
        for layer_key, layer_data in results.items():
            for matrix_name, metrics in layer_data.items():
                if not metrics["is_identical"]:
                    all_mods.append((layer_key, matrix_name, metrics["frobenius_norm"], metrics["relative_frobenius"]))
        all_mods.sort(key=lambda x: x[2], reverse=True)
        for layer_key, matrix_name, frob, rel in all_mods[:15]:
            print(f"    {layer_key}.{matrix_name}: ||Delta||={frob:.4f} (rel={rel:.2e})")

        # Check if modifications are uniform or concentrated
        layer_norms = {}
        for layer_key, layer_data in results.items():
            total_norm = sum(
                m["frobenius_norm"] ** 2
                for m in layer_data.values()
                if not m["is_identical"]
            ) ** 0.5
            layer_norms[layer_key] = total_norm
        print()
        print("  Per-layer total ||DeltaW||:")
        for layer_key in sorted(layer_norms.keys(), key=lambda x: int(x.split("_")[1])):
            norm = layer_norms[layer_key]
            bar = "#" * min(50, int(norm * 10))
            print(f"    {layer_key}: {norm:.4f} {bar}")

    elif n_identical > 0:
        print("  RESULT: NO WEIGHT MODIFICATION DETECTED")
        print("  The abliterated model appears to use runtime hooks (Sumandora-style).")
        print("  Weight-diff reverse engineering is NOT viable for this model.")
        print()
        print("  Alternative approaches:")
        print("  1. Find the abliteration script/config used by huihui-ai")
        print("  2. Use our own WRMD vectors (already extracted)")
        print("  3. Try bias-merge approach (ElSnacko/llm-steering)")
    else:
        print("  RESULT: NO DATA (all tensors missing or errored)")

    # Save results
    output = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "layers_checked": layers,
        "n_modified": n_modified,
        "n_identical": n_identical,
        "n_missing": n_missing,
        "weight_modification_detected": n_modified > 0,
        "per_layer": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
