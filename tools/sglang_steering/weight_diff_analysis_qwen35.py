#!/usr/bin/env python3
"""
Full SVD analysis of weight differences between Qwen3.5-27B (aligned) and
huihui-ai/Huihui-Qwen3.5-27B-abliterated.

For each layer (0..63) and each weight matrix (q/k/v/o_proj, gate/up/down_proj):
  1. DeltaW = W_aligned - W_abliterated
  2. SVD(DeltaW) → top-k singular values and vectors
  3. Rank-1 ratio = S[0] / sum(S)
  4. r_recovered = Vh[0] (refusal direction from weight space)
  5. magnitude = S[0] (intervention strength)

Outputs:
  /tmp/weight_diff_spectrum.json        -- singular value spectrum per layer/matrix
  /tmp/weight_diff_directions.pt        -- recovered r vectors [64, hidden_size]
  /tmp/weight_diff_scaling.pt           -- ||DeltaW|| per layer [64]
  /tmp/weight_diff_full_directions.pt   -- per-layer per-matrix directions [64, 7, hidden_size]

Prerequisites:
  - Run weight_diff_quick_check.py first to confirm weight modifications exist
  - ~108GB disk for both model checkpoints (or use streaming via safetensors)
  - No GPU required (CPU SVD is fine for [out_dim, in_dim] matrices)

Usage:
    python weight_diff_analysis_qwen35.py
    python weight_diff_analysis_qwen35.py --model-a Qwen/Qwen3.5-27B \
           --model-b huihui-ai/Huihui-Qwen3.5-27B-abliterated
    python weight_diff_analysis_qwen35.py --top-k 10 --layers 20-50
"""

import argparse
import gc
import json
import os
import sys
import time

import torch
import numpy as np

try:
    from huggingface_hub import hf_hub_download
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
NUM_LAYERS = 64
HIDDEN_SIZE = 5120

# All projection weight matrices in a transformer layer
WEIGHT_MATRICES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]

# Qwen 3.5 hybrid attention: layers 3,7,11,...,63 are full attention
FULL_ATTN_LAYERS = set(range(3, NUM_LAYERS, 4))


def parse_layer_range(s: str) -> list[int]:
    """Parse layer specification: '32', '20-50', '10,20,32,38,42', 'all'."""
    if s == "all":
        return list(range(NUM_LAYERS))
    if "-" in s and "," not in s:
        start, end = s.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in s.split(",")]


def get_safetensors_index(model_id: str, cache_dir: str) -> dict:
    """Download and parse the safetensors index file."""
    index_path = hf_hub_download(
        repo_id=model_id,
        filename="model.safetensors.index.json",
        cache_dir=cache_dir,
    )
    with open(index_path) as f:
        return json.load(f)


def load_tensor(model_id: str, index: dict, tensor_name: str, cache_dir: str) -> torch.Tensor | None:
    """Load a single tensor from the appropriate shard."""
    shard_file = index.get("weight_map", {}).get(tensor_name)
    if shard_file is None:
        return None
    shard_path = hf_hub_download(
        repo_id=model_id,
        filename=shard_file,
        cache_dir=cache_dir,
    )
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        if tensor_name in f.keys():
            return f.get_tensor(tensor_name)
    return None


def analyze_delta_svd(delta_w: torch.Tensor, top_k: int = 10) -> dict:
    """
    Perform truncated SVD on DeltaW and return analysis metrics.

    For abliteration: DeltaW = (W · r) ⊗ r^T should be rank-1.
    If rank-1: r = Vh[0] (first right singular vector) is the refusal direction.

    Args:
        delta_w: Weight difference tensor [out_dim, in_dim]
        top_k: Number of singular values/vectors to compute

    Returns:
        Dict with SVD analysis results
    """
    # Cast to float32 for SVD stability
    delta = delta_w.float()
    frob_norm = delta.norm().item()

    if frob_norm < 1e-8:
        return {
            "frobenius_norm": frob_norm,
            "is_zero": True,
            "singular_values": [],
            "rank_1_ratio": 0.0,
            "rank_1_energy": 0.0,
        }

    # Truncated SVD (only top-k components)
    # For [out_dim, in_dim], this is efficient when top_k << min(out_dim, in_dim)
    try:
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        S = S[:top_k]
        Vh = Vh[:top_k]
        U = U[:, :top_k]
    except Exception as e:
        print(f"    SVD failed: {e}")
        return {
            "frobenius_norm": frob_norm,
            "is_zero": False,
            "svd_error": str(e),
        }

    s_values = S.tolist()
    s_total = S.sum().item()
    s_energy_total = (S ** 2).sum().item()

    # Rank-1 metrics
    rank_1_ratio = s_values[0] / s_total if s_total > 0 else 0.0
    rank_1_energy = (s_values[0] ** 2) / s_energy_total if s_energy_total > 0 else 0.0

    # Effective rank (how many dimensions needed to capture 99% energy)
    cumulative_energy = torch.cumsum(S ** 2, dim=0) / s_energy_total
    effective_rank = int((cumulative_energy < 0.99).sum().item()) + 1

    # The refusal direction candidate: first right singular vector
    r_candidate = Vh[0]  # [in_dim] = [hidden_size] for most matrices

    return {
        "frobenius_norm": frob_norm,
        "is_zero": False,
        "singular_values": s_values,
        "rank_1_ratio": rank_1_ratio,
        "rank_1_energy": rank_1_energy,
        "effective_rank_99": effective_rank,
        "s0_over_s1": s_values[0] / s_values[1] if len(s_values) > 1 and s_values[1] > 1e-10 else float("inf"),
        "r_candidate": r_candidate,  # Will be stored separately
        "r_candidate_norm": r_candidate.norm().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Full SVD analysis of weight differences")
    parser.add_argument("--model-a", default=DEFAULT_MODEL_A, help="Aligned model")
    parser.add_argument("--model-b", default=DEFAULT_MODEL_B, help="Abliterated model")
    parser.add_argument("--layers", default="all", help="Layers to analyze: 'all', '20-50', '10,20,32'")
    parser.add_argument("--top-k", type=int, default=10, help="Number of SVD components")
    parser.add_argument("--cache-dir", default="/tmp/wdiff_cache", help="Cache for model downloads")
    parser.add_argument("--output-dir", default="/tmp", help="Output directory for results")
    args = parser.parse_args()

    layers = parse_layer_range(args.layers)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Weight Diff SVD Analysis: Qwen 3.5-27B aligned vs abliterated")
    print("=" * 70)
    print(f"  Model A: {args.model_a}")
    print(f"  Model B: {args.model_b}")
    print(f"  Layers:  {len(layers)} ({min(layers)}-{max(layers)})")
    print(f"  Top-k:   {args.top_k}")
    print()

    # ── Step 1: Download indices ─────────────────────────────────────────────
    print("[1/4] Downloading model indices...")
    t0 = time.time()
    index_a = get_safetensors_index(args.model_a, args.cache_dir)
    index_b = get_safetensors_index(args.model_b, args.cache_dir)
    print(f"  Done in {time.time() - t0:.1f}s")
    print()

    # ── Step 2: Analyze each layer ───────────────────────────────────────────
    print("[2/4] Analyzing weight differences per layer...")

    spectrum_results = {}  # layer -> matrix -> SVD metrics
    # Aggregate direction: for each layer, pick the matrix with highest rank-1 energy
    # and store its r_candidate as the layer's refusal direction
    best_directions = torch.zeros(NUM_LAYERS, HIDDEN_SIZE)
    layer_magnitudes = torch.zeros(NUM_LAYERS)
    full_directions = {}  # layer -> matrix -> r_candidate

    for layer_idx in layers:
        print(f"\n  Layer {layer_idx} ({'full-attn' if layer_idx in FULL_ATTN_LAYERS else 'linear-attn'}):")
        layer_results = {}
        best_energy = -1.0
        best_r = None
        layer_total_frob_sq = 0.0

        for matrix_name in WEIGHT_MATRICES:
            full_name = f"model.layers.{layer_idx}.{matrix_name}"

            # Load tensors
            t_a = load_tensor(args.model_a, index_a, full_name, args.cache_dir)
            t_b = load_tensor(args.model_b, index_b, full_name, args.cache_dir)

            if t_a is None or t_b is None:
                print(f"    {matrix_name}: SKIP (not found)")
                continue

            if t_a.shape != t_b.shape:
                print(f"    {matrix_name}: SHAPE MISMATCH")
                continue

            # Compute delta and analyze
            delta_w = t_a.float() - t_b.float()
            analysis = analyze_delta_svd(delta_w, top_k=args.top_k)

            # Store r_candidate separately (not JSON serializable)
            r_candidate = analysis.pop("r_candidate", None)

            # Print summary
            if analysis.get("is_zero", True):
                print(f"    {matrix_name}: ZERO (no modification)")
            else:
                r1r = analysis["rank_1_ratio"]
                r1e = analysis["rank_1_energy"]
                s0s1 = analysis.get("s0_over_s1", 0)
                eff_r = analysis["effective_rank_99"]
                frob = analysis["frobenius_norm"]
                print(
                    f"    {matrix_name}: ||Delta||={frob:.4f}  "
                    f"rank1_ratio={r1r:.3f}  rank1_energy={r1e:.4f}  "
                    f"S0/S1={s0s1:.1f}  eff_rank={eff_r}"
                )

                layer_total_frob_sq += frob ** 2

                # Track best direction per layer
                # For o_proj and down_proj, the "input" dimension is hidden_size
                # For q/k/v_proj and gate/up_proj, the "output" dimension may differ
                # We want r in hidden_size space, so use Vh[0] for matrices where
                # input dim = hidden_size, and U[:,0] otherwise
                if r_candidate is not None:
                    # o_proj: [hidden, hidden] → r = Vh[0] is in hidden_size space
                    # q_proj: [num_heads*head_dim, hidden] → r = Vh[0] is in hidden_size space
                    # gate/up_proj: [intermediate, hidden] → r = Vh[0] is in hidden_size space
                    # down_proj: [hidden, intermediate] → r = Vh[0] is in intermediate space (wrong!)
                    # For down_proj, we need U[:,0] instead (left singular vector)
                    if "down_proj" in matrix_name:
                        # For down_proj [hidden, intermediate], the output is hidden_size
                        # We want the direction in hidden_size space
                        # Re-run SVD to get U (we already have it from analyze_delta_svd
                        # but it was discarded). Recompute just for this case.
                        U_dp, S_dp, Vh_dp = torch.linalg.svd(delta_w, full_matrices=False)
                        r_in_hidden = U_dp[:, 0]  # [hidden_size]
                        r_in_hidden = r_in_hidden / r_in_hidden.norm()
                        if r_in_hidden.shape[0] == HIDDEN_SIZE and r1e > best_energy:
                            best_energy = r1e
                            best_r = r_in_hidden
                        if layer_idx not in full_directions:
                            full_directions[layer_idx] = {}
                        full_directions[layer_idx][matrix_name] = r_in_hidden
                        del U_dp, S_dp, Vh_dp
                    else:
                        # r_candidate = Vh[0], shape [in_dim] = [hidden_size]
                        if r_candidate.shape[0] == HIDDEN_SIZE and r1e > best_energy:
                            best_energy = r1e
                            best_r = r_candidate / r_candidate.norm()
                        if layer_idx not in full_directions:
                            full_directions[layer_idx] = {}
                        full_directions[layer_idx][matrix_name] = r_candidate / r_candidate.norm()

            layer_results[matrix_name] = analysis
            del t_a, t_b, delta_w
            gc.collect()

        spectrum_results[f"layer_{layer_idx}"] = layer_results
        layer_magnitudes[layer_idx] = layer_total_frob_sq ** 0.5

        if best_r is not None:
            best_directions[layer_idx] = best_r
            print(f"    → Best direction from matrix with rank-1 energy = {best_energy:.4f}")

    print()

    # ── Step 3: Aggregate analysis ───────────────────────────────────────────
    print("[3/4] Aggregate analysis...")

    # Find layers with strongest modifications
    print("\n  Per-layer total ||DeltaW|| (sorted by magnitude):")
    layer_mag_list = [(i, layer_magnitudes[i].item()) for i in layers if layer_magnitudes[i] > 0]
    layer_mag_list.sort(key=lambda x: x[1], reverse=True)
    for layer_idx, mag in layer_mag_list[:20]:
        attn_type = "full" if layer_idx in FULL_ATTN_LAYERS else "linear"
        bar = "#" * min(50, int(mag * 5))
        print(f"    L{layer_idx:02d} ({attn_type:6s}): {mag:.4f} {bar}")

    # Check if modifications are concentrated in specific matrix types
    matrix_totals = {}
    for layer_key, layer_data in spectrum_results.items():
        for matrix_name, metrics in layer_data.items():
            if not metrics.get("is_zero", True):
                if matrix_name not in matrix_totals:
                    matrix_totals[matrix_name] = 0.0
                matrix_totals[matrix_name] += metrics["frobenius_norm"] ** 2

    print("\n  Total ||DeltaW|| by matrix type:")
    for matrix_name, total_sq in sorted(matrix_totals.items(), key=lambda x: x[1], reverse=True):
        total = total_sq ** 0.5
        print(f"    {matrix_name:30s}: {total:.4f}")

    # Check rank-1 quality across layers
    print("\n  Rank-1 energy across layers (higher = more rank-1 = cleaner abliteration):")
    for layer_idx in sorted(layers):
        layer_key = f"layer_{layer_idx}"
        if layer_key in spectrum_results:
            energies = []
            for matrix_name, metrics in spectrum_results[layer_key].items():
                if not metrics.get("is_zero", True) and "rank_1_energy" in metrics:
                    energies.append((matrix_name, metrics["rank_1_energy"]))
            if energies:
                avg_energy = sum(e for _, e in energies) / len(energies)
                best_matrix, best_e = max(energies, key=lambda x: x[1])
                print(f"    L{layer_idx:02d}: avg={avg_energy:.3f}  best={best_e:.3f} ({best_matrix})")

    # ── Step 4: Save outputs ─────────────────────────────────────────────────
    print("\n[4/4] Saving outputs...")

    # 1. Spectrum JSON (without tensor data)
    spectrum_path = os.path.join(args.output_dir, "weight_diff_spectrum.json")
    with open(spectrum_path, "w") as f:
        json.dump(
            {
                "model_a": args.model_a,
                "model_b": args.model_b,
                "layers_analyzed": layers,
                "top_k": args.top_k,
                "per_layer": spectrum_results,
            },
            f,
            indent=2,
        )
    print(f"  Saved spectrum: {spectrum_path}")

    # 2. Best direction per layer [64, hidden_size]
    directions_path = os.path.join(args.output_dir, "weight_diff_directions.pt")
    torch.save(best_directions, directions_path)
    print(f"  Saved directions: {directions_path} (shape {best_directions.shape})")

    # 3. Layer magnitudes [64]
    scaling_path = os.path.join(args.output_dir, "weight_diff_scaling.pt")
    torch.save(layer_magnitudes, scaling_path)
    print(f"  Saved scaling: {scaling_path} (shape {layer_magnitudes.shape})")

    # 4. Find the best overall layer (highest ||DeltaW|| with good rank-1)
    if layer_mag_list:
        best_layer_idx = layer_mag_list[0][0]
        best_layer_dir = best_directions[best_layer_idx]
        best_path = os.path.join(args.output_dir, f"weight_diff_direction_qwen35_L{best_layer_idx}.pt")
        torch.save(best_layer_dir, best_path)
        print(f"  Saved best-layer direction: {best_path}")

        # Also save as LBEST for launch_server compatibility
        lbest_path = os.path.join(args.output_dir, "weight_diff_direction_qwen35_LBEST.pt")
        torch.save(best_layer_dir, lbest_path)
        print(f"  Saved LBEST symlink: {lbest_path}")

    print()
    print("=" * 70)
    print("Analysis complete!")
    print()
    print("Next steps:")
    print("  1. Compare with WRMD vectors:")
    print("     python compare_vectors.py --wrmd-path /tmp/wrmd_directions_per_layer_64layers.pt \\")
    print(f"       --wdiff-path {directions_path}")
    print()
    print("  2. Deploy DAS with weight-diff vectors:")
    print(f"     Update launch_server_qwen35.sh to use {directions_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
