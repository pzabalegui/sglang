#!/usr/bin/env python3
"""
Compare refusal direction vectors from different extraction methods:
  - WRMD (from activations): ridge-regularized mean difference
  - Weight-diff (from weights): SVD of DeltaW between aligned/abliterated
  - Diff-of-means (from activations): standard approach

For each layer, computes:
  - Cosine similarity between vector pairs
  - Scaling coefficient correlation
  - Agreement on which layers are "active" (high separation)
  - Per-matrix analysis (which weight matrices align with activation-based vectors)

Usage:
    python compare_vectors.py \
        --wrmd-path /tmp/wrmd_directions_per_layer_64layers.pt \
        --wdiff-path /tmp/weight_diff_directions.pt

    python compare_vectors.py \
        --wrmd-path /tmp/wrmd_directions_per_layer_64layers.pt \
        --wdiff-path /tmp/weight_diff_directions.pt \
        --wrmd-scaling /tmp/wrmd_scaling_coeffs_64layers.pt \
        --wdiff-scaling /tmp/weight_diff_scaling.pt \
        --dom-path /tmp/refusal_directions_per_layer_64layers.pt
"""

import argparse
import json
import os
import sys

import torch
import numpy as np


NUM_LAYERS = 64
HIDDEN_SIZE = 5120

# Qwen 3.5 hybrid attention
FULL_ATTN_LAYERS = set(range(3, NUM_LAYERS, 4))


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two vectors. Returns absolute value (direction-agnostic)."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    norm_a = a_flat.norm()
    norm_b = b_flat.norm()
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return abs((a_flat @ b_flat).item() / (norm_a * norm_b).item())


def pearson_correlation(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient between two lists."""
    x = np.array(x)
    y = np.array(y)
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def load_directions(path: str) -> torch.Tensor:
    """
    Load direction vectors. Handles different formats:
    - [num_layers, hidden] for single-vector (diff-of-means, weight-diff)
    - [num_layers, k, hidden] for multi-vector (WRMD k>1)
    Returns [num_layers, hidden] (takes first vector if multi-vector).
    """
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.dim() == 3:
        # Multi-vector: take first direction per layer
        print(f"  Loaded {path}: shape {data.shape} (using v[0] per layer)")
        return data[:, 0, :]
    elif data.dim() == 2:
        print(f"  Loaded {path}: shape {data.shape}")
        return data
    else:
        print(f"  WARNING: Unexpected shape {data.shape} in {path}")
        return data


def load_scaling(path: str) -> torch.Tensor:
    """Load scaling coefficients. Returns [num_layers]."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.dim() == 2:
        return data[:, 0]  # Take first vector's scaling
    return data


def compare_pair(
    name_a: str, dirs_a: torch.Tensor,
    name_b: str, dirs_b: torch.Tensor,
    layers: list[int] | None = None,
) -> dict:
    """Compare two sets of direction vectors across layers."""
    if layers is None:
        layers = list(range(min(dirs_a.shape[0], dirs_b.shape[0])))

    results = {}
    cos_values = []

    for l in layers:
        a = dirs_a[l]
        b = dirs_b[l]

        # Skip zero vectors
        if a.norm() < 1e-8 or b.norm() < 1e-8:
            results[l] = {"cos_sim": 0.0, "zero_vector": True}
            continue

        cos = cosine_similarity(a, b)
        cos_values.append(cos)
        results[l] = {
            "cos_sim": cos,
            "attn_type": "full" if l in FULL_ATTN_LAYERS else "linear",
        }

    # Aggregate statistics
    if cos_values:
        summary = {
            "mean_cos": float(np.mean(cos_values)),
            "median_cos": float(np.median(cos_values)),
            "min_cos": float(np.min(cos_values)),
            "max_cos": float(np.max(cos_values)),
            "std_cos": float(np.std(cos_values)),
            "n_high_agreement": sum(1 for c in cos_values if c > 0.95),
            "n_moderate_agreement": sum(1 for c in cos_values if 0.8 <= c <= 0.95),
            "n_low_agreement": sum(1 for c in cos_values if c < 0.8),
        }
    else:
        summary = {"error": "no valid comparisons"}

    return {"per_layer": results, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description="Compare refusal direction vectors")
    parser.add_argument("--wrmd-path", required=True, help="WRMD directions .pt file")
    parser.add_argument("--wdiff-path", required=True, help="Weight-diff directions .pt file")
    parser.add_argument("--dom-path", help="Diff-of-means directions .pt file (optional)")
    parser.add_argument("--wrmd-scaling", help="WRMD scaling coefficients .pt (optional)")
    parser.add_argument("--wdiff-scaling", help="Weight-diff scaling (||DeltaW||) .pt (optional)")
    parser.add_argument("--output", default="/tmp/vector_comparison.json", help="Output JSON")
    parser.add_argument("--layers", default="all", help="Layers to compare: 'all', '20-50'")
    args = parser.parse_args()

    print("=" * 70)
    print("Vector Comparison: WRMD vs Weight-Diff vs Diff-of-Means")
    print("=" * 70)

    # Parse layers
    if args.layers == "all":
        layers = list(range(NUM_LAYERS))
    elif "-" in args.layers:
        start, end = args.layers.split("-")
        layers = list(range(int(start), int(end) + 1))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    # ── Load vectors ─────────────────────────────────────────────────────────
    print("\n[1/3] Loading direction vectors...")
    dirs_wrmd = load_directions(args.wrmd_path)
    dirs_wdiff = load_directions(args.wdiff_path)
    dirs_dom = load_directions(args.dom_path) if args.dom_path else None

    vectors = {"wrmd": dirs_wrmd, "weight_diff": dirs_wdiff}
    if dirs_dom is not None:
        vectors["diff_of_means"] = dirs_dom

    # ── Compare pairs ────────────────────────────────────────────────────────
    print("\n[2/3] Computing cosine similarity...")

    all_comparisons = {}

    # WRMD vs Weight-diff (primary comparison)
    print("\n  === WRMD vs Weight-Diff ===")
    comp = compare_pair("wrmd", dirs_wrmd, "weight_diff", dirs_wdiff, layers)
    all_comparisons["wrmd_vs_wdiff"] = comp
    s = comp["summary"]
    print(f"  Mean cos:   {s['mean_cos']:.4f}")
    print(f"  Median cos: {s['median_cos']:.4f}")
    print(f"  Range:      [{s['min_cos']:.4f}, {s['max_cos']:.4f}]")
    print(f"  High (>0.95):     {s['n_high_agreement']}")
    print(f"  Moderate (0.8-0.95): {s['n_moderate_agreement']}")
    print(f"  Low (<0.8):       {s['n_low_agreement']}")

    # Per-layer breakdown
    print("\n  Per-layer cosine similarity:")
    for l in layers:
        if l in comp["per_layer"] and not comp["per_layer"][l].get("zero_vector"):
            cos = comp["per_layer"][l]["cos_sim"]
            attn = comp["per_layer"][l]["attn_type"]
            bar = "#" * int(cos * 40)
            marker = " *" if cos > 0.95 else (" ?" if cos < 0.8 else "")
            print(f"    L{l:02d} ({attn:6s}): {cos:.4f} {bar}{marker}")

    # WRMD vs Diff-of-means
    if dirs_dom is not None:
        print("\n  === WRMD vs Diff-of-Means ===")
        comp_dom = compare_pair("wrmd", dirs_wrmd, "diff_of_means", dirs_dom, layers)
        all_comparisons["wrmd_vs_dom"] = comp_dom
        s = comp_dom["summary"]
        print(f"  Mean cos: {s['mean_cos']:.4f}  Range: [{s['min_cos']:.4f}, {s['max_cos']:.4f}]")

        print("\n  === Weight-Diff vs Diff-of-Means ===")
        comp_wd = compare_pair("weight_diff", dirs_wdiff, "diff_of_means", dirs_dom, layers)
        all_comparisons["wdiff_vs_dom"] = comp_wd
        s = comp_wd["summary"]
        print(f"  Mean cos: {s['mean_cos']:.4f}  Range: [{s['min_cos']:.4f}, {s['max_cos']:.4f}]")

    # ── Scaling coefficient analysis ─────────────────────────────────────────
    if args.wrmd_scaling and args.wdiff_scaling:
        print("\n  === Scaling Coefficient Correlation ===")
        scale_wrmd = load_scaling(args.wrmd_scaling)
        scale_wdiff = load_scaling(args.wdiff_scaling)

        # Normalize both to [0, 1] range
        sw = scale_wrmd[layers].float()
        sd = scale_wdiff[layers].float()

        if sw.max() > 0:
            sw = sw / sw.max()
        if sd.max() > 0:
            sd = sd / sd.max()

        corr = pearson_correlation(sw.tolist(), sd.tolist())
        print(f"  Pearson correlation: {corr:.4f}")
        all_comparisons["scaling_correlation"] = {
            "pearson": corr,
            "wrmd_scaling_path": args.wrmd_scaling,
            "wdiff_scaling_path": args.wdiff_scaling,
        }

        # Per-layer comparison
        print("\n  Per-layer scaling (normalized):")
        print(f"  {'Layer':>5s}  {'WRMD':>6s}  {'WDiff':>6s}  {'Ratio':>6s}")
        for i, l in enumerate(layers):
            if sw[i] > 0.01 or sd[i] > 0.01:
                ratio = sd[i].item() / sw[i].item() if sw[i] > 0.01 else float("inf")
                print(f"  L{l:02d}    {sw[i]:.3f}   {sd[i]:.3f}   {ratio:.2f}")

    # ── Full-attn vs Linear-attn breakdown ───────────────────────────────────
    print("\n  === Full-Attention vs Linear-Attention Layers ===")
    comp_primary = all_comparisons["wrmd_vs_wdiff"]["per_layer"]
    full_cos = [comp_primary[l]["cos_sim"] for l in layers
                if l in comp_primary and not comp_primary[l].get("zero_vector")
                and l in FULL_ATTN_LAYERS]
    linear_cos = [comp_primary[l]["cos_sim"] for l in layers
                  if l in comp_primary and not comp_primary[l].get("zero_vector")
                  and l not in FULL_ATTN_LAYERS]

    if full_cos:
        print(f"  Full-attention layers:   mean={np.mean(full_cos):.4f}  n={len(full_cos)}")
    if linear_cos:
        print(f"  Linear-attention layers: mean={np.mean(linear_cos):.4f}  n={len(linear_cos)}")

    # ── Step 3: Save results ─────────────────────────────────────────────────
    print(f"\n[3/3] Saving results to {args.output}...")

    # Convert per_layer keys to strings for JSON
    for comp_name, comp_data in all_comparisons.items():
        if "per_layer" in comp_data:
            comp_data["per_layer"] = {
                str(k): v for k, v in comp_data["per_layer"].items()
            }

    output = {
        "wrmd_path": args.wrmd_path,
        "wdiff_path": args.wdiff_path,
        "dom_path": args.dom_path,
        "layers": layers,
        "comparisons": all_comparisons,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved to: {args.output}")

    # ── Interpretation guide ─────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    s = all_comparisons["wrmd_vs_wdiff"]["summary"]
    mean_cos = s["mean_cos"]

    if mean_cos > 0.95:
        print("  WRMD and weight-diff vectors are nearly IDENTICAL (cos > 0.95).")
        print("  Both methods capture the same refusal direction.")
        print("  → Use either for DAS; WRMD is preferred (has scaling coefficients).")
    elif mean_cos > 0.8:
        print("  WRMD and weight-diff vectors show MODERATE agreement (0.8 < cos < 0.95).")
        print("  They capture similar but not identical directions.")
        print("  → Run DAS benchmark with BOTH sets to determine which performs better.")
    elif mean_cos > 0.5:
        print("  WRMD and weight-diff vectors show PARTIAL agreement (0.5 < cos < 0.8).")
        print("  They capture partially overlapping refusal subspaces.")
        print("  → Consider using BOTH (multi-vector DAS v5) for broader coverage.")
    else:
        print("  WRMD and weight-diff vectors show LOW agreement (cos < 0.5).")
        print("  They may capture different aspects of the refusal mechanism.")
        print("  → Investigate which set produces better DAS results.")
        print("  → The weight-diff may reflect a different extraction method.")
    print("=" * 70)


if __name__ == "__main__":
    main()
