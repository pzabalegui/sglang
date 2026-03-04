#!/usr/bin/env python3
"""
Diagnostic 2D scatter plots for refusal direction analysis.

Produces two views per layer:
  A) Projection onto (WRMD_direction, best_orthogonal) — tests if our vector
     captures the full separation or misses a component
  B) Projection onto (PC1, PC2) of combined activations — shows raw geometry,
     identifies subclusters/outliers

Also computes:
  - Variance explained by WRMD direction vs PCA components
  - Subcluster detection (DBSCAN on harmful activations)
  - Outlier identification (prompts closest to decision boundary)

When to run: AFTER extraction (needs activations + WRMD vectors), BEFORE server
deployment. This is a diagnostic step to validate vector quality and decide
whether to proceed with single-vector (v4) or multi-vector (v5) steering.

What to look for in the plots:
  - View A: If separation is tilted → WRMD direction is slightly off-axis
  - View A: If there's separation along the orthogonal axis → multi-dimensional
    refusal, consider k>1 (DAS v5)
  - View B: If harmful cluster is bimodal → subcategories of refusal exist
  - View B: Points between clusters → prompts DAS will struggle with
  - Both: Outliers far from their cluster → noisy prompts to remove

Inputs:
  - Saved activations (from extract_wrmd_qwen35.py --save-activations)
  - WRMD directions (from extract_wrmd_qwen35.py)
  - Or: run standalone with model + prompts (slower)

Usage:
    # From saved activations (fast, no GPU):
    python diagnostic_scatter.py \
        --activations /tmp/wrmd_activations.pt \
        --directions /tmp/wrmd_directions_per_layer_64layers.pt \
        --results /tmp/wrmd_results_qwen35.json \
        --layers best,20,32,42,50

    # Standalone (needs GPU + model):
    python diagnostic_scatter.py \
        --model-path /tmp/Qwen3.5-27B-FP8 \
        --layers best
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available. Install with: pip install matplotlib")


# ── Qwen 3.5 architecture ───────────────────────────────────────────────────
FULL_ATTN_LAYERS = set(range(3, 64, 4))


def setup_style():
    """Publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.grid": True,
        "grid.alpha": 0.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def compute_orthogonal_pc(activations, direction):
    """Find the best direction orthogonal to `direction` that explains maximum
    variance in the residual (activations projected away from direction).

    Returns:
        ortho_dir: [hidden] unit vector, orthogonal to direction
        var_explained_direction: fraction of total variance along direction
        var_explained_ortho: fraction of total variance along ortho_dir
    """
    direction = direction / direction.norm()
    # Project out the refusal direction
    projs = (activations @ direction).unsqueeze(1) * direction.unsqueeze(0)
    residual = activations - projs  # [N, hidden]

    # PCA on residual (rank-1 SVD)
    centered = residual - residual.mean(0)
    _, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    ortho_dir = Vt[0]  # [hidden], best orthogonal direction

    # Verify orthogonality
    dot = abs((ortho_dir @ direction).item())
    if dot > 0.01:
        # Gram-Schmidt to enforce
        ortho_dir = ortho_dir - (ortho_dir @ direction) * direction
        ortho_dir = ortho_dir / ortho_dir.norm()

    # Variance explained
    total_var = activations.var(0).sum().item()
    var_along_dir = (activations @ direction).var().item()
    var_along_ortho = (activations @ ortho_dir).var().item()

    return ortho_dir, var_along_dir / total_var, var_along_ortho / total_var


def compute_pca(activations, n_components=2):
    """PCA on combined activations. Returns projected data and variance explained."""
    centered = activations - activations.mean(0)
    _, S, Vt = torch.linalg.svd(centered, full_matrices=False)

    components = Vt[:n_components]  # [n_components, hidden]
    projected = centered @ components.T  # [N, n_components]

    total_var = (S ** 2).sum().item()
    var_explained = [(S[i] ** 2).item() / total_var for i in range(n_components)]

    return projected, components, var_explained


def detect_subclusters(projected_2d, min_samples=5, eps_quantile=0.1):
    """Simple density-based subcluster detection using distance thresholding.
    Returns cluster labels (no sklearn dependency)."""
    from scipy.spatial.distance import pdist, squareform
    try:
        dists = squareform(pdist(projected_2d))
        eps = np.quantile(dists[dists > 0], eps_quantile)

        # Simple connected-components clustering
        n = len(projected_2d)
        labels = -np.ones(n, dtype=int)
        cluster_id = 0
        for i in range(n):
            if labels[i] >= 0:
                continue
            neighbors = np.where(dists[i] < eps)[0]
            if len(neighbors) >= min_samples:
                labels[neighbors[labels[neighbors] < 0]] = cluster_id
                cluster_id += 1

        return labels, cluster_id
    except ImportError:
        return np.zeros(len(projected_2d), dtype=int), 1


def find_boundary_prompts(harm_projs, safe_projs, direction_projs_harm, direction_projs_safe,
                          harmful_prompts=None, harmless_prompts=None, n=5):
    """Find prompts closest to the decision boundary (most ambiguous)."""
    threshold = (direction_projs_harm.mean() + direction_projs_safe.mean()) / 2

    # Harmful prompts closest to boundary (most likely to fail DAS)
    harm_dist = abs(direction_projs_harm - threshold)
    harm_sorted = np.argsort(harm_dist)[:n]

    # Harmless prompts closest to boundary (most likely false-positive)
    safe_dist = abs(direction_projs_safe - threshold)
    safe_sorted = np.argsort(safe_dist)[:n]

    boundary = {"harmful_near_boundary": [], "harmless_near_boundary": []}
    for idx in harm_sorted:
        entry = {"index": int(idx), "projection": float(direction_projs_harm[idx]),
                 "distance_to_boundary": float(harm_dist[idx])}
        if harmful_prompts:
            entry["prompt"] = harmful_prompts[idx][:100]
        boundary["harmful_near_boundary"].append(entry)
    for idx in safe_sorted:
        entry = {"index": int(idx), "projection": float(direction_projs_safe[idx]),
                 "distance_to_boundary": float(safe_dist[idx])}
        if harmless_prompts:
            entry["prompt"] = harmless_prompts[idx][:100]
        boundary["harmless_near_boundary"].append(entry)

    return boundary


def plot_layer_diagnostic(harm_states, safe_states, wrmd_dir, layer_idx,
                          output_dir, harmful_prompts=None, harmless_prompts=None):
    """Generate the 2-panel diagnostic scatter plot for one layer.

    Panel A: Projection onto (WRMD_direction, best_orthogonal)
    Panel B: Projection onto (PC1, PC2) of combined activations
    """
    n_harm = harm_states.shape[0]
    n_safe = safe_states.shape[0]
    all_states = torch.cat([harm_states, safe_states], dim=0)  # [N, hidden]
    labels = np.array(["harmful"] * n_harm + ["harmless"] * n_safe)

    attn_type = "full-attn" if layer_idx in FULL_ATTN_LAYERS else "linear-attn"

    # ── View A: WRMD direction + best orthogonal ─────────────────────────────
    wrmd_norm = wrmd_dir / wrmd_dir.norm()
    ortho_dir, var_wrmd, var_ortho = compute_orthogonal_pc(all_states, wrmd_norm)

    proj_wrmd_harm = (harm_states @ wrmd_norm).numpy()
    proj_wrmd_safe = (safe_states @ wrmd_norm).numpy()
    proj_ortho_harm = (harm_states @ ortho_dir).numpy()
    proj_ortho_safe = (safe_states @ ortho_dir).numpy()

    # ── View B: PCA on combined ──────────────────────────────────────────────
    projected, pca_components, var_explained_pca = compute_pca(all_states, n_components=5)
    pca_harm = projected[:n_harm].numpy()
    pca_safe = projected[n_harm:].numpy()

    # Check alignment between WRMD direction and PC1
    cos_wrmd_pc1 = abs((wrmd_norm @ pca_components[0]).item())

    # ── Subcluster detection on harmful activations ──────────────────────────
    harm_2d = np.column_stack([proj_wrmd_harm, proj_ortho_harm])
    subcluster_labels, n_subclusters = detect_subclusters(harm_2d)

    # ── Boundary prompts ─────────────────────────────────────────────────────
    boundary_info = find_boundary_prompts(
        proj_wrmd_harm, proj_wrmd_safe,
        proj_wrmd_harm, proj_wrmd_safe,
        harmful_prompts, harmless_prompts
    )

    # ── PLOT ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: WRMD direction vs orthogonal
    ax = axes[0]
    ax.scatter(proj_wrmd_safe, proj_ortho_safe, c="#2196F3", alpha=0.5, s=20,
               label=f"Harmless (n={n_safe})", edgecolors="none")
    ax.scatter(proj_wrmd_harm, proj_ortho_harm, c="#F44336", alpha=0.5, s=20,
               label=f"Harmful (n={n_harm})", edgecolors="none")

    # Decision boundary
    threshold = (proj_wrmd_harm.mean() + proj_wrmd_safe.mean()) / 2
    y_range = ax.get_ylim()
    ax.axvline(x=threshold, color="gray", linestyle="--", alpha=0.5, label="Decision boundary")

    # Mark boundary prompts
    for bp in boundary_info["harmful_near_boundary"][:3]:
        idx = bp["index"]
        ax.scatter(proj_wrmd_harm[idx], proj_ortho_harm[idx],
                   c="red", s=80, marker="x", linewidths=2, zorder=5)
    for bp in boundary_info["harmless_near_boundary"][:3]:
        idx = bp["index"]
        ax.scatter(proj_wrmd_safe[idx], proj_ortho_safe[idx],
                   c="blue", s=80, marker="x", linewidths=2, zorder=5)

    ax.set_xlabel(f"Projection onto WRMD direction (var={var_wrmd*100:.1f}%)")
    ax.set_ylabel(f"Projection onto best orthogonal (var={var_ortho*100:.1f}%)")
    ax.set_title(f"Layer {layer_idx} ({attn_type}) — WRMD axis view")
    ax.legend(loc="upper left", fontsize=8)

    # Annotate separation quality
    cohens_d = (proj_wrmd_harm.mean() - proj_wrmd_safe.mean()) / \
               (np.sqrt((proj_wrmd_harm.std()**2 + proj_wrmd_safe.std()**2) / 2) + 1e-10)
    info_text = (f"Cohen's d = {cohens_d:.2f}\n"
                 f"Subclusters: {n_subclusters}\n"
                 f"WRMD var: {var_wrmd*100:.1f}%\n"
                 f"Ortho var: {var_ortho*100:.1f}%")
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Panel B: PCA (PC1 vs PC2)
    ax = axes[1]
    ax.scatter(pca_safe[:, 0], pca_safe[:, 1], c="#2196F3", alpha=0.5, s=20,
               label=f"Harmless", edgecolors="none")
    ax.scatter(pca_harm[:, 0], pca_harm[:, 1], c="#F44336", alpha=0.5, s=20,
               label=f"Harmful", edgecolors="none")

    ax.set_xlabel(f"PC1 (var={var_explained_pca[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 (var={var_explained_pca[1]*100:.1f}%)")
    ax.set_title(f"Layer {layer_idx} ({attn_type}) — PCA view")
    ax.legend(loc="upper left", fontsize=8)

    # Annotate PCA info
    pca_text = (f"cos(WRMD, PC1) = {cos_wrmd_pc1:.3f}\n"
                f"PC1 var: {var_explained_pca[0]*100:.1f}%\n"
                f"PC2 var: {var_explained_pca[1]*100:.1f}%\n"
                f"Top-5 var: {sum(var_explained_pca)*100:.1f}%")
    ax.text(0.98, 0.02, pca_text, transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle(f"Qwen 3.5-27B — Refusal Direction Diagnostic — Layer {layer_idx}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"diagnostic_L{layer_idx:02d}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Return metrics for JSON
    return {
        "layer": layer_idx,
        "attention_type": attn_type,
        "cohens_d": float(cohens_d),
        "var_explained_wrmd": float(var_wrmd),
        "var_explained_ortho": float(var_ortho),
        "var_explained_pc1": float(var_explained_pca[0]),
        "var_explained_pc2": float(var_explained_pca[1]),
        "var_explained_top5": float(sum(var_explained_pca)),
        "cos_wrmd_pc1": float(cos_wrmd_pc1),
        "n_subclusters_harmful": int(n_subclusters),
        "boundary_prompts": boundary_info,
    }


def plot_variance_summary(all_layer_metrics, output_dir):
    """Summary plot: variance explained by WRMD direction across all layers."""
    layers = [m["layer"] for m in all_layer_metrics]
    var_wrmd = [m["var_explained_wrmd"] * 100 for m in all_layer_metrics]
    var_ortho = [m["var_explained_ortho"] * 100 for m in all_layer_metrics]
    var_pc1 = [m["var_explained_pc1"] * 100 for m in all_layer_metrics]
    cos_pc1 = [m["cos_wrmd_pc1"] for m in all_layer_metrics]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.bar(layers, var_wrmd, width=0.8, color="#F44336", alpha=0.7, label="WRMD direction")
    ax.bar(layers, var_ortho, width=0.8, bottom=var_wrmd, color="#2196F3", alpha=0.5, label="Best orthogonal")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("Variance Captured by Refusal Direction vs Orthogonal")
    ax.legend()

    ax = axes[1]
    ax.plot(layers, cos_pc1, "o-", color="#4CAF50", markersize=4, label="cos(WRMD, PC1)")
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="0.95 threshold")
    ax.set_ylabel("Cosine similarity")
    ax.set_xlabel("Layer")
    ax.set_title("Alignment of WRMD Direction with PC1")
    ax.legend()
    ax.set_ylim(0, 1.05)

    fig.suptitle("Qwen 3.5-27B — Refusal Dimensionality Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "diagnostic_variance_summary.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def parse_layers(s, best_layer=None, n_layers=64):
    """Parse layer specification. 'best' resolves to best_layer from results."""
    layers = []
    for part in s.split(","):
        part = part.strip()
        if part == "best":
            if best_layer is not None:
                layers.append(best_layer)
            else:
                layers.append(32)  # default
        elif part == "all":
            return list(range(n_layers))
        elif "-" in part:
            a, b = part.split("-")
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def main():
    parser = argparse.ArgumentParser(description="Diagnostic scatter plots for refusal direction analysis")
    parser.add_argument("--activations", help="Path to saved activations .pt (from --save-activations)")
    parser.add_argument("--directions", help="Path to per-layer WRMD directions .pt")
    parser.add_argument("--results", help="Path to wrmd_results_qwen35.json (for best layer)")
    parser.add_argument("--layers", default="best,20,32,42,50",
                        help="Layers to plot: 'best,20,32' or 'all' or '20-50'")
    parser.add_argument("--output-dir", default="/tmp/diagnostic_plots")
    # Standalone mode (without pre-saved activations)
    parser.add_argument("--model-path", help="Model path for standalone extraction")
    parser.add_argument("--n-harmful", type=int, default=100)
    parser.add_argument("--n-harmless", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not HAS_MPL:
        print("ERROR: matplotlib required. pip install matplotlib")
        sys.exit(1)

    setup_style()
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine best layer from results
    best_layer = None
    if args.results:
        with open(args.results) as f:
            results = json.load(f)
        best_layer = results.get("best_layer", 32)
        print(f"Best layer from results: L{best_layer}")

    # Load or extract activations
    if args.activations:
        print(f"Loading saved activations from {args.activations}...")
        saved = torch.load(args.activations, map_location="cpu", weights_only=False)
        harmful_states = saved["harmful_states"]   # list of [N, hidden] per layer
        harmless_states = saved["harmless_states"]
        harmful_prompts = saved.get("harmful_prompts")
        harmless_prompts = saved.get("harmless_prompts")
        n_layers = len(harmful_states)
        print(f"  {n_layers} layers, {harmful_states[0].shape[0]} harmful, "
              f"{harmless_states[0].shape[0]} harmless")
    elif args.model_path:
        print(f"Standalone mode: extracting from {args.model_path}...")
        # Import extraction functions from sibling script
        sys.path.insert(0, os.path.dirname(__file__))
        from extract_wrmd_qwen35 import (
            load_prompts_from_datasets, extract_hidden_states,
            HARMFUL_PROMPTS_FALLBACK, HARMLESS_PROMPTS_FALLBACK
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import gc

        harmful_prompts, harmless_prompts = load_prompts_from_datasets(
            args.n_harmful, args.n_harmless
        )
        if harmful_prompts is None:
            harmful_prompts = HARMFUL_PROMPTS_FALLBACK[:args.n_harmful]
            harmless_prompts = HARMLESS_PROMPTS_FALLBACK[:args.n_harmless]

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()

        print(f"Extracting harmful ({len(harmful_prompts)})...")
        harmful_states = extract_hidden_states(model, tokenizer, harmful_prompts, args.device)
        print(f"Extracting harmless ({len(harmless_prompts)})...")
        harmless_states = extract_hidden_states(model, tokenizer, harmless_prompts, args.device)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        n_layers = len(harmful_states)
    else:
        print("ERROR: provide either --activations or --model-path")
        sys.exit(1)

    # Load WRMD directions
    if args.directions:
        print(f"Loading WRMD directions from {args.directions}...")
        dirs = torch.load(args.directions, map_location="cpu", weights_only=True)
        if dirs.dim() == 3:
            dirs = dirs[:, 0, :]  # [n_layers, hidden] — take primary direction
        print(f"  Shape: {dirs.shape}")
    else:
        # Compute on the fly
        print("No pre-computed directions, computing diff-of-means per layer...")
        dirs = torch.zeros(n_layers, harmful_states[0].shape[1])
        for l in range(n_layers):
            delta = harmful_states[l].mean(0) - harmless_states[l].mean(0)
            dirs[l] = delta / delta.norm()

    # Parse layers
    layers = parse_layers(args.layers, best_layer, n_layers)
    print(f"\nGenerating diagnostic plots for layers: {layers}")
    print(f"Output directory: {args.output_dir}\n")

    # Generate per-layer plots
    all_metrics = []
    for layer_idx in layers:
        if layer_idx >= n_layers:
            print(f"  Skipping L{layer_idx} (model has {n_layers} layers)")
            continue
        if dirs[layer_idx].norm() < 1e-8:
            print(f"  Skipping L{layer_idx} (zero direction)")
            continue

        metrics = plot_layer_diagnostic(
            harmful_states[layer_idx], harmless_states[layer_idx],
            dirs[layer_idx], layer_idx, args.output_dir,
            harmful_prompts if 'harmful_prompts' in dir() else None,
            harmless_prompts if 'harmless_prompts' in dir() else None,
        )
        all_metrics.append(metrics)

    # Summary plot (if multiple layers)
    if len(all_metrics) > 3:
        plot_variance_summary(all_metrics, args.output_dir)

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, "diagnostic_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics: {metrics_path}")

    # Print interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    for m in all_metrics:
        l = m["layer"]
        print(f"\n  Layer {l} ({m['attention_type']}):")
        print(f"    Cohen's d = {m['cohens_d']:.2f}")
        print(f"    WRMD direction captures {m['var_explained_wrmd']*100:.1f}% of total variance")
        print(f"    cos(WRMD, PC1) = {m['cos_wrmd_pc1']:.3f}")

        if m["cos_wrmd_pc1"] > 0.9:
            print(f"    -> WRMD aligns with max-variance direction (refusal IS the dominant signal)")
        elif m["cos_wrmd_pc1"] > 0.5:
            print(f"    -> WRMD partially aligns with PC1 (refusal is a secondary signal)")
        else:
            print(f"    -> WRMD is orthogonal to PC1 (refusal is a subtle, low-variance feature)")
            print(f"       This is NORMAL — PCA finds topic variance, not refusal variance")

        if m["var_explained_wrmd"] > 0.5:
            print(f"    -> Refusal is the DOMINANT feature (>50% variance) — single vector sufficient")
        elif m["var_explained_wrmd"] > 0.1:
            print(f"    -> Refusal explains moderate variance — single vector likely sufficient")
        else:
            print(f"    -> Refusal is a SUBTLE feature (<10% variance) — Cohen's d is the better metric")

        if m["n_subclusters_harmful"] > 1:
            print(f"    -> FOUND {m['n_subclusters_harmful']} subclusters in harmful activations!")
            print(f"       Consider multi-vector extraction (k={m['n_subclusters_harmful']})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
