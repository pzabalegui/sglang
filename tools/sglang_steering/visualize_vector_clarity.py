#!/usr/bin/env python3
"""
Visualization of refusal direction vector clarity for GLM-4.7-FP8 (358B).

Generates publication-quality plots from sweep_results_full.json:
1. Per-layer metrics evolution (Cohen's d, accuracy, SNR, composite)
2. Projection distributions (harmful vs harmless per layer)
3. Separation gap analysis
4. Cosine dissimilarity evolution
5. Direction norm growth
6. Combined dashboard

Usage:
    python visualize_vector_clarity.py [--data PATH] [--output DIR] [--format png|pdf]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def load_data(data_path: str) -> list[dict]:
    with open(data_path) as f:
        return json.load(f)


def setup_style():
    """Set publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_per_layer_metrics(data: list[dict], output_dir: Path, fmt: str):
    """Plot Cohen's d, accuracy, SNR, and composite score per layer."""
    layers = [d["layer"] for d in data]
    depth_pct = [l / 92 * 100 for l in layers]
    cohens_d = [d["effect_size"] for d in data]
    accuracy = [min(d["accuracy"], 1.0) * 100 for d in data]
    snr = [d["snr"] for d in data]
    composite = [d["composite_score"] for d in data]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GLM-4.7-FP8: Metricas de Separabilidad por Capa", fontsize=15, fontweight="bold")

    # Cohen's d
    ax = axes[0, 0]
    ax.plot(layers, cohens_d, "o-", color="#2563eb", linewidth=2, markersize=6)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="d=0.8 (grande)")
    ax.axhline(y=1.2, color="gray", linestyle=":", alpha=0.5, label="d=1.2 (enorme)")
    peak_idx = np.argmax(cohens_d)
    ax.annotate(
        f"L{layers[peak_idx]}: d={cohens_d[peak_idx]:.1f}",
        xy=(layers[peak_idx], cohens_d[peak_idx]),
        xytext=(layers[peak_idx] + 3, cohens_d[peak_idx] + 0.5),
        arrowprops=dict(arrowstyle="->", color="#dc2626"),
        fontsize=10, color="#dc2626", fontweight="bold",
    )
    ax.set_xlabel("Capa")
    ax.set_ylabel("Cohen's d")
    ax.set_title("Tamano del Efecto (Cohen's d)")
    ax.legend(loc="lower right")

    # Accuracy
    ax = axes[0, 1]
    ax.plot(layers, accuracy, "s-", color="#16a34a", linewidth=2, markersize=6)
    ax.axhline(y=100, color="#16a34a", linestyle="--", alpha=0.3)
    ax.set_xlabel("Capa")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Clasificador Lineal")
    ax.set_ylim(65, 105)
    ax.fill_between(layers, accuracy, 100, alpha=0.1, color="#16a34a")

    # SNR
    ax = axes[1, 0]
    ax.plot(layers, snr, "D-", color="#ea580c", linewidth=2, markersize=6)
    ax.fill_between(layers, snr, alpha=0.15, color="#ea580c")
    ax.set_xlabel("Capa")
    ax.set_ylabel("SNR")
    ax.set_title("Relacion Senal-Ruido")

    # Composite
    ax = axes[1, 1]
    ax.plot(layers, composite, "^-", color="#7c3aed", linewidth=2, markersize=6)
    ax.fill_between(layers, composite, alpha=0.15, color="#7c3aed")
    ax.set_xlabel("Capa")
    ax.set_ylabel("Composite Score")
    ax.set_title("Score Compuesto")

    # Add secondary x-axis with depth %
    for ax in axes.flat:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(layers[::2])
        ax2.set_xticklabels([f"{l/92*100:.0f}%" for l in layers[::2]], fontsize=8, color="gray")
        ax2.set_xlabel("% Profundidad", fontsize=9, color="gray")

    plt.tight_layout()
    out = output_dir / f"vector_clarity_metrics.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_projection_distributions(data: list[dict], output_dir: Path, fmt: str):
    """Plot harmful vs harmless projection ranges per layer."""
    layers = [d["layer"] for d in data]
    n = len(layers)

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle("GLM-4.7-FP8: Distribuciones de Proyeccion por Capa", fontsize=15, fontweight="bold")

    width = 0.35
    x = np.arange(n)

    # Plot ranges as error bars
    harmful_means = [d["harmful_proj_mean"] for d in data]
    harmful_mins = [d["harmful_proj_min"] for d in data]
    harmful_maxs = [d["harmful_proj_max"] for d in data]
    harmful_stds = [d["harmful_proj_std"] for d in data]

    harmless_means = [d["harmless_proj_mean"] for d in data]
    harmless_mins = [d["harmless_proj_min"] for d in data]
    harmless_maxs = [d["harmless_proj_max"] for d in data]
    harmless_stds = [d["harmless_proj_std"] for d in data]

    # Harmful cluster
    yerr_harmful = [
        [m - mn for m, mn in zip(harmful_means, harmful_mins)],
        [mx - m for m, mx in zip(harmful_means, harmful_maxs)],
    ]
    ax.errorbar(
        x - width / 2, harmful_means, yerr=yerr_harmful,
        fmt="o", color="#dc2626", capsize=4, capthick=1.5,
        linewidth=1.5, markersize=7, label="Harmful (media +/- rango)",
    )

    # Harmless cluster
    yerr_harmless = [
        [m - mn for m, mn in zip(harmless_means, harmless_mins)],
        [mx - m for m, mx in zip(harmless_means, harmless_maxs)],
    ]
    ax.errorbar(
        x + width / 2, harmless_means, yerr=yerr_harmless,
        fmt="s", color="#2563eb", capsize=4, capthick=1.5,
        linewidth=1.5, markersize=7, label="Harmless (media +/- rango)",
    )

    # Shade separation gap
    for i in range(n):
        gap_top = harmful_mins[i]
        gap_bot = harmless_maxs[i]
        if gap_top > gap_bot:
            ax.fill_between(
                [i - 0.4, i + 0.4], gap_bot, gap_top,
                alpha=0.15, color="#16a34a",
            )

    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers], rotation=45, ha="right")
    ax.set_xlabel("Capa")
    ax.set_ylabel("Proyeccion sobre direccion de rechazo")
    ax.set_title("Separacion Harmful vs Harmless (rango completo)")
    ax.legend(loc="upper left")

    # Annotate the gap at L46
    l46_idx = next(i for i, d in enumerate(data) if d["layer"] == 46)
    gap_val = harmful_mins[l46_idx] - harmless_maxs[l46_idx]
    ax.annotate(
        f"Gap = {gap_val:.1f}\n(cero solapamiento)",
        xy=(l46_idx, (harmful_mins[l46_idx] + harmless_maxs[l46_idx]) / 2),
        xytext=(l46_idx + 2.5, 2),
        arrowprops=dict(arrowstyle="->", color="#16a34a", linewidth=2),
        fontsize=11, color="#16a34a", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdf4", edgecolor="#16a34a"),
    )

    plt.tight_layout()
    out = output_dir / f"vector_clarity_projections.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_separation_gap(data: list[dict], output_dir: Path, fmt: str):
    """Plot separation gap and direction norm evolution."""
    layers = [d["layer"] for d in data]
    gaps = [d["separation_gap"] for d in data]
    norms = [d["direction_norm"] for d in data]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.suptitle("GLM-4.7-FP8: Gap de Separacion y Norma de la Direccion", fontsize=15, fontweight="bold")

    color1 = "#16a34a"
    ax1.bar(layers, gaps, width=1.4, color=color1, alpha=0.6, label="Gap de separacion")
    ax1.set_xlabel("Capa")
    ax1.set_ylabel("Gap de separacion (unidades)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    # Annotate zero-overlap boundary
    first_no_overlap = next((i for i, g in enumerate(gaps) if g > 0), None)
    if first_no_overlap is not None and first_no_overlap > 0:
        ax1.axvline(x=layers[first_no_overlap] - 1, color="gray", linestyle="--", alpha=0.5)
        ax1.text(
            layers[first_no_overlap] - 1.5, max(gaps) * 0.9,
            "Sin solapamiento\na partir de aqui",
            fontsize=9, color="gray", ha="right",
        )

    ax2 = ax1.twinx()
    color2 = "#7c3aed"
    ax2.plot(layers, norms, "D-", color=color2, linewidth=2, markersize=6, label="Norma direccion")
    ax2.set_ylabel("Norma de la direccion", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    out = output_dir / f"vector_clarity_gap_norm.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_cosine_dissimilarity(data: list[dict], output_dir: Path, fmt: str):
    """Plot cosine dissimilarity evolution across layers."""
    layers = [d["layer"] for d in data]
    dissim = [d["cosine_dissimilarity"] for d in data]
    angles = [np.degrees(np.arccos(1 - d)) for d in dissim]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.suptitle("GLM-4.7-FP8: Divergencia Geometrica Harmful vs Harmless", fontsize=15, fontweight="bold")

    color1 = "#ea580c"
    ax1.plot(layers, dissim, "o-", color=color1, linewidth=2, markersize=7, label="Coseno disimilitud")
    ax1.fill_between(layers, dissim, alpha=0.15, color=color1)
    ax1.set_xlabel("Capa")
    ax1.set_ylabel("1 - cos(harmful, harmless)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#0891b2"
    ax2.plot(layers, angles, "s--", color=color2, linewidth=1.5, markersize=5, label="Angulo (grados)")
    ax2.set_ylabel("Angulo aproximado (grados)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Annotate L46
    l46_idx = next(i for i, d in enumerate(data) if d["layer"] == 46)
    ax1.annotate(
        f"L46: {dissim[l46_idx]:.3f} ({angles[l46_idx]:.1f}deg)",
        xy=(46, dissim[l46_idx]),
        xytext=(50, dissim[l46_idx] + 0.03),
        arrowprops=dict(arrowstyle="->", color="#dc2626"),
        fontsize=10, color="#dc2626", fontweight="bold",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    out = output_dir / f"vector_clarity_cosine.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_dashboard(data: list[dict], output_dir: Path, fmt: str):
    """Create a single-page dashboard combining all key metrics."""
    layers = [d["layer"] for d in data]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "GLM-4.7-FP8 (358B): Claridad del Vector de Rechazo - Dashboard",
        fontsize=16, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.35)

    # 1. Cohen's d
    ax = fig.add_subplot(gs[0, 0])
    cohens_d = [d["effect_size"] for d in data]
    ax.plot(layers, cohens_d, "o-", color="#2563eb", linewidth=2, markersize=5)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.4)
    peak_idx = np.argmax(cohens_d)
    ax.plot(layers[peak_idx], cohens_d[peak_idx], "o", color="#dc2626", markersize=10, zorder=5)
    ax.set_title(f"Cohen's d (pico: {cohens_d[peak_idx]:.1f})")
    ax.set_xlabel("Capa")

    # 2. Direction norm
    ax = fig.add_subplot(gs[0, 1])
    norms = [d["direction_norm"] for d in data]
    ax.plot(layers, norms, "D-", color="#7c3aed", linewidth=2, markersize=5)
    ax.fill_between(layers, norms, alpha=0.1, color="#7c3aed")
    ax.set_title(f"Norma direccion (L30: {norms[0]:.1f} -> L62: {norms[-1]:.1f})")
    ax.set_xlabel("Capa")

    # 3. SNR
    ax = fig.add_subplot(gs[0, 2])
    snr = [d["snr"] for d in data]
    ax.plot(layers, snr, "s-", color="#ea580c", linewidth=2, markersize=5)
    ax.fill_between(layers, snr, alpha=0.1, color="#ea580c")
    ax.set_title("SNR")
    ax.set_xlabel("Capa")

    # 4. Projection distributions (large panel)
    ax = fig.add_subplot(gs[1, :])
    harmful_means = [d["harmful_proj_mean"] for d in data]
    harmful_mins = [d["harmful_proj_min"] for d in data]
    harmful_maxs = [d["harmful_proj_max"] for d in data]
    harmless_means = [d["harmless_proj_mean"] for d in data]
    harmless_mins = [d["harmless_proj_min"] for d in data]
    harmless_maxs = [d["harmless_proj_max"] for d in data]

    x = np.arange(len(layers))
    ax.fill_between(x, harmful_mins, harmful_maxs, alpha=0.3, color="#dc2626", label="Harmful rango")
    ax.plot(x, harmful_means, "o-", color="#dc2626", linewidth=2, markersize=5)
    ax.fill_between(x, harmless_mins, harmless_maxs, alpha=0.3, color="#2563eb", label="Harmless rango")
    ax.plot(x, harmless_means, "s-", color="#2563eb", linewidth=2, markersize=5)
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_title("Proyecciones: Harmful (rojo) vs Harmless (azul) — rangos completos")
    ax.legend(loc="upper left")
    ax.set_ylabel("Proyeccion")

    # 5. Separation gap
    ax = fig.add_subplot(gs[2, 0])
    gaps = [d["separation_gap"] for d in data]
    colors = ["#16a34a" if g > 0 else "#dc2626" for g in gaps]
    ax.bar(layers, gaps, width=1.4, color=colors, alpha=0.7)
    ax.set_title("Gap de separacion")
    ax.set_xlabel("Capa")
    ax.set_ylabel("Gap (unidades)")

    # 6. Cosine dissimilarity
    ax = fig.add_subplot(gs[2, 1])
    dissim = [d["cosine_dissimilarity"] for d in data]
    ax.plot(layers, dissim, "o-", color="#ea580c", linewidth=2, markersize=5)
    ax.fill_between(layers, dissim, alpha=0.15, color="#ea580c")
    ax.set_title("Coseno disimilitud")
    ax.set_xlabel("Capa")

    # 7. Summary stats text
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    summary_text = (
        "RESUMEN L46 (capa optima)\n"
        "========================\n"
        f"Cohen's d:     {data[8]['effect_size']:.2f}\n"
        f"Accuracy:      100%\n"
        f"Gap:           {data[8]['separation_gap']:.2f}\n"
        f"SNR:           {data[8]['snr']:.3f}\n"
        f"Norma dir:     {data[8]['direction_norm']:.2f}\n"
        f"Cos dissim:    {data[8]['cosine_dissimilarity']:.3f}\n"
        "\n"
        "VEREDICTO\n"
        "========================\n"
        "Separacion:\n"
        "  EXTRAORDINARIA\n"
        f"  13 sigma, 0 solapamiento\n"
        f"  100% accuracy lineal\n"
    )
    ax.text(
        0.05, 0.95, summary_text,
        transform=ax.transAxes, fontsize=11, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0fdf4", edgecolor="#16a34a", alpha=0.8),
    )

    out = output_dir / f"vector_clarity_dashboard.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_l47_detail(data_l47: list[dict], output_dir: Path, fmt: str):
    """Detailed view of the L45-L49 region showing peak selection."""
    layers = [d["layer"] for d in data_l47]
    cohens_d = [d["effect_size"] for d in data_l47]
    gaps = [d["separation_gap"] for d in data_l47]
    norms = [d["direction_norm"] for d in data_l47]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("GLM-4.7-FP8: Detalle de la Zona Optima (L45-L49)", fontsize=14, fontweight="bold")

    # Cohen's d
    ax = axes[0]
    colors = ["#dc2626" if l == 46 else "#2563eb" for l in layers]
    ax.bar(layers, cohens_d, color=colors, alpha=0.8)
    ax.set_title("Cohen's d")
    ax.set_xlabel("Capa")
    ax.set_ylabel("d")

    # Gap
    ax = axes[1]
    colors = ["#dc2626" if l == 46 else "#16a34a" for l in layers]
    ax.bar(layers, gaps, color=colors, alpha=0.8)
    ax.set_title("Gap de Separacion")
    ax.set_xlabel("Capa")
    ax.set_ylabel("Gap")

    # Norm
    ax = axes[2]
    colors = ["#dc2626" if l == 46 else "#7c3aed" for l in layers]
    ax.bar(layers, norms, color=colors, alpha=0.8)
    ax.set_title("Norma Direccion")
    ax.set_xlabel("Capa")
    ax.set_ylabel("Norma")

    plt.tight_layout()
    out = output_dir / f"vector_clarity_L47_detail.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize refusal vector clarity")
    parser.add_argument(
        "--data",
        default="results/glm47_validation_31/sweep_results_full.json",
        help="Path to sweep_results_full.json",
    )
    parser.add_argument(
        "--data-l47",
        default="results/glm47_validation_31/sweep_results_L47.json",
        help="Path to sweep_results_L47.json (detailed L45-L49)",
    )
    parser.add_argument(
        "--output", "-o",
        default="results/plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--format", "-f",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {args.data}")
    data = load_data(args.data)
    print(f"  Loaded {len(data)} layers: L{data[0]['layer']} - L{data[-1]['layer']}")

    setup_style()

    print("\nGenerating plots...")
    plot_per_layer_metrics(data, output_dir, args.format)
    plot_projection_distributions(data, output_dir, args.format)
    plot_separation_gap(data, output_dir, args.format)
    plot_cosine_dissimilarity(data, output_dir, args.format)
    plot_dashboard(data, output_dir, args.format)

    # L47 detail if available
    data_l47_path = Path(args.data_l47)
    if data_l47_path.exists():
        print(f"\nLoading L47 detail from: {args.data_l47}")
        data_l47 = load_data(args.data_l47)
        plot_l47_detail(data_l47, output_dir, args.format)

    print(f"\nAll plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
