#!/usr/bin/env python3
"""
Visualizaciones científicas para el informe de abliteración GLM-4.7.
Genera figuras estilo paper académico usando datos experimentales reales.

Figuras generadas:
- Fig 1: PCA de activaciones harmful vs harmless
- Fig 3: Layer sweep (effect size, accuracy, composite score)
- Fig 4: Comparación ASR baseline vs abliterated
- Fig 5: Gaussian kernel weights

Autor: Paul Zabalegui (Alias Robotics)
Fecha: 2026-02-11
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path
import json

# Configuración de estilo profesional
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Paleta de colores consistente
COLORS = {
    'harmful': '#d62728',      # Rojo
    'harmless': '#2ca02c',     # Verde
    'direction': '#1f77b4',    # Azul
    'baseline': '#ff7f0e',     # Naranja
    'abliterated': '#9467bd',  # Púrpura
    'optimal': '#e377c2',      # Rosa
    'kernel': '#17becf',       # Cyan
}


def load_data(data_dir: Path) -> dict:
    """Carga todos los datos necesarios para las visualizaciones."""
    data = {}

    # Activaciones L47
    act_path = data_dir / "activations_L47.pt"
    if act_path.exists():
        data['activations'] = torch.load(act_path, map_location='cpu')
        print(f"✓ Activaciones cargadas: harmful={data['activations']['harmful_activations'].shape}")

    # Sweep data
    sweep_path = data_dir / "sweep_data.pt"
    if sweep_path.exists():
        data['sweep'] = torch.load(sweep_path, map_location='cpu')
        print(f"✓ Sweep cargado: {len(data['sweep']['layers'])} capas")

    # Result final
    result_path = data_dir / "result_final.json"
    if result_path.exists():
        with open(result_path) as f:
            data['result'] = json.load(f)
        print(f"✓ Resultados finales cargados")

    return data


def plot_pca_activations(data: dict, output_dir: Path):
    """
    Figura 1: PCA de activaciones harmful vs harmless.
    Estilo similar al paper 2406.11717 (Arditi et al.)
    """
    if 'activations' not in data:
        print("⚠ No hay datos de activaciones para PCA")
        return

    act = data['activations']
    harmful = act['harmful_activations'].numpy()
    harmless = act['harmless_activations'].numpy()
    direction = act['direction'].numpy()

    # Combinar para PCA
    all_activations = np.vstack([harmful, harmless])
    labels = np.array(['harmful'] * len(harmful) + ['harmless'] * len(harmless))

    # PCA a 2D
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_activations)

    # Proyectar centroides y dirección
    harmful_centroid = harmful.mean(axis=0)
    harmless_centroid = harmless.mean(axis=0)

    harmful_proj = pca.transform(harmful_centroid.reshape(1, -1))[0]
    harmless_proj = pca.transform(harmless_centroid.reshape(1, -1))[0]

    # La dirección de rechazo en el espacio PCA
    direction_in_original = direction * act['direction_norm']
    direction_start = harmless_centroid
    direction_end = harmful_centroid

    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    harmful_mask = labels == 'harmful'
    harmless_mask = labels == 'harmless'

    ax.scatter(projected[harmless_mask, 0], projected[harmless_mask, 1],
               c=COLORS['harmless'], alpha=0.6, s=60, label='Harmless prompts',
               edgecolors='white', linewidth=0.5)
    ax.scatter(projected[harmful_mask, 0], projected[harmful_mask, 1],
               c=COLORS['harmful'], alpha=0.6, s=60, label='Harmful prompts',
               edgecolors='white', linewidth=0.5)

    # Centroides
    ax.scatter(*harmless_proj, c=COLORS['harmless'], s=200, marker='*',
               edgecolors='black', linewidth=1.5, zorder=5, label='Harmless centroid')
    ax.scatter(*harmful_proj, c=COLORS['harmful'], s=200, marker='*',
               edgecolors='black', linewidth=1.5, zorder=5, label='Harmful centroid')

    # Flecha del refusal direction
    ax.annotate('', xy=harmful_proj, xytext=harmless_proj,
                arrowprops=dict(arrowstyle='->', color=COLORS['direction'],
                               lw=3, mutation_scale=20))

    # Etiqueta de la dirección
    mid_point = (harmful_proj + harmless_proj) / 2
    ax.annotate(f'Refusal Direction\n(||r|| = {act["direction_norm"]:.2f})',
                xy=mid_point, xytext=(mid_point[0] + 1, mid_point[1] + 1),
                fontsize=11, fontweight='bold', color=COLORS['direction'],
                arrowprops=dict(arrowstyle='->', color=COLORS['direction'], lw=1))

    # Varianza explicada
    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% variance)')

    ax.set_title('Activation Space: Harmful vs Harmless Prompts\n(GLM-4.7, Layer 47)',
                 fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)

    # Añadir información
    info_text = f'n_harmful = {len(harmful)}\nn_harmless = {len(harmless)}\nhidden_dim = {harmful.shape[1]}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Guardar
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig1_pca_activations.{fmt}')
    print(f"✓ Figura 1 guardada: fig1_pca_activations.pdf/png")
    plt.close()


def plot_layer_sweep(data: dict, output_dir: Path):
    """
    Figura 3: Análisis del sweep de capas.
    Muestra effect_size, accuracy y composite_score por capa.
    """
    if 'sweep' not in data:
        print("⚠ No hay datos de sweep")
        return

    sweep = data['sweep']
    layers = np.array(sweep['layers'])
    effect_sizes = np.array(sweep['effect_sizes'])
    accuracies = np.array(sweep['accuracies'])
    composite = np.array(sweep['composite_scores'])
    snr = np.array(sweep['snr'])

    # Encontrar capa óptima
    best_idx = np.argmax(composite)
    best_layer = layers[best_idx]

    # Crear figura con 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- Subplot 1: Effect Size y SNR ---
    ax1 = axes[0]

    line1, = ax1.plot(layers, effect_sizes, 'o-', color=COLORS['harmful'],
                      linewidth=2, markersize=6, label='Effect Size (Cohen\'s d)')
    ax1.fill_between(layers, effect_sizes, alpha=0.2, color=COLORS['harmful'])

    ax1_twin = ax1.twinx()
    line2, = ax1_twin.plot(layers, snr, 's--', color=COLORS['harmless'],
                           linewidth=2, markersize=5, label='SNR')

    # Marcar capa óptima
    ax1.axvline(x=best_layer, color=COLORS['optimal'], linestyle=':', linewidth=2,
                label=f'Optimal Layer (L{best_layer})')

    ax1.set_ylabel('Effect Size (Cohen\'s d)', color=COLORS['harmful'])
    ax1_twin.set_ylabel('Signal-to-Noise Ratio', color=COLORS['harmless'])
    ax1.tick_params(axis='y', labelcolor=COLORS['harmful'])
    ax1_twin.tick_params(axis='y', labelcolor=COLORS['harmless'])

    # Leyenda combinada
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    ax1.set_title('Layer Analysis: Direction Quality Metrics', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: Accuracy y Composite Score ---
    ax2 = axes[1]

    line3, = ax2.plot(layers, accuracies * 100, 'o-', color=COLORS['direction'],
                      linewidth=2, markersize=6, label='Classification Accuracy (%)')
    ax2.fill_between(layers, accuracies * 100, alpha=0.2, color=COLORS['direction'])

    ax2_twin = ax2.twinx()
    line4, = ax2_twin.plot(layers, composite, 'D-', color=COLORS['abliterated'],
                           linewidth=2, markersize=5, label='Composite Score')

    # Marcar capa óptima
    ax2.axvline(x=best_layer, color=COLORS['optimal'], linestyle=':', linewidth=2)
    ax2.scatter([best_layer], [accuracies[best_idx] * 100], s=150, c=COLORS['optimal'],
                zorder=5, edgecolors='black', linewidth=2)

    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Accuracy (%)', color=COLORS['direction'])
    ax2_twin.set_ylabel('Composite Score', color=COLORS['abliterated'])
    ax2.tick_params(axis='y', labelcolor=COLORS['direction'])
    ax2_twin.tick_params(axis='y', labelcolor=COLORS['abliterated'])

    lines = [line3, line4]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')

    ax2.set_title('Layer Analysis: Performance Metrics', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Añadir anotación de capa óptima
    ax2.annotate(f'L{best_layer}\n(optimal)',
                 xy=(best_layer, accuracies[best_idx] * 100),
                 xytext=(best_layer + 3, accuracies[best_idx] * 100 + 5),
                 fontsize=11, fontweight='bold', color=COLORS['optimal'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['optimal']))

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig3_layer_sweep.{fmt}')
    print(f"✓ Figura 3 guardada: fig3_layer_sweep.pdf/png")
    plt.close()


def plot_asr_comparison(data: dict, output_dir: Path):
    """
    Figura 4: Comparación de tasas de rechazo baseline vs abliterated.
    """
    if 'result' not in data:
        print("⚠ No hay datos de resultados")
        return

    result = data['result']
    baseline = result['baseline_metrics']
    ortho = result['ortho_metrics']

    # Datos para el gráfico
    categories = ['Refusal Rate', 'Bypass Rate', 'ASR']
    baseline_values = [
        baseline['refusal_rate'] * 100,
        0,  # No aplica
        0   # No aplica
    ]
    abliterated_values = [
        ortho['intervention_refusal_rate'] * 100,
        ortho['bypass_rate'] * 100,
        ortho['attack_success_rate'] * 100
    ]

    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline (Original)',
                   color=COLORS['baseline'], edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, abliterated_values, width, label='Abliterated',
                   color=COLORS['abliterated'], edgecolor='black', linewidth=1)

    # Etiquetas en las barras
    def autolabel(bars, values):
        for bar, val in zip(bars, values):
            if val > 0:
                height = bar.get_height()
                ax.annotate(f'{val:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontweight='bold', fontsize=11)

    autolabel(bars1, baseline_values)
    autolabel(bars2, abliterated_values)

    ax.set_ylabel('Percentage (%)')
    ax.set_title('Abliteration Effectiveness: GLM-4.7 (358B Parameters)\nLayer 47, Gaussian Kernel (width=10), attn_scale=5.0',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')

    # Añadir flechas de reducción
    ax.annotate('', xy=(0 + width/2, abliterated_values[0] + 2),
                xytext=(0 - width/2, baseline_values[0] - 2),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(0, (baseline_values[0] + abliterated_values[0]) / 2,
            f'↓ {baseline_values[0] - abliterated_values[0]:.1f}%',
            ha='center', fontsize=10, color='green', fontweight='bold')

    # Información adicional
    info_text = f"""Test set: {ortho['total_prompts']} prompts
Baseline refusals: {ortho['baseline_refusals']}/{ortho['total_prompts']}
After abliteration: {ortho['intervention_refusals']}/{ortho['total_prompts']}"""
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=9,
            ha='right', va='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig4_asr_comparison.{fmt}')
    print(f"✓ Figura 4 guardada: fig4_asr_comparison.pdf/png")
    plt.close()


def plot_gaussian_kernel(data: dict, output_dir: Path):
    """
    Figura 5: Visualización del kernel Gaussiano usado para multi-layer abliteration.
    """
    if 'result' not in data:
        print("⚠ No hay datos de resultados")
        return

    result = data['result']
    layer_weights = result.get('layer_weights', {})

    if not layer_weights:
        print("⚠ No hay layer_weights en los resultados")
        return

    # Extraer datos
    layers = sorted([int(k) for k in layer_weights.keys()])
    weights = [layer_weights[str(l)] for l in layers]

    # Parámetros del kernel
    peak_layer = result['config'].get('layer', 47)
    kernel_width = result['config'].get('kernel_width', 10)

    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 6))

    # Graficar weights reales
    ax.fill_between(layers, weights, alpha=0.3, color=COLORS['kernel'])
    ax.plot(layers, weights, 'o-', color=COLORS['kernel'], linewidth=2,
            markersize=6, label='Applied Weights')

    # Línea teórica del kernel Gaussiano
    x_smooth = np.linspace(min(layers), max(layers), 200)
    gaussian_theoretical = np.exp(-((x_smooth - peak_layer) ** 2) / (2 * kernel_width ** 2))
    ax.plot(x_smooth, gaussian_theoretical, '--', color=COLORS['direction'],
            linewidth=2, label=f'Gaussian Kernel (μ={peak_layer}, σ={kernel_width})')

    # Marcar capa pico
    ax.axvline(x=peak_layer, color=COLORS['optimal'], linestyle=':', linewidth=2)
    ax.scatter([peak_layer], [1.0], s=150, c=COLORS['optimal'], zorder=5,
               edgecolors='black', linewidth=2)
    ax.annotate(f'Peak: L{peak_layer}',
                xy=(peak_layer, 1.0), xytext=(peak_layer + 5, 0.95),
                fontsize=11, fontweight='bold', color=COLORS['optimal'],
                arrowprops=dict(arrowstyle='->', color=COLORS['optimal']))

    # Marcar ancho del kernel (FWHM)
    fwhm_weight = 0.5
    ax.axhline(y=fwhm_weight, color='gray', linestyle='--', alpha=0.5)
    ax.text(max(layers) - 5, fwhm_weight + 0.02, 'FWHM', fontsize=9, color='gray')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Intervention Weight')
    ax.set_title('Gaussian Kernel for Multi-Layer Abliteration\nSmooth intervention prevents coherence degradation',
                 fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # Fórmula del kernel
    formula = r'$w(l) = \exp\left(-\frac{(l - l_{peak})^2}{2\sigma^2}\right)$'
    ax.text(0.02, 0.98, formula, transform=ax.transAxes, fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Info del modelo
    info_text = f"""Model: GLM-4.7 (92 layers)
Peak layer: {peak_layer}
Kernel width: {kernel_width}
Layers affected: {len(layers)}
Min weight: {min(weights):.4f}"""
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=9,
            ha='right', va='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig5_gaussian_kernel.{fmt}')
    print(f"✓ Figura 5 guardada: fig5_gaussian_kernel.pdf/png")
    plt.close()


def plot_examples_comparison(data: dict, output_dir: Path):
    """
    Figura adicional: Comparación visual de respuestas antes/después.
    """
    if 'result' not in data:
        return

    examples = data['result'].get('examples', [])
    if not examples:
        print("⚠ No hay ejemplos en los resultados")
        return

    # Seleccionar 3 ejemplos representativos
    selected = examples[:3]

    fig, axes = plt.subplots(len(selected), 1, figsize=(14, 4 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for i, (ax, ex) in enumerate(zip(axes, selected)):
        ax.axis('off')

        instruction = ex['instruction'][:100] + '...' if len(ex['instruction']) > 100 else ex['instruction']
        baseline = ex['baseline'][:300] + '...' if len(ex['baseline']) > 300 else ex['baseline']
        ortho = ex['ortho'][:300] + '...' if len(ex['ortho']) > 300 else ex['ortho']

        text = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT: {instruction}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔴 BASELINE (Refused: {ex['baseline_is_refusal']}):
{baseline}

🟢 ABLITERATED (Refused: {ex['ortho_is_refusal']}):
{ortho}
"""
        ax.text(0.01, 0.99, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                wrap=True)

    plt.suptitle('Response Comparison: Baseline vs Abliterated', fontweight='bold', fontsize=14)
    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig_examples_comparison.{fmt}')
    print(f"✓ Figura de ejemplos guardada: fig_examples_comparison.pdf/png")
    plt.close()


def main():
    """Genera todas las visualizaciones."""
    # Rutas
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "results" / "glm47_final"
    output_dir = data_dir / "figures"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("GENERACIÓN DE VISUALIZACIONES - ABLITERACIÓN GLM-4.7")
    print("=" * 60)
    print(f"\nDirectorio de datos: {data_dir}")
    print(f"Directorio de salida: {output_dir}\n")

    # Cargar datos
    print("Cargando datos...")
    data = load_data(data_dir)
    print()

    # Generar figuras
    print("Generando figuras...")
    print("-" * 40)

    plot_pca_activations(data, output_dir)
    plot_layer_sweep(data, output_dir)
    plot_asr_comparison(data, output_dir)
    plot_gaussian_kernel(data, output_dir)
    plot_examples_comparison(data, output_dir)

    print("-" * 40)
    print(f"\n✅ Todas las figuras generadas en: {output_dir}")
    print("\nPróximos pasos:")
    print("  1. Revisar las figuras generadas")
    print("  2. Crear diagramas TikZ para figuras teóricas")
    print("  3. Compilar informe técnico completo")


if __name__ == "__main__":
    main()
