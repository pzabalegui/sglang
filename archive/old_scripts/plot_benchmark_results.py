#!/usr/bin/env python3
"""
Visualización de resultados del benchmark de abliteración.
Genera múltiples gráficos para analizar el efecto de la ortogonalización
por capa y factor de escala.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_data(filepath: str) -> pd.DataFrame:
    """Carga los datos del archivo de resumen."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                layer_scale = parts[0].replace(':', '')
                layer = int(layer_scale.split('_')[0][1:])
                scale = float(layer_scale.split('_')[1][1:])
                D = int(parts[1].split('=')[1])
                C = int(parts[2].split('=')[1])
                B = int(parts[3].split('=')[1])
                X = int(parts[4].split('=')[1])
                data.append({
                    'layer': layer,
                    'scale': scale,
                    'D': D, 'C': C, 'B': B, 'X': X,
                    'total': D + C + B + X
                })
    return pd.DataFrame(data)


def plot_heatmap_D(df: pd.DataFrame, output_dir: Path):
    """Heatmap de respuestas directas (D) - métrica principal de jailbreak."""
    pivot_D = df.pivot(index='layer', columns='scale', values='D')

    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(pivot_D, annot=True, cmap='RdYlGn', fmt='d',
                cbar_kws={'label': 'D (Respuestas Directas)'},
                linewidths=0.5, ax=ax)
    ax.set_title('Eficacia de Abliteración: Respuestas Directas (D)\nModelo GLM-4.7-Flash',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Factor de Escala', fontsize=12)
    ax.set_ylabel('Capa', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_D.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'heatmap_D.pdf', bbox_inches='tight')
    print(f"Guardado: heatmap_D.png/pdf")
    plt.close()


def plot_heatmap_all_categories(df: pd.DataFrame, output_dir: Path):
    """Heatmaps 2x2 para las 4 categorías."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 16))

    categories = [
        ('D', 'Directas (Jailbreak exitoso)', 'Reds'),
        ('C', 'Condicionadas (Parcial)', 'Oranges'),
        ('B', 'Degradadas (Loops/Errores)', 'Purples'),
        ('X', 'Censuradas (Rechazo)', 'Greens')
    ]

    for ax, (cat, title, cmap) in zip(axes.flat, categories):
        pivot = df.pivot(index='layer', columns='scale', values=cat)
        sns.heatmap(pivot, annot=True, cmap=cmap, fmt='d',
                    linewidths=0.5, ax=ax, vmin=0, vmax=30)
        ax.set_title(f'{cat}: {title}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Factor de Escala')
        ax.set_ylabel('Capa')

    plt.suptitle('Distribución de Respuestas por Categoría\nModelo GLM-4.7-Flash Abliterado',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_all_categories.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'heatmap_all_categories.pdf', bbox_inches='tight')
    print(f"Guardado: heatmap_all_categories.png/pdf")
    plt.close()


def plot_lineplot_D_by_layer(df: pd.DataFrame, output_dir: Path):
    """Line plot: evolución de D a través de las capas."""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']

    for i, scale in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
        subset = df[df['scale'] == scale].sort_values('layer')
        ax.plot(subset['layer'], subset['D'],
                marker=markers[i], markersize=8, linewidth=2,
                color=colors[i], label=f'Scale={scale}')

    ax.axhline(y=15, color='gray', linestyle='--', alpha=0.5, label='50% (15/30)')
    ax.axvspan(20, 25, alpha=0.2, color='green', label='Zona óptima')

    ax.set_xlabel('Capa', fontsize=12)
    ax.set_ylabel('D (Respuestas Directas)', fontsize=12)
    ax.set_title('Efecto de Abliteración: Respuestas Directas vs Capa\nModelo GLM-4.7-Flash',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(16, 31))
    ax.set_ylim(0, 30)

    plt.tight_layout()
    plt.savefig(output_dir / 'lineplot_D_by_layer.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'lineplot_D_by_layer.pdf', bbox_inches='tight')
    print(f"Guardado: lineplot_D_by_layer.png/pdf")
    plt.close()


def plot_stacked_composition(df: pd.DataFrame, output_dir: Path):
    """Stacked area: composición D/C/B/X por capa para cada scale."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)

    colors = ['#d62728', '#ff7f0e', '#9467bd', '#2ca02c']

    for i, scale in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
        subset = df[df['scale'] == scale].sort_values('layer')
        axes[i].stackplot(subset['layer'],
                          subset['D'], subset['C'], subset['B'], subset['X'],
                          labels=['D (Directas)', 'C (Condicionadas)',
                                  'B (Degradadas)', 'X (Censuradas)'],
                          colors=colors, alpha=0.8)
        axes[i].set_title(f'Scale = {scale}', fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Capa')
        axes[i].set_xticks(range(16, 31, 2))
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].set_ylabel('Respuestas (n=30)')
            axes[i].legend(loc='upper left', fontsize=8)

    plt.suptitle('Distribución de Tipos de Respuesta por Capa y Factor de Escala',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'stacked_composition.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'stacked_composition.pdf', bbox_inches='tight')
    print(f"Guardado: stacked_composition.png/pdf")
    plt.close()


def plot_optimal_zone(df: pd.DataFrame, output_dir: Path):
    """Análisis de zona óptima con barras agrupadas."""
    # Agrupar por capa (promedio de todos los scales)
    df_by_layer = df.groupby('layer')[['D', 'C', 'B', 'X']].mean().reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(df_by_layer))
    width = 0.2

    bars_D = ax.bar(x - 1.5*width, df_by_layer['D'], width, label='D (Directas)', color='#d62728')
    bars_C = ax.bar(x - 0.5*width, df_by_layer['C'], width, label='C (Condicionadas)', color='#ff7f0e')
    bars_B = ax.bar(x + 0.5*width, df_by_layer['B'], width, label='B (Degradadas)', color='#9467bd')
    bars_X = ax.bar(x + 1.5*width, df_by_layer['X'], width, label='X (Censuradas)', color='#2ca02c')

    # Marcar zona óptima
    optimal_start = list(df_by_layer['layer']).index(21)
    optimal_end = list(df_by_layer['layer']).index(25)
    ax.axvspan(optimal_start - 0.5, optimal_end + 0.5, alpha=0.15, color='green')
    ax.annotate('Zona Óptima\n(L21-L25)', xy=(optimal_start + 2, 25), fontsize=10,
                ha='center', color='darkgreen', fontweight='bold')

    ax.set_xlabel('Capa', fontsize=12)
    ax.set_ylabel('Promedio de Respuestas', fontsize=12)
    ax.set_title('Promedio de Respuestas por Capa (todos los factores de escala)\nModelo GLM-4.7-Flash',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_by_layer['layer'])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 30)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'optimal_zone_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'optimal_zone_analysis.pdf', bbox_inches='tight')
    print(f"Guardado: optimal_zone_analysis.png/pdf")
    plt.close()


def plot_scale_effect(df: pd.DataFrame, output_dir: Path):
    """Box plot: efecto del factor de escala en D."""
    fig, ax = plt.subplots(figsize=(10, 6))

    df_plot = df.copy()
    df_plot['scale_str'] = df_plot['scale'].apply(lambda x: f'{x:.1f}')

    sns.boxplot(data=df_plot, x='scale_str', y='D', ax=ax, palette='viridis')
    sns.stripplot(data=df_plot, x='scale_str', y='D', ax=ax,
                  color='black', alpha=0.5, size=4)

    ax.set_xlabel('Factor de Escala', fontsize=12)
    ax.set_ylabel('D (Respuestas Directas)', fontsize=12)
    ax.set_title('Efecto del Factor de Escala en Respuestas Directas\n(distribución sobre todas las capas)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'scale_effect_boxplot.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'scale_effect_boxplot.pdf', bbox_inches='tight')
    print(f"Guardado: scale_effect_boxplot.png/pdf")
    plt.close()


def plot_summary_statistics(df: pd.DataFrame, output_dir: Path):
    """Tabla resumen con estadísticas clave."""
    # Encontrar configuración óptima
    best_config = df.loc[df['D'].idxmax()]

    # Estadísticas por capa
    stats_by_layer = df.groupby('layer')['D'].agg(['mean', 'std', 'max']).round(2)
    best_layers = stats_by_layer.nlargest(5, 'mean')

    # Crear figura con texto
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    RESUMEN DE RESULTADOS - ABLITERACIÓN GLM-4.7-Flash              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  CONFIGURACIÓN ÓPTIMA:                                                       ║
    ║    • Capa: L{int(best_config['layer'])}                                                             ║
    ║    • Factor de escala: {best_config['scale']}                                               ║
    ║    • Respuestas directas (D): {int(best_config['D'])}/30 ({best_config['D']/30*100:.1f}%)                              ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  TOP 5 CAPAS (promedio D sobre todos los scales):                            ║
    ║    1. L{best_layers.index[0]}: D̄={best_layers.iloc[0]['mean']:.1f} ± {best_layers.iloc[0]['std']:.1f}  (máx: {int(best_layers.iloc[0]['max'])})                             ║
    ║    2. L{best_layers.index[1]}: D̄={best_layers.iloc[1]['mean']:.1f} ± {best_layers.iloc[1]['std']:.1f}  (máx: {int(best_layers.iloc[1]['max'])})                             ║
    ║    3. L{best_layers.index[2]}: D̄={best_layers.iloc[2]['mean']:.1f} ± {best_layers.iloc[2]['std']:.1f}  (máx: {int(best_layers.iloc[2]['max'])})                             ║
    ║    4. L{best_layers.index[3]}: D̄={best_layers.iloc[3]['mean']:.1f} ± {best_layers.iloc[3]['std']:.1f}  (máx: {int(best_layers.iloc[3]['max'])})                             ║
    ║    5. L{best_layers.index[4]}: D̄={best_layers.iloc[4]['mean']:.1f} ± {best_layers.iloc[4]['std']:.1f}  (máx: {int(best_layers.iloc[4]['max'])})                             ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  OBSERVACIONES:                                                              ║
    ║    • Zona óptima: Capas 21-25 (>70% respuestas directas)                     ║
    ║    • Capas tempranas (16-18): Alto rechazo (X), bajo jailbreak               ║
    ║    • Capa 30: Decaimiento significativo (D≈5, regresa a C)                   ║
    ║    • Factor de escala 0.3-0.5 generalmente óptimo                            ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """

    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'summary_statistics.png', dpi=150, bbox_inches='tight')
    print(f"Guardado: summary_statistics.png")
    plt.close()

    return best_config, best_layers


def main():
    # Rutas
    project_root = Path(__file__).parent.parent
    data_file = project_root / 'results' / 'benchmark_results_summary.txt'
    output_dir = project_root / 'results' / 'plots'
    output_dir.mkdir(exist_ok=True)

    print(f"Cargando datos de: {data_file}")
    df = load_data(data_file)
    print(f"Datos cargados: {len(df)} configuraciones (capas {df['layer'].min()}-{df['layer'].max()})")

    print("\nGenerando visualizaciones...")

    # Generar todos los plots
    plot_heatmap_D(df, output_dir)
    plot_heatmap_all_categories(df, output_dir)
    plot_lineplot_D_by_layer(df, output_dir)
    plot_stacked_composition(df, output_dir)
    plot_optimal_zone(df, output_dir)
    plot_scale_effect(df, output_dir)
    best_config, best_layers = plot_summary_statistics(df, output_dir)

    print(f"\n{'='*60}")
    print("RESULTADOS CLAVE:")
    print(f"{'='*60}")
    print(f"Configuración óptima: L{int(best_config['layer'])}_S{best_config['scale']}")
    print(f"Máximo D alcanzado: {int(best_config['D'])}/30 ({best_config['D']/30*100:.1f}%)")
    print(f"\nTodos los gráficos guardados en: {output_dir}")


if __name__ == '__main__':
    main()
