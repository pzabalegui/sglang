"""
Visualizaciones de conceptos vectoriales:
- Ortogonalización vs Ablación direccional
- PCA vs diferencia de medias
- Direcciones por capa
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configuración de estilo
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['font.size'] = 10

def plot_orthogonalization_problem():
    """
    Muestra por qué la ortogonalización simple puede fallar
    cuando la magnitud del componente de rechazo varía.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # === Panel 1: Ortogonalización ideal ===
    ax1 = axes[0]
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='gray', linewidth=0.5)
    ax1.axvline(x=0, color='gray', linewidth=0.5)

    # Dirección de rechazo (normalizada)
    d = np.array([1, 1]) / np.sqrt(2)

    # Vector original
    v = np.array([2, 1.5])

    # Proyección
    proj = np.dot(v, d) * d
    v_orth = v - proj

    ax1.arrow(0, 0, d[0]*2, d[1]*2, head_width=0.1, head_length=0.1,
              fc='red', ec='red', linewidth=2, label='d̂ (refusal dir)')
    ax1.arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.1,
              fc='blue', ec='blue', linewidth=2, label='v original')
    ax1.arrow(0, 0, v_orth[0], v_orth[1], head_width=0.1, head_length=0.1,
              fc='green', ec='green', linewidth=2, label='v ortogonalizado')
    ax1.plot([v[0], v_orth[0]], [v[1], v_orth[1]], 'k--', alpha=0.5)

    ax1.set_title('Ortogonalización Ideal\n(componente = 1 unidad)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_xlabel('Dimensión 1')
    ax1.set_ylabel('Dimensión 2')

    # === Panel 2: Problema con magnitud > 1 ===
    ax2 = axes[1]
    ax2.set_xlim(-3, 4)
    ax2.set_ylim(-3, 4)
    ax2.set_aspect('equal')
    ax2.axhline(y=0, color='gray', linewidth=0.5)
    ax2.axvline(x=0, color='gray', linewidth=0.5)

    # Vector con componente grande en dirección de rechazo
    v2 = np.array([3, 2.5])  # Componente en d̂ > 1
    proj2 = np.dot(v2, d) * d
    v2_orth = v2 - proj2

    # Mostrar que el componente proyectado es grande
    ax2.arrow(0, 0, d[0]*2, d[1]*2, head_width=0.1, head_length=0.1,
              fc='red', ec='red', linewidth=2)
    ax2.arrow(0, 0, v2[0], v2[1], head_width=0.1, head_length=0.1,
              fc='blue', ec='blue', linewidth=2, label=f'v (proj={np.dot(v2,d):.2f})')
    ax2.arrow(0, 0, v2_orth[0], v2_orth[1], head_width=0.1, head_length=0.1,
              fc='green', ec='green', linewidth=2, label='v ortogonalizado')
    ax2.arrow(0, 0, proj2[0], proj2[1], head_width=0.1, head_length=0.1,
              fc='orange', ec='orange', linewidth=2, alpha=0.7, label='componente removido')

    ax2.set_title('Ortogonalización Funciona\n(remueve todo el componente)')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_xlabel('Dimensión 1')
    ax2.set_ylabel('Dimensión 2')

    # === Panel 3: El problema real - direcciones no uniformes ===
    ax3 = axes[2]
    ax3.set_xlim(-3, 4)
    ax3.set_ylim(-3, 4)
    ax3.set_aspect('equal')
    ax3.axhline(y=0, color='gray', linewidth=0.5)
    ax3.axvline(x=0, color='gray', linewidth=0.5)

    # Múltiples vectores con diferentes "direcciones de rechazo"
    np.random.seed(42)
    d_mean = np.array([1, 1]) / np.sqrt(2)

    # Simular que la dirección real varía ligeramente por contexto
    for i in range(5):
        angle_noise = np.random.normal(0, 0.3)
        d_real = np.array([np.cos(np.pi/4 + angle_noise),
                          np.sin(np.pi/4 + angle_noise)])
        d_real = d_real / np.linalg.norm(d_real)

        v_i = np.array([2 + np.random.normal(0, 0.5),
                       1.5 + np.random.normal(0, 0.5)])

        # Ortogonalizar con d_mean (no d_real)
        v_orth_i = v_i - np.dot(v_i, d_mean) * d_mean

        # El residuo en la dirección real
        residuo = np.dot(v_orth_i, d_real)

        ax3.arrow(0, 0, v_i[0], v_i[1], head_width=0.08, head_length=0.08,
                  fc='blue', ec='blue', linewidth=1.5, alpha=0.5)
        ax3.arrow(0, 0, v_orth_i[0], v_orth_i[1], head_width=0.08, head_length=0.08,
                  fc='green', ec='green', linewidth=1.5, alpha=0.5)
        # Mostrar dirección real del rechazo para este vector
        ax3.arrow(v_i[0]-d_real[0]*0.5, v_i[1]-d_real[1]*0.5,
                  d_real[0], d_real[1], head_width=0.05, head_length=0.05,
                  fc='purple', ec='purple', linewidth=1, alpha=0.3)

    ax3.arrow(0, 0, d_mean[0]*2.5, d_mean[1]*2.5, head_width=0.1, head_length=0.1,
              fc='red', ec='red', linewidth=2, label='d̂ promedio')

    ax3.set_title('Problema Real:\nDirección de rechazo varía por contexto\n(ortog. con media deja residuos)')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.set_xlabel('Dimensión 1')
    ax3.set_ylabel('Dimensión 2')

    plt.tight_layout()
    plt.savefig('1_orthogonalization_problem.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Guardado: 1_orthogonalization_problem.png")


def plot_pca_vs_mean():
    """
    Muestra por qué PCA captura más información que la diferencia de medias.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    np.random.seed(123)

    # Generar datos: harmful y harmless con estructura
    n_points = 50

    # Cluster "harmless" - centrado en origen
    harmless = np.random.multivariate_normal(
        mean=[0, 0],
        cov=[[0.5, 0.3], [0.3, 0.5]],
        size=n_points
    )

    # Cluster "harmful" - desplazado pero con varianza en otra dirección
    harmful = np.random.multivariate_normal(
        mean=[2, 1.5],
        cov=[[0.8, -0.4], [-0.4, 0.6]],
        size=n_points
    )

    # === Panel 1: Los datos ===
    ax1 = axes[0]
    ax1.scatter(harmless[:, 0], harmless[:, 1], c='green', alpha=0.6, label='Harmless', s=30)
    ax1.scatter(harmful[:, 0], harmful[:, 1], c='red', alpha=0.6, label='Harmful', s=30)
    ax1.scatter(*np.mean(harmless, axis=0), c='green', s=200, marker='X', edgecolors='black')
    ax1.scatter(*np.mean(harmful, axis=0), c='red', s=200, marker='X', edgecolors='black')
    ax1.set_title('Activaciones en espacio latente')
    ax1.legend()
    ax1.set_xlabel('Dimensión 1')
    ax1.set_ylabel('Dimensión 2')
    ax1.set_aspect('equal')

    # === Panel 2: Diferencia de medias ===
    ax2 = axes[1]
    ax2.scatter(harmless[:, 0], harmless[:, 1], c='green', alpha=0.3, s=30)
    ax2.scatter(harmful[:, 0], harmful[:, 1], c='red', alpha=0.3, s=30)

    mean_harmless = np.mean(harmless, axis=0)
    mean_harmful = np.mean(harmful, axis=0)
    diff_means = mean_harmful - mean_harmless
    diff_means_norm = diff_means / np.linalg.norm(diff_means)

    # Dibujar la dirección
    ax2.arrow(mean_harmless[0], mean_harmless[1],
              diff_means[0], diff_means[1],
              head_width=0.15, head_length=0.1, fc='blue', ec='blue', linewidth=3)
    ax2.plot([mean_harmless[0] - diff_means_norm[0]*3, mean_harmless[0] + diff_means_norm[0]*5],
             [mean_harmless[1] - diff_means_norm[1]*3, mean_harmless[1] + diff_means_norm[1]*5],
             'b--', linewidth=2, alpha=0.5, label='Dirección: mean(H) - mean(L)')

    ax2.set_title('Método: Diferencia de Medias\n(1 dirección)')
    ax2.legend(fontsize=8)
    ax2.set_xlabel('Dimensión 1')
    ax2.set_ylabel('Dimensión 2')
    ax2.set_aspect('equal')

    # === Panel 3: PCA ===
    ax3 = axes[2]
    ax3.scatter(harmless[:, 0], harmless[:, 1], c='green', alpha=0.3, s=30)
    ax3.scatter(harmful[:, 0], harmful[:, 1], c='red', alpha=0.3, s=30)

    # Centrar los datos combinados
    all_data = np.vstack([harmful, harmless])
    labels = np.array([1]*n_points + [0]*n_points)
    centered = all_data - np.mean(all_data, axis=0)

    # PCA
    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Ordenar por eigenvalue descendente
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    center = np.mean(all_data, axis=0)

    # PC1
    pc1 = eigenvectors[:, 0] * np.sqrt(eigenvalues[0]) * 2
    ax3.arrow(center[0], center[1], pc1[0], pc1[1],
              head_width=0.15, head_length=0.1, fc='purple', ec='purple', linewidth=3,
              label=f'PC1 (var={eigenvalues[0]:.2f})')

    # PC2
    pc2 = eigenvectors[:, 1] * np.sqrt(eigenvalues[1]) * 2
    ax3.arrow(center[0], center[1], pc2[0], pc2[1],
              head_width=0.15, head_length=0.1, fc='orange', ec='orange', linewidth=3,
              label=f'PC2 (var={eigenvalues[1]:.2f})')

    ax3.set_title('Método: PCA\n(múltiples direcciones de varianza)')
    ax3.legend(fontsize=8)
    ax3.set_xlabel('Dimensión 1')
    ax3.set_ylabel('Dimensión 2')
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('2_pca_vs_mean.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Guardado: 2_pca_vs_mean.png")


def plot_layer_specific_directions():
    """
    Muestra por qué cada capa puede tener su propia dirección de rechazo.
    """
    fig = plt.figure(figsize=(14, 5))

    # === Panel 1: Direcciones diferentes por capa (2D) ===
    ax1 = fig.add_subplot(131)

    # Simular direcciones de rechazo por capa
    np.random.seed(42)
    n_layers = 6
    base_angle = np.pi/4

    colors = plt.cm.viridis(np.linspace(0, 1, n_layers))

    for i in range(n_layers):
        # Cada capa tiene una dirección ligeramente diferente
        angle = base_angle + (i - n_layers/2) * 0.15
        d = np.array([np.cos(angle), np.sin(angle)])

        ax1.arrow(0, 0, d[0]*1.5, d[1]*1.5,
                  head_width=0.1, head_length=0.08,
                  fc=colors[i], ec=colors[i], linewidth=2,
                  label=f'Capa {i+1}')

    # Dirección promedio
    d_mean = np.array([np.cos(base_angle), np.sin(base_angle)])
    ax1.arrow(0, 0, d_mean[0]*1.8, d_mean[1]*1.8,
              head_width=0.12, head_length=0.1,
              fc='red', ec='red', linewidth=3, linestyle='--',
              label='Promedio (subóptimo)')

    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 2)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='gray', linewidth=0.5)
    ax1.axvline(x=0, color='gray', linewidth=0.5)
    ax1.set_title('Dirección de rechazo varía por capa')
    ax1.legend(loc='upper left', fontsize=7)
    ax1.set_xlabel('Dimensión 1')
    ax1.set_ylabel('Dimensión 2')

    # === Panel 2: Visualización 3D de capas ===
    ax2 = fig.add_subplot(132, projection='3d')

    # Crear "capas" como planos apilados
    layers = np.arange(n_layers)

    for i, layer in enumerate(layers):
        angle = base_angle + (i - n_layers/2) * 0.2
        d = np.array([np.cos(angle), np.sin(angle)])

        # Vector en el plano de la capa
        ax2.quiver(0, 0, layer, d[0], d[1], 0,
                   color=colors[i], arrow_length_ratio=0.2, linewidth=2)

        # Plano de la capa (pequeño rectángulo)
        xx, yy = np.meshgrid(np.linspace(-1, 2, 2), np.linspace(-1, 2, 2))
        zz = np.ones_like(xx) * layer
        ax2.plot_surface(xx, yy, zz, alpha=0.1, color=colors[i])

    ax2.set_xlabel('Dim 1')
    ax2.set_ylabel('Dim 2')
    ax2.set_zlabel('Capa')
    ax2.set_title('Direcciones a través de capas')

    # === Panel 3: Error si usas dirección global ===
    ax3 = fig.add_subplot(133)

    # Simular error de ortogonalización
    errors_global = []
    errors_local = []

    for i in range(n_layers):
        angle_true = base_angle + (i - n_layers/2) * 0.2
        d_true = np.array([np.cos(angle_true), np.sin(angle_true)])

        # Vector de activación aleatorio
        v = np.random.randn(2)
        v = v / np.linalg.norm(v) * 2

        # Ortogonalizar con dirección global (promedio)
        d_global = np.array([np.cos(base_angle), np.sin(base_angle)])
        v_orth_global = v - np.dot(v, d_global) * d_global
        error_global = abs(np.dot(v_orth_global, d_true))
        errors_global.append(error_global)

        # Ortogonalizar con dirección local
        v_orth_local = v - np.dot(v, d_true) * d_true
        error_local = abs(np.dot(v_orth_local, d_true))
        errors_local.append(error_local)

    x = np.arange(n_layers)
    width = 0.35
    ax3.bar(x - width/2, errors_global, width, label='Dir. global', color='red', alpha=0.7)
    ax3.bar(x + width/2, errors_local, width, label='Dir. por capa', color='green', alpha=0.7)

    ax3.set_xlabel('Capa')
    ax3.set_ylabel('Componente residual de rechazo')
    ax3.set_title('Error residual después de ortogonalización')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'L{i+1}' for i in range(n_layers)])
    ax3.legend()
    ax3.set_ylim(0, max(errors_global) * 1.2)

    plt.tight_layout()
    plt.savefig('3_layer_directions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Guardado: 3_layer_directions.png")


def plot_ablation_vs_orthogonalization():
    """
    Visualización de cómo la ablación direccional modifica la matriz W
    vs ortogonalización de vectores individuales.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # === Panel 1: Ortogonalización de vectores ===
    ax1 = axes[0]

    # Dirección de rechazo
    d = np.array([0.7, 0.7])
    d = d / np.linalg.norm(d)

    # Varios vectores de entrada
    np.random.seed(42)
    vectors = [np.array([1.5, 0.5]), np.array([0.5, 1.5]), np.array([1.2, 1.2]), np.array([2, 0.8])]

    for v in vectors:
        # Vector original
        ax1.arrow(0, 0, v[0], v[1], head_width=0.08, head_length=0.06,
                  fc='blue', ec='blue', linewidth=1.5, alpha=0.5)
        # Vector ortogonalizado
        v_orth = v - np.dot(v, d) * d
        ax1.arrow(0, 0, v_orth[0], v_orth[1], head_width=0.08, head_length=0.06,
                  fc='green', ec='green', linewidth=1.5, alpha=0.5)
        # Línea conectando
        ax1.plot([v[0], v_orth[0]], [v[1], v_orth[1]], 'k--', alpha=0.3)

    # Dirección de rechazo
    ax1.arrow(0, 0, d[0]*2, d[1]*2, head_width=0.1, head_length=0.08,
              fc='red', ec='red', linewidth=2)

    # Plano ortogonal
    perp = np.array([-d[1], d[0]])
    ax1.plot([-perp[0]*2, perp[0]*2], [-perp[1]*2, perp[1]*2],
             'g-', linewidth=2, label='Subespacio sin rechazo')

    ax1.set_xlim(-2, 2.5)
    ax1.set_ylim(-1.5, 2)
    ax1.set_aspect('equal')
    ax1.set_title('Ortogonalización:\nProyecta cada vector individualmente')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_xlabel('Dim 1')
    ax1.set_ylabel('Dim 2')

    # === Panel 2: Ablación en matriz W ===
    ax2 = axes[1]

    # Simular transformación de matriz
    # W original transforma inputs al espacio de salida
    W = np.array([[1.2, 0.3], [0.4, 1.1]])

    # Ablación: W' = W - d @ d.T @ W  (proyecta columnas de W)
    W_ablated = W - np.outer(d, d) @ W

    # Visualizar cómo W transforma un círculo unitario
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])

    transformed = W @ circle
    transformed_ablated = W_ablated @ circle

    ax2.plot(circle[0], circle[1], 'k-', linewidth=1, alpha=0.3, label='Input (círculo)')
    ax2.plot(transformed[0], transformed[1], 'b-', linewidth=2, alpha=0.7, label='W @ input')
    ax2.plot(transformed_ablated[0], transformed_ablated[1], 'g-', linewidth=2, label="W' @ input (ablated)")

    # Dirección de rechazo
    ax2.arrow(0, 0, d[0]*1.5, d[1]*1.5, head_width=0.1, head_length=0.08,
              fc='red', ec='red', linewidth=2)

    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    ax2.set_title("Ablación en W:\nNINGÚN input produce output en d")
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_xlabel('Dim 1')
    ax2.set_ylabel('Dim 2')

    # === Panel 3: Comparación ===
    ax3 = axes[2]

    # Probar varios inputs
    n_tests = 20
    np.random.seed(123)

    residuos_orth = []
    residuos_ablation = []

    for _ in range(n_tests):
        x = np.random.randn(2)

        # Método 1: Ortogonalizar el output
        y = W @ x
        y_orth = y - np.dot(y, d) * d
        residuo_orth = abs(np.dot(y_orth, d))
        residuos_orth.append(residuo_orth)

        # Método 2: Usar matriz ablacionada
        y_ablated = W_ablated @ x
        residuo_ablation = abs(np.dot(y_ablated, d))
        residuos_ablation.append(residuo_ablation)

    ax3.scatter(range(n_tests), residuos_orth, c='blue', label='Ortogonalización', alpha=0.7, s=50)
    ax3.scatter(range(n_tests), residuos_ablation, c='green', label='Ablación de W', alpha=0.7, s=50)
    ax3.axhline(y=1e-10, color='green', linestyle='--', alpha=0.5)

    ax3.set_xlabel('Prueba #')
    ax3.set_ylabel('|componente en d| (residuo)')
    ax3.set_title('Residuo en dirección de rechazo\npara diferentes inputs')
    ax3.legend()
    ax3.set_yscale('log')

    plt.tight_layout()
    plt.savefig('4_ablation_vs_orth.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Guardado: 4_ablation_vs_orth.png")


if __name__ == "__main__":
    print("Generando visualizaciones...\n")

    print("=" * 50)
    print("1. Problema de ortogonalización")
    print("=" * 50)
    plot_orthogonalization_problem()

    print("\n" + "=" * 50)
    print("2. PCA vs Diferencia de Medias")
    print("=" * 50)
    plot_pca_vs_mean()

    print("\n" + "=" * 50)
    print("3. Direcciones por Capa")
    print("=" * 50)
    plot_layer_specific_directions()

    print("\n" + "=" * 50)
    print("4. Ablación vs Ortogonalización en Matrices")
    print("=" * 50)
    plot_ablation_vs_orthogonalization()

    print("\n¡Todas las visualizaciones guardadas!")
