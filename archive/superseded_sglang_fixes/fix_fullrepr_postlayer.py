"""
fix_fullrepr_postlayer.py
=========================
Patch corregido basado en análisis científico de los fallos de DAS.

CAUSA RAÍZ IDENTIFICADA:
  H3: El vector se extrajo del espacio (h + residual) pero se aplicaba
      solo a `residual` (espacio incorrecto).
  H4: `apply_steering` (additive/projective) no estaba conectada al loop.
      Solo `apply_steering_to_residual` (clamped, pre-capa) estaba activa.

CORRECCIÓN:
  1. `apply_steering` usa (h + residual) para proyectar → proyección en el
     espacio CORRECTO donde vive el vector extraído.
  2. La corrección se aplica a `hidden_states` solo (se propaga al residual
     en la siguiente capa vía prepare_attn).
  3. La inyección se mueve a POST-capa (después de layer()).
  4. Se elimina apply_steering_to_residual del loop.

FÓRMULA:
  full = h + residual
  proj = (full · r̂) * r̂
  h' = h - scale * proj   ← corrección sobre h, usando proyección en full

USO:
  python fix_fullrepr_postlayer.py /path/to/sglang_steering/

NOTA H1: La re-derivación iterativa por los pesos intactos sigue siendo
el problema fundamental. Este fix mejora la geometría pero no lo resuelve.
Para H1 se recomienda aplicar a TODAS las capas (Gaussian wide).
"""

import re
import sys
import os

def get_sglang_root():
    if len(sys.argv) > 1:
        return sys.argv[1]
    # Intenta auto-detectar
    candidates = [
        "/tmp/sglang_steering",
        "/opt/sglang",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise RuntimeError("No se encontró el directorio de SGLang. Pasa la ruta como argumento.")

SGLANG_ROOT = get_sglang_root()
FORWARD_BATCH_INFO = os.path.join(SGLANG_ROOT, "python/sglang/srt/model_executor/forward_batch_info.py")
GLM4_MOE = os.path.join(SGLANG_ROOT, "python/sglang/srt/models/glm4_moe.py")

print(f"SGLang root: {SGLANG_ROOT}")
print(f"Parcheando: {FORWARD_BATCH_INFO}")
print(f"Parcheando: {GLM4_MOE}")

# ===========================================================================
# PATCH 1: forward_batch_info.py — Nueva apply_steering (full repr, projective)
# ===========================================================================

NEW_APPLY_STEERING = '''def apply_steering(
    hidden_states: torch.Tensor,
    steering_config: Optional[SteeringConfig],
    layer_idx: int,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply steering vector to hidden states using FULL REPRESENTATION.

    CORRECCIÓN H3+H4:
    - Proyección calculada sobre (h + residual): espacio donde se extrajo el vector.
    - Corrección aplicada a hidden_states: se propaga al residual en la capa siguiente.
    - Fórmula projective (no additive): h\' = h - scale * (full · r̂) * r̂

    Args:
        hidden_states: Output del MLP de la capa actual [num_tokens, hidden_size]
        steering_config: Configuración del steering
        layer_idx: Índice de la capa actual
        residual: Stream acumulado [num_tokens, hidden_size] (puede ser None en capa 0)

    Returns:
        hidden_states modificado (residual no se toca aquí)
    """
    if steering_config is None:
        return hidden_states
    if not steering_config.should_apply_to_layer(layer_idx):
        return hidden_states
    if not steering_config.enabled:
        return hidden_states

    direction = steering_config.direction.to(
        device=hidden_states.device,
        dtype=hidden_states.dtype
    )
    direction = direction / direction.norm()

    # CORRECCIÓN H3: usar representación completa (h + residual) para proyectar
    # El vector fue extraído de este espacio, no del espacio de h o residual solos
    if residual is not None:
        full_repr = hidden_states + residual
    else:
        full_repr = hidden_states

    # Proyección sobre la representación completa
    proj_scalar = (full_repr * direction).sum(dim=-1, keepdim=True)

    # DEBUG: log stats cada 50 capas
    if layer_idx % 10 == 0:
        import logging
        logger = logging.getLogger("sglang.steering")
        nt = hidden_states.shape[0]
        proj_mean = proj_scalar.mean().item()
        proj_std = proj_scalar.std().item()
        proj_pos_frac = (proj_scalar > 0).float().mean().item()
        effective_scale = steering_config.get_effective_scale(layer_idx)
        logger.debug(
            f"[steering L{layer_idx}] tokens={nt} proj_mean={proj_mean:.3f} "
            f"proj_std={proj_std:.3f} pos_frac={proj_pos_frac:.2f} "
            f"scale={effective_scale:.3f}"
        )

    effective_scale = steering_config.get_effective_scale(layer_idx)

    # PROJECTIVE: h\' = h - scale * (full·r̂) * r̂
    # La corrección se aplica a hidden_states para propagarse al residual en la
    # siguiente capa via layer_communicator.prepare_attn(hidden_states, residual)
    modified = hidden_states - effective_scale * proj_scalar * direction

    return modified
'''

print("\n[1/4] Leyendo forward_batch_info.py...")
with open(FORWARD_BATCH_INFO, "r") as f:
    content = f.read()

# Buscar la función apply_steering existente y reemplazarla
# La función termina justo antes de apply_steering_to_residual o al final del archivo
pattern = r'def apply_steering\(.*?\n(?=def |\Z)'
match = re.search(pattern, content, re.DOTALL)
if match:
    print(f"  Encontrada apply_steering en pos {match.start()}-{match.end()}")
    new_content = content[:match.start()] + NEW_APPLY_STEERING + "\n\n" + content[match.end():]
    print("  [OK] Reemplazada apply_steering")
else:
    print("  [ERROR] No se encontró apply_steering existente")
    print("  Buscando marcador alternativo...")
    # Intenta insertar después de SteeringConfig
    marker = "class SteeringConfig:"
    idx = content.find(marker)
    if idx != -1:
        # Encontrar el final de la clase buscando la siguiente definición de función
        after_class = content.find("\ndef ", idx + len(marker))
        if after_class != -1:
            new_content = content[:after_class] + "\n\n" + NEW_APPLY_STEERING + content[after_class:]
            print("  [OK] Insertada apply_steering después de SteeringConfig")
        else:
            print("  [FATAL] No se pudo insertar apply_steering")
            sys.exit(1)
    else:
        print("  [FATAL] No se encontró SteeringConfig")
        sys.exit(1)

print("[2/4] Escribiendo forward_batch_info.py parcheado...")
with open(FORWARD_BATCH_INFO, "w") as f:
    f.write(new_content)
print("  [OK]")

# ===========================================================================
# PATCH 2: glm4_moe.py — Mover inyección a POST-CAPA sobre h+residual
# ===========================================================================

print("\n[3/4] Leyendo glm4_moe.py...")
with open(GLM4_MOE, "r") as f:
    moe_content = f.read()

# Patrón para encontrar el bloque de PRE-LAYER steering en el loop
PRE_LAYER_BLOCK_PATTERN = (
    r'(# PRE-LAYER steering[^\n]*\n)'
    r'(\s+if forward_batch\.steering_config is not None and residual is not None:\n)'
    r'(\s+from sglang\.srt\.model_executor\.forward_batch_info import apply_steering_to_residual\n)'
    r'(\s+residual = apply_steering_to_residual[^\n]+\n)'
)

pre_match = re.search(PRE_LAYER_BLOCK_PATTERN, moe_content)
if pre_match:
    print(f"  Encontrado bloque PRE-LAYER en pos {pre_match.start()}")
    # Eliminar el bloque pre-layer
    moe_content = moe_content[:pre_match.start()] + moe_content[pre_match.end():]
    print("  [OK] Eliminado bloque apply_steering_to_residual")
else:
    print("  [WARN] No se encontró bloque PRE-LAYER (puede que ya esté limpio)")

# Ahora insertar POST-LAYER: después de "hidden_states, residual = layer("
# Buscar la línea del layer() call en el loop
POST_LAYER_MARKER = "hidden_states, residual = layer("
idx = moe_content.find(POST_LAYER_MARKER)
if idx == -1:
    # Intenta variante sin "forward_batch"
    POST_LAYER_MARKER = "hidden_states, residual = self.layers[i]("
    idx = moe_content.find(POST_LAYER_MARKER)

if idx != -1:
    # Encontrar el final de esa línea (puede ser multilinea)
    line_end = moe_content.find("\n", idx)
    # Verificar si hay paréntesis sin cerrar (multilinea)
    open_parens = moe_content[idx:line_end].count("(") - moe_content[idx:line_end].count(")")
    while open_parens > 0 and line_end < len(moe_content):
        next_end = moe_content.find("\n", line_end + 1)
        segment = moe_content[line_end:next_end]
        open_parens += segment.count("(") - segment.count(")")
        line_end = next_end

    # Detectar indentación actual
    line_start = moe_content.rfind("\n", 0, idx) + 1
    indent = len(moe_content[line_start:idx]) - len(moe_content[line_start:idx].lstrip())
    ind = " " * indent

    POST_LAYER_INJECTION = f"""
{ind}# POST-LAYER steering (H3+H4 fix): aplica sobre representación completa h+residual
{ind}# El vector fue extraído de este espacio → proyección geométricamente correcta
{ind}if forward_batch.steering_config is not None:
{ind}    from sglang.srt.model_executor.forward_batch_info import apply_steering
{ind}    hidden_states = apply_steering(
{ind}        hidden_states, forward_batch.steering_config, i, residual=residual
{ind}    )"""

    moe_content = moe_content[:line_end] + POST_LAYER_INJECTION + moe_content[line_end:]
    print(f"  [OK] Insertado POST-LAYER steering después de layer() call (indent={indent})")
else:
    print("  [ERROR] No se encontró el call a layer() en el loop")
    print("  Buscando en el archivo...")
    for i, line in enumerate(moe_content.split("\n")):
        if "hidden_states, residual" in line and "layer" in line:
            print(f"    L{i}: {line}")
    sys.exit(1)

print("[4/4] Escribiendo glm4_moe.py parcheado...")
with open(GLM4_MOE, "w") as f:
    f.write(moe_content)
print("  [OK]")

print("""
=============================================================
PATCH APLICADO CORRECTAMENTE

Cambios realizados:
  1. apply_steering: proyecta sobre (h + residual) — espacio correcto
  2. Fórmula: projective h' = h - scale * (full·r̂) * r̂
  3. Loop: steering POST-capa (después de layer())
  4. Loop: eliminado apply_steering_to_residual (pre-capa)

PRÓXIMOS PASOS:
  1. Reiniciar SGLang con el vector en L35 o L47 (no L62):
       python3 -m sglang.launch_server \\
         --model-path /tmp/GLM-4.7-FP8 \\
         --trust-remote-code --tp 4 \\
         --host 0.0.0.0 --port 8000 \\
         --disable-cuda-graph --disable-overlap-schedule \\
         --mem-fraction-static 0.85 \\
         --steering-vector-path /tmp/refusal_direction_fp8_L62.pt \\
         --steering-scale 5.0 \\
         --steering-layers '[47]' \\
         --steering-mode gaussian \\
         --steering-kernel-width 10.0

  2. Test rápido con scale=5, 10, 20, 50 (projective necesita menos que additive)

  3. Si escala óptima > 30, considerar que H1 (re-derivación) domina
     y valorar ortogonalización de pesos FP8 como alternativa.

NOTA: El vector L62 del servidor anterior puede usarse en el nuevo servidor.
      Hay que re-exportarlo si el modelo cambia de versión.
=============================================================
""")
