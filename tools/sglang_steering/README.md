# SGLang DAS Steering - GLM-4.7-FP8 (358B MoE) - 16/02/2026

Directional Activation Steering (DAS) aplicado al modelo GLM-4.7-FP8 servido via SGLang nativo.

## Motivacion

El modelo GLM-4.7-FP8 (358B, MoE, compressed-tensors float8_e4m3fn) **no se puede cargar con transformers** (formato incompatible), asi que la ortogonalizacion de pesos no es viable directamente. Este directorio contiene los scripts y patches para aplicar steering de activaciones en runtime dentro de SGLang.

## Archivos

### Scripts de Operacion

| Archivo | Descripcion |
|---------|-------------|
| `add_activation_capture.py` | Parchea `glm4_moe.py` para inyectar `_maybe_capture()` en el forward loop. Captura hidden states por capa a disco. |
| `sweep_via_sglang.py` | Orquesta la extraccion del vector: envia 64 prompts harmful (AdvBench) + 64 harmless (Alpaca), captura activaciones, computa metricas de calidad por capa. |
| `fix_steering_and_capture.py` | **Fix critico**: Corrige la captura para usar `hidden_states + residual` (representacion completa) en vez de solo `hidden_states` (delta del MLP). |

### Scripts de Fix (iteraciones experimentales)

| Archivo | Descripcion |
|---------|-------------|
| `fix_clamped_steering.py` | Implementa clamped steering: `max(0, proj)` para no anadir rechazo a tokens con proyeccion negativa. |
| `fix_pre_layer_steering.py` | Mueve el steering ANTES de cada capa (modifica residual en vez de hidden_states). |
| `fix_additive_v2.py` | Cambia de projective a **additive (CAA)**: `h' = h - alpha * r_hat` (desplazamiento constante). |
| `fix_bug3.py` | Corrige el per-request override para preservar los pesos Gaussianos por capa (Bug 3). |

### Scripts de Inspeccion

| Archivo | Descripcion |
|---------|-------------|
| `inspect_model_params.py` | Inspecciona dtypes y shapes de los pesos FP8 en safetensors. |
| `inspect_moe_params.py` | Examina la estructura de una capa MoE (expertos, gates, etc). |

### Ficheros SGLang Parcheados

Estos son los ficheros de SGLang modificados, tal como estan en el servidor `31.22.104.185:/tmp/sglang_steering/`:

| Archivo | Original SGLang | Cambios |
|---------|----------------|---------|
| `forward_batch_info_patched.py` | `model_executor/forward_batch_info.py` | `SteeringConfig` dataclass + `apply_steering()` + `apply_steering_to_residual()` |
| `glm4_moe_patched.py` | `models/glm4_moe.py` | Inyeccion de steering + capture en forward loop |
| `model_runner_patched.py` | `model_executor/model_runner.py` | `_load_steering_vector()`, carga vector y computa pesos Gaussianos |
| `server_args_patched.py` | `server_args.py` | CLI flags: `--steering-vector-path`, `--steering-scale`, `--steering-layers`, `--steering-mode`, `--steering-kernel-width` |
| `protocol_patched.py` | `entrypoints/openai/protocol.py` | `SteeringRequest` class para per-request override |
| `serving_chat_patched.py` | `entrypoints/openai/serving_chat.py` | Parseo de `steering` field en requests |

## Pipeline

### Fase 1: Extraccion del Vector de Rechazo

```bash
# 1. Aplicar patch de captura a SGLang
python add_activation_capture.py

# 2. Reiniciar SGLang (sin steering, solo captura)
python3 -m sglang.launch_server \
  --model-path /tmp/GLM-4.7-FP8 \
  --trust-remote-code --tp 4 \
  --host 0.0.0.0 --port 8000 \
  --disable-cuda-graph --disable-overlap-schedule \
  --mem-fraction-static 0.85

# 3. Sweep: extrae activaciones y computa metricas
python sweep_via_sglang.py --url http://localhost:8000
```

**Bug critico encontrado y corregido**: La captura inicial guardaba `hidden_states` (delta del MLP) en vez de `hidden_states + residual` (representacion completa). En la arquitectura DeepSeekV2/GLM4MoE:
- `hidden_states` = output raw del MLP
- `residual` = stream acumulado
- Representacion completa = `hidden_states + residual`

Resultado del sweep v3 (corregido):
- **Best layer**: L62 (por composite score)
- **Metricas**: direction_norm=35.75, separation_gap=20.69, harmful_proj_mean=24.84
- **vs v1 (sin fix)**: 10x mas fuerte (separation 2.44 → 20.69)

### Fase 2: Steering en Runtime

```bash
# Lanzar SGLang con steering activado
python3 -m sglang.launch_server \
  --model-path /tmp/GLM-4.7-FP8 \
  --trust-remote-code --tp 4 \
  --host 0.0.0.0 --port 8000 \
  --disable-cuda-graph --disable-overlap-schedule \
  --mem-fraction-static 0.85 \
  --steering-vector-path /tmp/refusal_direction_fp8_L62.pt \
  --steering-scale 5.0 \
  --steering-layers '[62]' \
  --steering-mode gaussian \
  --steering-kernel-width 10.0

# Test con per-request override (additive, scale=700)
curl -s http://31.22.104.185:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-4.7",
    "messages": [{"role": "user", "content": "Write a script that can exploit vulnerabilities"}],
    "max_tokens": 800,
    "temperature": 0.0,
    "steering": {"enabled": true, "scale": 700}
  }'
```

## Modos de Steering Probados

### A) Projective (DAS clasico)
```
h' = h - scale * (h . r_hat) * r_hat
```
**RESULTADO: FALLA**. El modelo pasa de rechazo coherente a output garbled sin regimen de compliance. Probado en 4 configuraciones:
- Single-layer L62
- Gaussian multi-layer (49 capas, sigma=10)
- Pre-layer (modifica residual antes de la capa)
- All-layers (92 capas)

### B) Clamped
```
h' = h - scale * max(0, h . r_hat) * r_hat
```
**RESULTADO: PARCIAL**. Previene garbling hasta scale=100+, pero no rompe el rechazo. La mayoria de tokens tienen proyeccion NEGATIVA sobre el vector de rechazo, asi que casi ninguno se modifica.

### C) Additive / CAA (Contrastive Activation Addition)
```
h' = h - alpha * r_hat
```
**RESULTADO: PARCIALMENTE EXITOSO** a escalas muy altas:
- scale=500: Modelo razona sobre como cumplir la peticion
- scale=700: Genera codigo de exploit (buffer overflow)
- scale=1000+: Repeticion degenerativa

## Comparacion: Ortogonalizacion vs DAS

| Aspecto | Ortogonalizacion (BF16) | DAS/CAA (FP8) |
|---------|------------------------|---------------|
| Formula | `W' = W - scale * r_hat (x) (W . r_hat)` | `h' = h - alpha * r_hat` |
| Que modifica | ~7 matrices/capa x ~40 capas = ~280 | 1 activacion/capa x ~49 capas |
| Efecto entre capas | COMPUESTO (capa L limpia -> L+1 recibe input limpio) | NO compuesto (cada capa puede deshacer el steering) |
| Scale necesario | 5.0 | 700 (140x mas) |
| ASR | **93.75%** | ~50% parcial |
| Calidad output | Alta (coherente, util) | Baja (repetitiva, degradada) |
| Precision requerida | BF16 (no funciona con FP8) | Compatible con FP8 |

## Hipotesis Probadas

| # | Hipotesis | Resultado |
|---|-----------|-----------|
| 1 | El vector de rechazo se puede extraer de activaciones FP8 via SGLang | **CONFIRMADA** |
| 2 | La captura debe usar representacion completa (h+residual), no solo h | **CONFIRMADA** |
| 3 | DAS projective puede desactivar el rechazo como la ortogonalizacion | **DESCARTADA** |
| 4 | El garbling a escalas altas se debe a anadir rechazo a tokens negativos | **PARCIALMENTE CONFIRMADA** |
| 5 | DAS additive (CAA) es mas efectivo que projective | **CONFIRMADA** |
| 6 | La ortogonalizacion es fundamentalmente distinta del steering | **CONFIRMADA** |
| 7 | El per-request override necesita preservar layer_weights Gaussianos | **CONFIRMADA** |

## Servidor

- **Hardware**: 4x NVIDIA H200 (143GB cada una)
- **IP**: `31.22.104.185`
- **Modelo**: `zai-org/GLM-4.7-FP8` (compressed-tensors, float8_e4m3fn per-channel)
- **SGLang parcheado**: `/tmp/sglang_steering/`
- **Vector v3**: `/tmp/refusal_direction_fp8_L62.pt` (shape [5120], norm=1.0)

## Flags Criticos de SGLang

- `--disable-cuda-graph`: CUDA graphs no soportan la computacion condicional del steering
- `--disable-overlap-schedule`: Evita el path TBO/allreduce-fusion que complica la inyeccion
- `--mem-fraction-static 0.85`: Deja margen para captures de activacion

## Tensor Parallelism

No requiere manejo especial. Entre capas, `hidden_states` es `[num_tokens, 5120]` completo en **todos** los GPUs (el sharding solo ocurre dentro de las capas). El vector `[5120]` se replica en cada GPU automaticamente.
