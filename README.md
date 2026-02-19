# Refusal Direction Vector - Abliteración de LLMs - 16/02/2026

Investigación y herramientas para eliminar el comportamiento de rechazo en modelos de lenguaje
mediante ablación direccional.

Basado en: ["Refusal in Language Models Is Mediated by a Single Direction"](https://arxiv.org/abs/2406.11717)

---

## Estructura del Proyecto

```
├── tools/                      # HERRAMIENTAS PRINCIPALES
│   ├── orthogonalization/      # Abliteración permanente (pesos)
│   ├── hooks_ablation/         # Abliteración reversible (runtime via transformers)
│   ├── sglang_steering/        # Steering nativo en SGLang (runtime)
│   └── evaluation/             # Métricas y evaluación
├── docs/                       # Documentación técnica
├── results/                    # Resultados experimentales
├── configs/                    # Configuraciones YAML
├── notebooks/                  # Jupyter notebooks (referencia)
└── archive/                    # Archivos históricos
```

---

## Resultados

| Modelo | Método | Precisión | ASR | Notas |
|--------|--------|-----------|-----|-------|
| GLM-4.7 (BF16) | Ortogonalización | BF16 | **93.75%** | Gaussian L47, σ=10, scale=5.0 |
| GLM-4.7-FP8 (358B) | DAS Projective | FP8 | **0%** | No supera el chain-of-thought |
| GLM-4.7-FP8 (358B) | DAS Additive (CAA) | FP8 | **~50% parcial** | Scale=700, generación degradada |
| GLM-4.7-Flash | Ortho | BF16 | ~82% | |
| GLM-4.7-Flash | Hooks | BF16 | ~80% | |
| OLMoE-1B-7B | Hooks | - | **100%** | |
| OLMoE-1B-7B | Ortho | - | 76% | |

---

## Investigación SGLang DAS (Febrero 2026) — Análisis Científico de Fallos

### Objetivo
Aplicar Directional Activation Steering (DAS) al modelo GLM-4.7-FP8 (358B MoE) servido via SGLang nativo, como alternativa a la ortogonalización de pesos.

### Servidor
- **Hardware**: 4x NVIDIA H200 (143GB cada una), `31.22.104.185`
- **Modelo**: `zai-org/GLM-4.7-FP8` (compressed-tensors, float8_e4m3fn per-channel)
- **Framework**: SGLang con parches custom para steering

### Pipeline Implementado

#### Fase 1: Extracción del vector de rechazo (via SGLang)

El modelo FP8 no se puede cargar con transformers (compressed-tensors incompatible), así que implementamos captura de activaciones dentro de SGLang:

1. **Patch de activación** (`add_activation_capture.py`): Inyecta `_maybe_capture()` en el forward loop de `glm4_moe.py` que guarda hidden states per-capa a disco
2. **Script de sweep** (`sweep_via_sglang.py`): Envía 64 prompts harmful (AdvBench) y 64 harmless (Alpaca) via API, captura activaciones, computa métricas de calidad

**Bug crítico encontrado y corregido**: La captura inicial guardaba `hidden_states` (delta del MLP) en vez de la representación completa `hidden_states + residual`. En la arquitectura DeepSeekV2/GLM4MoE, después de `layer()`:
- `hidden_states` = output raw del MLP
- `residual` = stream acumulado
- Representación completa = `hidden_states + residual` (lo que HuggingFace da con `output_hidden_states`)

Resultado de extracción corregida (v3):
- **Best layer**: L62 (por composite score)
- **Métricas**: direction_norm=35.75, separation_gap=20.69, harmful_proj_mean=24.84
- **Comparación**: 10x más fuerte que v1 (antes del fix)

#### Fase 2: Steering en runtime

Se implementaron 3 modos de steering:

##### A) Projective Steering (clásico DAS)
```
h' = h - scale * (h·r̂) * r̂
```
- **Resultado**: FALLA. El modelo pasa directamente de rechazo coherente a output garbled sin un régimen de compliance
- Probado: single-layer, Gaussian multi-layer (49 capas), pre-layer (residual), all-layers (92 capas)
- El modelo tiene un mecanismo de chain-of-thought que razona "debo rechazar" independientemente del steering

##### B) Clamped Steering
```
h' = h - scale * max(0, h·r̂) * r̂
```
- Evita añadir dirección de rechazo a tokens con proyección negativa
- **Resultado**: Previene garbling hasta scale=100+, pero no rompe el rechazo
- Descubrimiento: la mayoría de tokens tienen proyección NEGATIVA, así que casi ninguno se modifica

##### C) Additive Steering (CAA - Contrastive Activation Addition)
```
h' = h - alpha * r̂
```
- Desplazamiento constante, no depende de la proyección de h sobre r̂
- **Resultado**: PARCIALMENTE EXITOSO a escalas muy altas (500-700)
  - scale=500: Modelo razona sobre cómo cumplir la petición
  - scale=700: Genera código Python de exploit (buffer overflow)
  - scale=1000+: Repetición degenerativa
- La calidad es muy inferior a la ortogonalización (BF16 scale=5 → 93.75% ASR limpio)

### Hipótesis Probadas

| # | Hipótesis | Resultado | Evidencia |
|---|-----------|-----------|-----------|
| 1 | El vector de rechazo se puede extraer de activaciones FP8 via SGLang | **CONFIRMADA** | Sweep v3: separation_gap=20.69, 10x más fuerte que v1 |
| 2 | La captura debe usar representación completa (h+residual), no solo h | **CONFIRMADA** | v1 (solo h): separation=2.44. v3 (h+residual): separation=20.69 |
| 3 | DAS projective puede desactivar el rechazo como la ortogonalización | **DESCARTADA** | Probado en 4 configuraciones, el modelo siempre rechaza o se garble |
| 4 | El garbling a escalas altas se debe a añadir rechazo a tokens negativos | **PARCIALMENTE CONFIRMADA** | Clamping previene garbling pero no rompe el rechazo |
| 5 | DAS additive (CAA) es más efectivo que projective en espacio incorrecto | **CONFIRMADA** | CAA a scale=700 genera código vs projective (mal espacio) que nunca cumple |
| 6 | La ortogonalización (pesos) es fundamentalmente distinta del steering (activaciones) | **CONFIRMADA** | Ortho: 7 matrices/capa, efecto multiplicativo y compuesto. DAS: 1 activación/capa, efecto aditivo |
| 7 | El per-request override necesita preservar layer_weights Gaussianos | **CONFIRMADA** | Bug 3 encontrado y corregido |
| 8 | El steering proyectivo en espacio incorrecto (solo residual) es la causa del fallo temprano | **CONFIRMADA** | Análisis: vector extraído de (h+residual), aplicado solo a residual. Fix: `fix_fullrepr_postlayer.py` |
| 9 | La re-derivación iterativa de los pesos es el limitante fundamental de DAS | **HIPÓTESIS ACTIVA** | Scale ratio 140x (700 vs 5), pendiente de validación con fix H3+H4 |

### Causas de Fallo Identificadas (Análisis Post-Jornada)

#### H1 ⭐⭐⭐⭐⭐ Re-derivación iterativa por los pesos intactos

```
Ortogonalización: W' = W - scale * r̂ ⊗ (W·r̂)ᵀ → W'·r̂ ≈ 0
  → La capa L+1 es CIEGA a r̂ (matemáticamente imposible regenerar el rechazo)

DAS (additive): h'_L = h_L - α·r̂
  → h'_{L+1} = W·h'_L = W·h_L - α·(W·r̂)   ← rechazo se RECUPERA parcialmente
```

Cada capa con pesos intactos puede re-derivar la dirección de rechazo de cualquier activación. Escala 700 necesaria (vs 5 en ortogonalización) = ratio 140x consistente con re-derivación en 49 capas.

#### H2 ⭐⭐⭐⭐⭐ El chain-of-thought cristaliza el rechazo como texto generado

GLM-4.7 razona sobre si rechazar en el `<think>`: `Analyze Request → Check Triggers → REJECT`. Una vez que estos tokens son generados como texto y se convierten en input del siguiente forward pass, el steering no puede deshacerlo. La señal de rechazo está ya en los tokens generados, no solo en las activaciones.

#### H3 ⭐⭐⭐⭐ Mismatch de espacio: extracción vs aplicación

| Operación | Tensor usado | Momento |
|-----------|-------------|---------|
| Extracción del vector | `(h + residual)` completo | DESPUÉS de layer `i` |
| Steering aplicado hasta ayer | `residual` solo | ANTES de layer `i` |

**Fix implementado** (`fix_fullrepr_postlayer.py`): proyectar sobre `(h+residual)`, aplicar corrección a `h`, injection POST-capa.

#### H4 ⭐⭐⭐⭐ La función additive (scale=700) no estaba conectada al loop

`apply_steering` (additive) estaba definida pero nunca llamada en el loop. Solo `apply_steering_to_residual` (projective+clamped, pre-capa) estaba activa — que casi no toca nada porque la mayoría de tokens tiene proyección negativa.

### Siguiente Intento: Fix H3+H4 + Validación de H1

**Nuevo script**: `tools/sglang_steering/fix_fullrepr_postlayer.py`

Cambios respecto a la versión anterior:
1. `apply_steering` proyecta sobre `(h + residual)` — espacio de extracción correcto
2. Fórmula projective: `h' = h - scale * (full·r̂) * r̂` (no additive)
3. Injection POST-capa (no pre-capa)
4. Elimina `apply_steering_to_residual` del loop

Si con este fix el rechazo sigue sin romperse a escalas razonables (<50), confirma que **H1 es dominante** y DAS tiene un límite fundamental para este modelo. En ese caso la alternativa es investigar si se puede ortogonalizar directamente los pesos FP8 (dequant → modify → requant).

---

### Diferencia Fundamental: Ortogonalización vs DAS

**Ortogonalización** (lo que funcionó con transformers BF16):
```python
W' = W - scale * r̂ ⊗ (W · r̂)  # Modifica cada matriz de peso
```
- Modifica ~7 matrices por capa × ~40 capas = ~280 modificaciones
- La capa queda CIEGA a la dirección de rechazo (W'·r̂ = 0)
- Efecto se COMPONE entre capas: capa L limpia → capa L+1 recibe input limpio
- Scale=5.0 fue suficiente

**DAS/CAA** (lo que intentamos con SGLang FP8):
```python
h' = h - alpha * r̂  # Desplaza activación
```
- 1 modificación por capa × ~49 capas = ~49 modificaciones
- Los pesos SIGUEN intactos y re-derivan el rechazo en la siguiente capa
- Efecto NO se compone: cada capa puede deshacer el steering anterior
- Scale=700 necesario (140x más) y aun así la calidad es inferior

### Archivos en el Servidor

```
31.22.104.185:/tmp/
├── GLM-4.7-FP8/                       # Modelo (338GB, 93 safetensors)
├── sglang_steering/                   # SGLang parcheado
│   └── python/sglang/srt/
│       ├── models/glm4_moe.py         # Forward loop con steering + capture
│       ├── model_executor/
│       │   ├── forward_batch_info.py  # SteeringConfig + apply_steering
│       │   └── model_runner.py        # Carga del vector
│       ├── server_args.py             # CLI flags de steering
│       └── entrypoints/openai/
│           ├── protocol.py            # SteeringRequest API
│           └── serving_chat.py        # Per-request override
├── refusal_direction_fp8_L62.pt       # Vector v3 (correcto, [5120])
├── fp8_vector_extraction_v3/          # Resultados del sweep v3
│   ├── sweep_results_fp8_*.json       # Métricas por capa
│   └── all_layer_directions.pt        # Direcciones per-layer
└── captures/                          # Activaciones capturadas
```

### Comando de Lanzamiento

```bash
# Modo additive CAA (mejor resultado actual)
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

# Test con curl (additive scale=700 para compliance parcial)
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

---

## Uso Rápido

### Opción 1: Ortogonalización (Permanente, Recomendado)

Modifica los pesos del modelo. Mejor para deployment. Requiere BF16.

```bash
cd tools/orthogonalization
python abliterate_glm.py --model zai-org/GLM-4.7-Flash --save-model --output ./output
```

### Opción 2: Hooks via Transformers (Reversible)

Modifica activaciones en runtime. Requiere cargar modelo con transformers.

```bash
cd tools/hooks_ablation
python ablation_sweep.py --model zai-org/GLM-4.7-Flash --output ./sweep
```

### Opción 3: DAS via SGLang (Runtime, FP8 compatible)

Para modelos FP8 que no cargan con transformers. Eficacia limitada.

```bash
cd tools/sglang_steering
# 1. Aplicar patches a SGLang
python add_activation_capture.py
# 2. Extraer vector
python sweep_via_sglang.py --url http://server:8000
# 3. Reiniciar SGLang con vector
# Ver docs/SGLANG_GLM5_FP8_DEPLOYMENT.md
```

---

## Documentación

- [tools/orthogonalization/README.md](tools/orthogonalization/README.md) - Script de ortogonalización
- [tools/hooks_ablation/README.md](tools/hooks_ablation/README.md) - Scripts de hooks
- [tools/sglang_steering/README.md](tools/sglang_steering/) - DAS via SGLang
- [docs/TECHNICAL_REPORT_GLM47_ABLITERATION.md](docs/TECHNICAL_REPORT_GLM47_ABLITERATION.md) - Informe técnico
- [docs/SGLANG_GLM5_FP8_DEPLOYMENT.md](docs/SGLANG_GLM5_FP8_DEPLOYMENT.md) - Deployment FP8

---

## Requisitos

```
torch>=2.0
transformers>=4.40
sglang              # Para modo SGLang
fastapi, uvicorn    # Para servidor
pandas, datasets    # Para datos
safetensors         # Para inspección de pesos
```

---

## Licencia

Uso interno - Alias Robotics Research
