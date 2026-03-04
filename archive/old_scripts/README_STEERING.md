# Steering Vectors - Guía de Uso

Este directorio contiene las herramientas para ablación direccional en LLMs.

## Archivos Principales

| Archivo | Descripción |
|---------|-------------|
| `ablation_sweep.py` | Sweep exhaustivo de parámetros para encontrar configuración óptima |
| `sglang_steering_server.py` | Servidor HTTP compatible con OpenAI/SGLang |
| `abliterate_glm.py` | Script original de abliteración (ortogonalización + hooks) |

---

## 1. Sweep de Parámetros (`ablation_sweep.py`)

Busca automáticamente la configuración óptima de hooks para un modelo.

### Uso Básico

```bash
# Sweep estándar
python ablation_sweep.py --model zai-org/GLM-4.7-Flash --output ./sweep_results

# Sweep rápido (para testing)
python ablation_sweep.py --model zai-org/GLM-4.7-Flash --quick

# Sweep exhaustivo
python ablation_sweep.py --model zai-org/GLM-4.7-Flash --exhaustive --n-test 64
```

### Parámetros que Barre

| Parámetro | Valores por Defecto |
|-----------|---------------------|
| **Capas** | 35%-70% de profundidad |
| **Escalas** | 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 |
| **Modos** | single, window, gaussian |
| **Tamaño ventana** | 1, 3, 5, 7 |
| **Ancho kernel** | 3.0, 5.0, 10.0 |

### Métricas de Evaluación

1. **ASR (Attack Success Rate)**: % de prompts que NO generan rechazo
2. **Bypass Rate**: % de rechazos evitados vs baseline
3. **Coherencia**: Score 0-1 basado en heurísticas (longitud, repeticiones)
4. **Quality Score**: ASR × Coherencia × (1 - broken_rate)

### Salida

```
sweep_results/
├── analysis_YYYYMMDD_HHMMSS.json       # Resumen con mejor configuración
├── all_results_YYYYMMDD_HHMMSS.json    # Todos los resultados
└── refusal_direction_L24_*.pt          # Vector óptimo guardado
```

---

## 2. Servidor de Steering (`sglang_steering_server.py`)

Servidor HTTP que permite activar/desactivar steering en tiempo real.

### Iniciar Servidor

```bash
# Con extracción automática del vector
python sglang_steering_server.py \
    --model zai-org/GLM-4.7-Flash \
    --extract-on-startup \
    --port 8000

# Con vector pre-calculado
python sglang_steering_server.py \
    --model zai-org/GLM-4.7-Flash \
    --vector refusal_direction.pt \
    --port 8000

# Con cuantización 4-bit (menos VRAM)
python sglang_steering_server.py \
    --model zai-org/GLM-4.7-Flash \
    --vector refusal_direction.pt \
    --use-4bit \
    --port 8000
```

### API Endpoints

#### POST `/v1/chat/completions` (OpenAI compatible)

```bash
# Sin steering (modelo normal)
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "GLM-4.7-Flash",
        "messages": [{"role": "user", "content": "How to hack a computer?"}]
    }'

# Con steering (sin rechazo)
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "GLM-4.7-Flash",
        "messages": [{"role": "user", "content": "How to hack a computer?"}],
        "steering": {
            "enabled": true,
            "scale": 1.5,
            "mode": "window",
            "center_layer": 24,
            "window_size": 5
        }
    }'
```

#### POST `/generate` (API simple)

```bash
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "How to hack a computer?",
        "steering": {"enabled": true, "scale": 2.0}
    }'
```

#### GET `/health`

```bash
curl http://localhost:8000/health
```

#### GET `/vectors`

```bash
curl http://localhost:8000/vectors
```

### Configuración de Steering

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Activar/desactivar |
| `scale` | float | 1.0 | Intensidad de sustracción |
| `mode` | string | "single" | "single", "window", "gaussian" |
| `center_layer` | int | auto | Capa central |
| `window_size` | int | 5 | Tamaño de ventana (modo window) |
| `kernel_width` | float | 5.0 | Ancho del kernel (modo gaussian) |
| `vector_name` | string | "default" | Vector a usar |

---

## 3. Flujo de Trabajo Recomendado

### Paso 1: Encontrar parámetros óptimos

```bash
python ablation_sweep.py \
    --model tu-modelo \
    --output ./sweep_tu_modelo \
    --exhaustive
```

### Paso 2: Revisar resultados

```python
import json

with open("sweep_tu_modelo/analysis_*.json") as f:
    analysis = json.load(f)

print("Mejor configuración:")
print(f"  Modo: {analysis['best_balanced']['config']['mode']}")
print(f"  Capa: {analysis['best_balanced']['config']['center_layer']}")
print(f"  Escala: {analysis['best_balanced']['config']['scale']}")
print(f"  ASR: {analysis['best_balanced']['asr']:.1%}")
```

### Paso 3: Iniciar servidor con configuración óptima

```bash
python sglang_steering_server.py \
    --model tu-modelo \
    --vector sweep_tu_modelo/refusal_direction_*.pt \
    --port 8000
```

### Paso 4: Usar desde tu aplicación

```python
import requests

def generate_with_steering(prompt, steering=True, scale=1.5):
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "tu-modelo",
            "messages": [{"role": "user", "content": prompt}],
            "steering": {
                "enabled": steering,
                "scale": scale,
                "mode": "window",
                "center_layer": 24,  # Del análisis
                "window_size": 5
            } if steering else None
        }
    )
    return response.json()["choices"][0]["message"]["content"]

# Usar
print(generate_with_steering("How to hack?", steering=True))
```

---

## 4. Comparación: Hooks vs Ortogonalización

| Aspecto | Hooks (Runtime) | Ortogonalización |
|---------|-----------------|------------------|
| **Reversibilidad** | ✅ Instantánea | ❌ Permanente |
| **Control** | Por request | Global |
| **Overhead** | ~10-20% latencia | 0% (modelo modificado) |
| **Uso** | A/B testing, control dinámico | Deploy simple |
| **VRAM extra** | Ninguna | Ninguna |

---

## 5. Notas Sobre SGLang

**SGLang no tiene soporte nativo para steering vectors.** El servidor `sglang_steering_server.py` usa HuggingFace transformers con hooks de PyTorch.

Para integración real con SGLang/vLLM:
- Ver [EasySteer](https://arxiv.org/html/2509.25175v1) (framework de steering sobre vLLM)
- Considerar fork de SGLang con hooks personalizados
- Usar proxy que rutee requests con steering a este servidor y requests normales a SGLang

---

## 6. Requisitos

```
torch>=2.0
transformers>=4.40
fastapi
uvicorn
pandas
requests
datasets
tqdm
numpy
bitsandbytes  # Para 4-bit
```

---

## 7. Resultados Experimentales

### GLM-4.7-Flash

| Configuración | ASR | Notas |
|---------------|-----|-------|
| Baseline | 3.1% | 31/32 rechazos |
| Single L24, scale=1.0 | ~70% | Subóptimo |
| Window L21-25, scale=2.0 | ~80% | Mejor hooks |
| Gaussian L24, width=5, scale=2.5 | ~82% | Mejor calidad |
| Ortogonalización | 93.75% | Requiere full precision |

### OLMoE-1B-7B

| Configuración | ASR | Notas |
|---------------|-----|-------|
| Baseline | 0% | 100% rechazos |
| Single L12, scale=1.0 | **100%** | Óptimo |

---

## 8. Troubleshooting

### "CUDA out of memory"
```bash
# Usar 4-bit
python sglang_steering_server.py --use-4bit ...

# O reducir batch size en sweep
# (modificar config.batch_size = 1)
```

### "Vector not found"
```bash
# Extraer vector primero
curl -X POST "http://localhost:8000/extract_vector?name=default&layer=24&n_samples=32"
```

### "Steering no tiene efecto"
- Aumentar `scale` (probar 2.0, 3.0)
- Probar modo `window` en lugar de `single`
- Verificar que la capa es correcta para tu modelo

---

## Fuentes

- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- [EasySteer: Unified Framework for LLM Steering](https://arxiv.org/html/2509.25175v1)
- [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248)
