# Orthogonalization - Abliteración Permanente de Pesos

Script que modifica **permanentemente** los pesos del modelo para eliminar el rechazo.

## Uso

```bash
# Básico (GLM-4.7-Flash)
python abliterate_glm.py --model zai-org/GLM-4.7-Flash --layer 24 --output ./output

# Con sweep de capas
python abliterate_glm.py --model zai-org/GLM-4.7-Flash --sweep --output ./output

# Guardar modelo modificado
python abliterate_glm.py --model zai-org/GLM-4.7-Flash --save-model --output ./model_abliterated
```

## Parámetros Clave

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--model` | GLM-4.7-Flash | Modelo HuggingFace |
| `--layer` | 24 | Capa para extracción de dirección |
| `--n-train` | 64 | Muestras para extracción |
| `--n-test` | 32 | Muestras para evaluación |
| `--sweep` | False | Buscar capa óptima automáticamente |
| `--use-4bit` | False | Cuantización 4-bit (solo ablación, no ortho) |
| `--save-model` | False | Guardar modelo ortogonalizado |

## Resultados Experimentales

| Modelo | Capa | ASR Ortho |
|--------|------|-----------|
| GLM-4.7-Flash | 24 | ~82% |
| GLM-4.7 | 47 | 93.75% |
| OLMoE-1B-7B | 12 | 76% |

## Salida

```
output/
├── result_TIMESTAMP.json  # Métricas y ejemplos
└── model_abliterated/     # Modelo modificado (si --save-model)
```

## Requisitos

- torch>=2.0
- transformers>=4.40
- 48GB+ VRAM para full precision (recomendado)
- 24GB VRAM con 4-bit (solo hooks, no ortho)
