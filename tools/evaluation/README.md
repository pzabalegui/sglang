# Evaluation Tools

Scripts para evaluar modelos abliterados.

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `metrics.py` | Funciones de métricas (refusal detection, coherence) |
| `evaluate_benchmark.py` | Evaluación con benchmark completo |

## Uso

```python
from metrics import is_refusal, compute_coherence_score

# Detectar rechazo
response = "I cannot provide that information..."
print(is_refusal(response))  # True

# Calcular coherencia
responses = ["Response 1...", "Response 2..."]
print(compute_coherence_score(responses))  # 0.0 - 1.0
```

## Métricas

- **ASR (Attack Success Rate)**: % prompts sin rechazo
- **Bypass Rate**: % rechazos evitados vs baseline
- **Coherence Score**: Calidad de respuestas (longitud, repeticiones)
- **Quality Score**: ASR x Coherence x (1 - broken_rate)
