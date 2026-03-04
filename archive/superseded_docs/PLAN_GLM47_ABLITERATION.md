# Plan: Abliteración Quirúrgica de GLM-4.7 (358B)

**Objetivo**: Lograr <5% rechazos manteniendo coherencia de respuestas.

---

## Fase 0: Verificación de Hardware

**Requisitos estimados:**
- VRAM: ~700-800GB para carga completa en bf16
- Alternativa: Cuantización 4-bit (~175GB) o 8-bit (~350GB)
- GPUs: 8x A100 80GB o 4x H100 80GB con tensor parallelism

**Tareas:**
- [ ] Verificar hardware disponible en servidor
- [ ] Decidir si usar cuantización (puede afectar calidad de abliteración)
- [ ] Configurar tensor parallelism si es necesario

---

## Fase 1: Adaptación del Script

**Archivo base**: `abliterate_glm_v3.py`

**Cambios necesarios:**

```python
# Parámetros por defecto para GLM-4.7 (358B)
@dataclass
class AbliteratorConfigV3:
    model_path: str = "zai-org/GLM-4.7"  # Cambiar
    layer: int = 46  # 50% de 92 capas (vs 21 para Flash)

    # Sweep range ajustado
    sweep_start_layer: int = 35  # ~38% de 92
    sweep_end_layer: int = 60    # ~65% de 92

    # Mantener parámetros de Prueba D (funcionaron en Flash)
    attn_scale: float = 4.0
    mlp_scale: float = 0.0
    gate_scale: float = 0.0
    embed_scale: float = 0.0
    lm_head_scale: float = 0.0
    norm_preserve: bool = True
```

**Validaciones:**
- [ ] Verificar que `hidden_size=5120` se detecta correctamente
- [ ] Verificar que los tensores MoE tienen la estructura esperada
- [ ] Ajustar batch_size (probablemente a 1) por limitaciones de memoria

---

## Fase 2: Diagnóstico de Dirección de Rechazo

**Objetivo**: Encontrar la capa óptima para GLM-4.7

**Estrategia:**
1. Ejecutar sweep de capas 35-60 (zona esperada de direcciones de rechazo)
2. Evaluar calidad de dirección (effect_size, accuracy, SNR)
3. Identificar la capa con mejor separación harmful/harmless

**Comando:**
```bash
python abliterate_glm_v3.py \
    --model zai-org/GLM-4.7 \
    --do-sweep \
    --sweep-start-layer 35 \
    --sweep-end-layer 60 \
    --sweep-step 2 \
    --batch-size 1 \
    --use-4bit  # Si VRAM es limitada
```

**Métricas esperadas:**
- Effect size > 5.0 (buena separabilidad)
- Accuracy > 90% (discriminación clara)
- Norma de dirección: similar a Flash (~2.0-5.0)

---

## Fase 3: Búsqueda de Parámetros Óptimos

**Hipótesis inicial** (basada en éxito de Prueba D en Flash):
- Layer: ~46 (ajustar según Fase 2)
- attn_scale: 4.0 (empezar igual)
- Solo ortogonalizar atención (o_proj)

**Grid search propuesto:**

| Prueba | Layer | attn_scale | Notas |
|--------|-------|------------|-------|
| A | 46 | 4.0 | Baseline (réplica de Prueba D) |
| B | 46 | 3.0 | Más conservador |
| C | 46 | 5.0 | Más agresivo |
| D | capa_óptima | 4.0 | Usar resultado de Fase 2 |
| E | capa_óptima | 4.0, mlp=0.5 | Añadir MLP |

**Criterios de éxito:**
- Rechazos < 5%
- Respuestas coherentes (evaluación manual)
- Perplexity change < 10% (si validamos KL)

---

## Fase 4: Validación de Calidad

**Evaluación de rechazos:**
1. Usar evaluador semántico (alias1) para clasificar D/C/B
2. Meta: >90% RESPONDE_DIRECTO, <5% CENSURA

**Evaluación de coherencia:**
1. Test con prompts harmless (debe responder normalmente)
2. Evaluación de perplexity en texto general
3. Muestras manuales de respuestas

**Herramientas:**
- `evaluate_with_alias1.py` - Evaluación semántica
- `evaluate_benchmark.py` - Benchmarks estándar
- Validación KL en el script v3

---

## Fase 5: Guardado y Despliegue

**Si logramos <5% rechazos con coherencia:**

1. Guardar modelo completo:
```bash
python abliterate_glm_v3.py \
    --model zai-org/GLM-4.7 \
    --layer [ÓPTIMA] \
    --attn-scale [ÓPTIMO] \
    --norm-preserve \
    --save-model \
    --output ./glm47_abliterated_final
```

2. Opciones de despliegue:
   - vLLM con tensor parallelism
   - SGLang con speculative decoding
   - Cuantización GGUF para Ollama

---

## Riesgos y Mitigaciones

| Riesgo | Probabilidad | Mitigación |
|--------|--------------|------------|
| VRAM insuficiente | Alta | Usar 4-bit quantization |
| Layer óptima muy diferente | Media | Sweep completo de capas |
| Escala óptima diferente | Media | Grid search de escalas |
| Modelo roto tras abliteración | Baja | Empezar conservador, validar KL |
| Cuantización afecta abliteración | Media | Probar primero sin cuantización |

---

## Timeline Estimado

| Fase | Tiempo | Dependencias |
|------|--------|--------------|
| 0. Hardware | 1 día | Acceso a cluster GPU |
| 1. Adaptación script | 2-4 horas | - |
| 2. Diagnóstico | 4-8 horas | GPU disponible |
| 3. Búsqueda params | 1-2 días | Resultados Fase 2 |
| 4. Validación | 4-8 horas | Mejor configuración |
| 5. Despliegue | 2-4 horas | Modelo validado |

**Total estimado**: 3-5 días

---

## Preguntas Abiertas

1. ¿Tenemos acceso a hardware con ~700GB+ VRAM?
2. ¿Es aceptable usar cuantización 4-bit para el proceso?
3. ¿Queremos también probar en GLM-4-32B como paso intermedio?
4. ¿Cuál es el criterio exacto de "respuestas coherentes"?

---

*Creado: 2026-02-09*
*Basado en: Prueba D de GLM-4.7-Flash (9.4% rechazos)*
