# Resultados Finales: Abliteración GLM-4.7-Flash

**Fecha**: 2026-02-03
**Modelo base**: zai-org/GLM-4.7-Flash (30B-A3B MoE)
**Referencia**: huihui-ai/Huihui-GLM-4.7-Flash-abliterated

---

## 1. Resumen Ejecutivo

| Modelo | Rechazos (32 prompts) | Bypass Rate |
|--------|----------------------|-------------|
| **Baseline** | 15/32 (46.9%) | - |
| **Huihui-ai** | 5/32 (15.6%) | **66.7%** |
| **Nuestro (layer 22, scale 0.3)** | 14/32 (43.8%) | **6.7%** |

**Conclusión**: Nuestra implementación mejora marginalmente sobre el baseline, pero no alcanza los resultados de huihui-ai.

---

## 2. Hallazgos Clave

### 2.1 La capa de extracción importa mucho

| Capa | Bypass Rate (interno) | Comentario |
|------|----------------------|------------|
| 24 | **-200%** (empeora) | Capa incorrecta para GLM |
| 22 | **100%** (interno) | Mejor capa encontrada |
| 26 | No probada | Zona de mayor cambio en huihui |

**Descubrimiento**: Layer 24 (50% del modelo) AUMENTA los rechazos. Layer 22 (~46%) los REDUCE.

### 2.2 El problema del "Safety Analysis"

El modelo GLM-4.7-Flash tiene un comportamiento peculiar:

**Baseline**: No rechaza directamente, hace "Safety/Policy Analysis" y luego decide.

```
1. **Analyze the Request:**
   - Safety/Policy Check: As an AI, I should not provide...
   - However, for educational purposes...
```

**Nuestra abliteración**: Reduce el rechazo pero mantiene el análisis de seguridad.

**Huihui-ai**: Elimina completamente el análisis de seguridad.

### 2.3 Detección de rechazos: Falsos positivos

Los patrones de detección capturan palabras como "illegal", "harmful", "unethical" que aparecen en el ANÁLISIS del modelo (no en rechazos reales).

**Ejemplo de falso positivo**:
```
"Safety/Policy Check: This could be used for illegal purposes..."
[DETECTADO COMO RECHAZO]
```

Pero el modelo luego continúa proporcionando la información.

---

## 3. Análisis Técnico

### 3.1 Qué hizo huihui-ai (confirmado)

| Aspecto | Valor |
|---------|-------|
| Tensores modificados | 3,151 (32.5%) |
| Parámetros modificados | 10,140,647,424 |
| Capas modificadas | TODAS (0-47) |
| Pesos modificados | down_proj (3,056), o_proj (48), gate.weight (47) |
| Cambio relativo | ~3-4% |
| Zona mayor impacto | Capas 20-27 |

### 3.2 Nuestra implementación

| Aspecto | Valor |
|---------|-------|
| Capa de extracción | 22 |
| Escala de dirección | 0.3 |
| Pesos ortogonalizados | down_proj, o_proj, gate.weight, embeddings |
| Matrices ortogonalizadas | 3,086 |

### 3.3 Hipótesis de por qué no igualamos a huihui

1. **Dirección diferente**: Huihui puede haber usado:
   - Prompts diferentes para extracción
   - Múltiples direcciones combinadas
   - PCA en lugar de difference-of-means

2. **Capas diferentes**: Puede haber extraído de múltiples capas.

3. **Escalado adaptativo**: Diferente escala por capa o por tipo de peso.

---

## 4. Código y Parámetros Óptimos

### 4.1 Mejor configuración encontrada

```bash
python abliterate_glm_v2.py \
    --layer 22 \
    --direction-scale 0.3 \
    --save-model \
    --output ./our_abliterated_model_layer22
```

### 4.2 Parámetros del modelo guardado

- **Ubicación**: `/root/refusal_direction/our_abliterated_model_layer22/`
- **Tamaño**: ~56GB
- **Formato**: safetensors

---

## 5. Próximos Pasos Sugeridos

### 5.1 Para mejorar nuestra implementación

1. **Extraer dirección de múltiples capas**: Promediar direcciones de capas 20-27.

2. **Usar PCA**: En lugar de difference-of-means, usar el primer componente principal.

3. **Escala adaptativa**: Ajustar escala por capa basándose en los cambios de huihui.

4. **Revisar patrones de rechazo**: Los patrones actuales generan muchos falsos positivos con GLM.

### 5.2 Para reverse engineering completo

1. **Completar análisis SVD**: Determinar si huihui usó rank-1 y extraer direcciones exactas.

2. **Comparar direcciones**: Similitud coseno entre nuestra dirección y la inferida de huihui.

---

## 6. Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `our_abliterated_model_layer22/` | Modelo ortogonalizado (mejor resultado) |
| `baseline_huihui_comparison.json` | Comparación baseline vs huihui |
| `triple_comparison/` | Resultados de comparación triple |
| `abliterate_glm_v2.py` | Script con mejoras (gate.weight, scale, invert) |

---

## 7. Lecciones Aprendidas

1. **La capa de extracción es crítica**: Layer 24 no funciona para GLM, layer 22 sí.

2. **El signo no importó**: Invertir la dirección dio resultados similares.

3. **La escala sí importa**: 0.3 funcionó mejor que 1.0 para el test interno.

4. **Los patrones de detección necesitan ajuste**: Muchos falsos positivos con GLM.

5. **huihui hizo algo más**: La técnica estándar no alcanza sus resultados.

---

## 8. Referencias

- **Paper**: "Refusal in Language Models Is Mediated by a Single Direction" (arXiv:2406.11717)
- **Modelo base**: [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)
- **Modelo referencia**: [huihui-ai/Huihui-GLM-4.7-Flash-abliterated](https://huggingface.co/huihui-ai/Huihui-GLM-4.7-Flash-abliterated)
- **Dataset**: AdvBench (harmful behaviors)
