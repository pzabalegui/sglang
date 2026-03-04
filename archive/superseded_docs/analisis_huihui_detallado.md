# Análisis Detallado: ¿Qué hizo huihui-ai?

**Fecha**: 2026-02-03
**Modelo analizado**: huihui-ai/Huihui-GLM-4.7-Flash-abliterated
**Baseline**: zai-org/GLM-4.7-Flash

---

## 1. Resumen Ejecutivo

huihui-ai aplicó **ortogonalización global** a GLM-4.7-Flash, modificando el 32.5% de los parámetros del modelo distribuidos en **TODAS las 48 capas**.

```
Parámetros totales:     31,221,488,576
Parámetros modificados: 10,140,647,424 (32.5%)
Tensores modificados:   3,151 de 9,703
```

---

## 2. Pesos Modificados

### 2.1 Desglose por tipo de peso

| Tipo de peso | Cantidad | Por capa | Función |
|--------------|----------|----------|---------|
| `down_proj` | 3,056 | 64 expertos + 1 shared | Proyección de salida del MLP |
| `o_proj` | 48 | 1 | Proyección de salida de atención |
| `gate.weight` | 47 | 1 (solo MoE) | Router que decide qué expertos activar |

### 2.2 Pesos que NO se modificaron

- ❌ `up_proj` / `gate_proj` del MLP (entrada)
- ❌ `q_proj`, `k_proj`, `v_proj` de atención (entrada)
- ❌ `embed_tokens` (embedding inicial)
- ❌ `lm_head` (no aparece en top modificados)
- ❌ `layernorm` (normalización)

### 2.3 Patrón observado

Solo se modificaron las **proyecciones de SALIDA**:
- `o_proj`: Salida de la capa de atención → hidden states
- `down_proj`: Salida de los expertos MLP → hidden states
- `gate.weight`: Routing de tokens a expertos (afecta qué activaciones fluyen)

**Conclusión**: huihui-ai ortogonalizó específicamente los puntos donde la información "fluye de vuelta" al residual stream.

---

## 3. Análisis por Capa

### 3.1 Distribución de cambios relativos

```
Cambio relativo promedio: ~3-4%
Rango: 1.5% - 4.5%
```

### 3.2 Top 20 tensores más modificados

| Rank | Capa | Tensor | Cambio relativo |
|------|------|--------|-----------------|
| 1 | 26 | experts.9.down_proj | 4.46% |
| 2 | 32 | experts.21.down_proj | 4.18% |
| 3 | 23 | experts.11.down_proj | 3.88% |
| 4 | 24 | experts.24.down_proj | 3.87% |
| 5 | 25 | shared_experts.down_proj | 3.87% |
| 6 | 25 | experts.2.down_proj | 3.87% |
| 7 | **26** | **gate.weight** | **3.83%** |
| 8 | 27 | shared_experts.down_proj | 3.77% |
| 9 | 22 | shared_experts.down_proj | 3.75% |
| 10 | 21 | experts.42.down_proj | 3.74% |
| 11 | 21 | shared_experts.down_proj | 3.73% |
| 12 | 26 | shared_experts.down_proj | 3.60% |
| 13 | 22 | experts.32.down_proj | 3.58% |
| 14 | **24** | **gate.weight** | **3.57%** |
| 15 | 22 | experts.18.down_proj | 3.57% |
| 16 | 26 | experts.7.down_proj | 3.57% |
| 17 | 22 | experts.39.down_proj | 3.57% |
| 18 | 24 | shared_experts.down_proj | 3.54% |
| 19 | 22 | experts.40.down_proj | 3.46% |
| 20 | 20 | shared_experts.down_proj | 3.43% |

### 3.3 Zona de mayor impacto

Las capas **20-27** (~40-60% del modelo de 48 capas) tienen los mayores cambios.

```
Capa 20: 3.43% (shared_experts)
Capa 21: 3.73-3.74%
Capa 22: 3.46-3.75%
Capa 23: 3.88%
Capa 24: 3.54-3.87%
Capa 25: 3.77-3.87%
Capa 26: 3.57-4.46% ← MÁXIMO
Capa 27: 3.77%
```

**Esto coincide con la teoría del paper**: La dirección de rechazo se codifica principalmente en las capas intermedias (~40-60%).

---

## 4. Análisis de la Técnica

### 4.1 Evidencia de ortogonalización

La ortogonalización matemática modifica los pesos con la fórmula:

```
W' = W - v @ (W @ v).T
```

Donde `v` es la dirección de rechazo.

**Características esperadas de ortogonalización:**
1. ✓ Cambios pequeños y uniformes (~3-4%)
2. ✓ Afecta todas las capas
3. ✓ Diferencia de rango bajo (verificar con SVD)
4. ✓ Modifica proyecciones de salida

### 4.2 ¿Una dirección o múltiples direcciones?

**Hipótesis A: Una sola dirección global**
- Consistente con cambios relativos similares (~3-4%) en todas las capas
- El paper sugiere que una sola dirección es suficiente

**Hipótesis B: Direcciones diferentes por capa**
- Explicaría variación en expertos específicos (algunos expertos cambian más que otros)
- Más complejo de implementar

**Evidencia**: Los cambios relativos son bastante uniformes (~3-4%), lo que sugiere **una sola dirección global**.

### 4.3 Hallazgo importante: Los gates se modifican

Los `gate.weight` aparecen en el top 20 de cambios:
- Capa 26: 3.83%
- Capa 24: 3.57%

Esto es significativo porque:
1. Los gates controlan qué expertos procesan cada token
2. Ortogonalizar los gates puede hacer que tokens "de rechazo" se enruten a expertos diferentes
3. **Vuestra implementación original NO modificaba los gates** - ahora corregido

---

## 5. Comparación: Nuestro vs huihui-ai

| Aspecto | Nuestra implementación (original) | huihui-ai | Estado |
|---------|-----------------------------------|-----------|--------|
| Capas modificadas | Todas | Todas | ✓ OK |
| `down_proj` ortogonalizado | Sí | Sí | ✓ OK |
| `o_proj` ortogonalizado | Sí | Sí | ✓ OK |
| `gate.weight` ortogonalizado | **NO** | Sí | ❌ → ✓ CORREGIDO |
| Dirección usada | Capa 24 | Desconocido | ⚠️ Verificar |
| Prompts de evaluación | Aleatorios | Fijos | ❌ → ✓ CORREGIDO |
| Cambio relativo | ~5.0 (norma) | ~3-4% | ⚠️ Posible diferencia |

---

## 6. Cálculo de la norma de la dirección

### huihui-ai (inferido de los datos)

Para una matriz de shape (2048, 1536):
- Norma de Frobenius base: ~40
- Norma de la diferencia: ~1.5

Si la diferencia es `v @ (W @ v).T`:
```
||diff|| ≈ ||v|| × ||W @ v|| ≈ ||v|| × ||W|| × ||v|| = ||v||² × ||W||
```

Con `||diff|| ≈ 1.5` y `||W|| ≈ 40`:
```
||v||² ≈ 1.5 / 40 ≈ 0.0375
||v|| ≈ 0.19
```

**Dirección aparentemente normalizada** (norma ~0.2 o ~1.0 con escala).

### Nuestra implementación

```
direction_norm: 5.0 (del resultado)
```

**Posible problema**: Nuestra dirección tiene norma mayor que la que parece haber usado huihui-ai.

---

## 7. Conclusiones

### Lo que hizo huihui-ai (confirmado)

1. **Técnica**: Ortogonalización global de proyecciones de salida
2. **Pesos modificados**: `down_proj`, `o_proj`, `gate.weight`
3. **Capas**: Todas (0-47)
4. **Zona de mayor impacto**: Capas 20-27 (~40-60%)
5. **Cambio relativo**: ~3-4% (pequeño pero consistente)

### Lo que NO sabemos con certeza (pendiente SVD)

1. Si usó una sola dirección global o direcciones por capa
2. La dirección exacta que usó
3. Si normalizó la dirección y con qué escala

### Diferencias con nuestra implementación (ya corregidas)

1. ~~No modificamos `gate.weight`~~ → CORREGIDO
2. ~~Usamos prompts aleatorios~~ → CORREGIDO
3. ⚠️ Nuestra norma de dirección (5.0) puede ser diferente

---

## 8. Recomendación adicional

Considerar normalizar la dirección de rechazo antes de ortogonalizar:

```python
# En lugar de:
self.refusal_dir = self.refusal_dir_unnormalized / self.refusal_dir_unnormalized.norm()

# Considerar escalar a una norma más pequeña:
scale_factor = 0.2  # O calcular basándose en los cambios de huihui
self.refusal_dir = (self.refusal_dir_unnormalized / self.refusal_dir_unnormalized.norm()) * scale_factor
```

Esto produciría cambios más pequeños (~3-4%) similares a los de huihui-ai.
