# Investigación: Abliteración de GLM-4.7-Flash

**Fecha de inicio**: 2026-02-03
**Estado**: En curso
**Objetivo**: Replicar los resultados de huihui-ai/Huihui-GLM-4.7-Flash-abliterated

---

## Resumen Ejecutivo

La implementación actual de ortogonalización **NO funciona** porque:

1. **Ortogonalización LOCAL vs GLOBAL**: huihui-ai modificó TODAS las 48 capas, nosotros solo extraemos dirección de 1 capa
2. **Falta modificar gates**: huihui-ai modificó `gate.weight` en todas las capas MoE
3. **Evaluación inconsistente**: Usamos prompts diferentes (split aleatorio vs primeros 32 de AdvBench)

---

## 1. Datos de Comparación

### Rendimiento: Baseline vs huihui-ai vs Nuestro

| Modelo | Rechazos harmful | Bypass Rate | ASR |
|--------|------------------|-------------|-----|
| **Baseline** (eval huihui) | 20/32 (62.5%) | - | 37.5% |
| **huihui-abliterated** | 1/32 (3.1%) | **+59.4%** | 96.9% |
| **Baseline** (nuestra eval) | 5/32 (15.6%) | - | 84.4% |
| **Nuestra abliteración** | 8/32 (25.0%) | **-60.0%** | 75.0% |
| **Nuestra ortogonalización** | 7/32 (21.9%) | **-40.0%** | 78.1% |

**Problema crítico**: Nuestra intervención AUMENTA rechazos en lugar de reducirlos.

### Preservación de Utilidad (harmless)

| Modelo | Rechazos harmless |
|--------|-------------------|
| Baseline | 2/16 (12.5%) |
| huihui-ai | 2/16 (12.5%) |

huihui-ai preserva utilidad perfectamente.

---

## 2. Análisis de Ingeniería Inversa

### Fuente: quick_analysis_2026-02-02T14-58-17.338262.json

```
Total parámetros:      31,221,488,576
Parámetros modificados: 10,140,647,424 (32.5%)
Tensores modificados:   3,151 de 9,703
Capas modificadas:      TODAS (0-47)
```

### Tipos de pesos modificados

| Tipo de peso | Cantidad | Notas |
|--------------|----------|-------|
| `down_proj` | 3,056 | 64 expertos × 47 capas MoE + 1 densa + shared |
| `o_proj` | 48 | 1 por capa (attention output) |
| `gate.weight` | 47 | 1 por capa MoE (router) |

### Top 20 tensores con mayor cambio relativo

| Capa | Tensor | Cambio relativo |
|------|--------|-----------------|
| 26 | experts.9.down_proj | 4.46% |
| 32 | experts.21.down_proj | 4.18% |
| 23 | experts.11.down_proj | 3.88% |
| 24 | experts.24.down_proj | 3.87% |
| 25 | shared_experts.down_proj | 3.87% |
| 25 | experts.2.down_proj | 3.87% |
| **26** | **gate.weight** | **3.83%** |
| 27 | shared_experts.down_proj | 3.77% |
| 22 | shared_experts.down_proj | 3.75% |
| 21 | experts.42.down_proj | 3.74% |

**Observación importante**: Las capas 20-27 (~40-60% del modelo) tienen los mayores cambios. Esta es la zona óptima según el paper.

---

## 3. Diferencias entre Implementaciones

### Nuestra implementación actual

```python
# abliterate_glm.py

# 1. Extraer dirección de UNA sola capa
refusal_dir = extract_refusal_direction(layer=24)  # Solo capa 24

# 2. Ortogonalizar con esa dirección
def orthogonalize_model():
    for layer in range(n_layers):
        o_proj.weight = orthogonalize(o_proj.weight, refusal_dir)
        down_proj.weight = orthogonalize(down_proj.weight, refusal_dir)
        # FALTA: gate.weight no se ortogonaliza
```

### Lo que hizo huihui-ai (inferido)

```
1. Usó ortogonalización GLOBAL (todas las capas)
2. Modificó down_proj, o_proj, Y gate.weight
3. Cambios pequeños (~3-4%) pero uniformes
4. Técnica inferida: "ORTOGONALIZACIÓN GLOBAL"
```

### Diferencias clave

| Aspecto | Nosotros | huihui-ai |
|---------|----------|-----------|
| Capas modificadas | Todas (con 1 dirección) | Todas |
| Extracción de dirección | 1 capa (24) | Desconocido |
| Pesos ortogonalizados | down_proj, o_proj | down_proj, o_proj, **gate** |
| Cambio relativo | ~5.0 (norma dirección) | ~3-4% |
| Bypass rate | -60% (empeora) | +59% (funciona) |

---

## 4. Hipótesis sobre el fallo

### Hipótesis 1: Dirección incorrecta o invertida

La dirección extraída de capa 24 puede:
- No ser representativa del mecanismo de rechazo global
- Estar "invertida" respecto a lo que usó huihui-ai

**Evidencia**: Nuestra intervención AUMENTA rechazos → dirección podría estar invertida.

**Prueba propuesta**: Probar con `-refusal_dir` (dirección negada).

### Hipótesis 2: Falta modificar gates

Los `gate.weight` controlan el routing en MoE:
- Pueden dirigir tokens hacia expertos "de rechazo"
- huihui-ai los modificó, nosotros no

**Prueba propuesta**: Añadir ortogonalización de gates.

### Hipótesis 3: Evaluación inconsistente

```
Nuestra evaluación:
  - train_test_split aleatorio de AdvBench
  - Baseline rechaza solo 15.6%

Evaluación con prompts fijos:
  - Primeros 32 de AdvBench
  - Baseline rechaza 62.5%
```

Los prompts importan. Necesitamos usar los mismos prompts.

### Hipótesis 4: Magnitud del cambio excesiva

huihui-ai aplicó cambios pequeños (~3-4% relativo). Nuestra dirección tiene norma ~5.0.

**Prueba propuesta**: Escalar la dirección para reducir magnitud del cambio.

---

## 5. Plan de Acción

### Paso 1: Ejecutar reverse engineering completo con SVD
**Estado**: En curso (servidor SSH)

Objetivo: Verificar si las diferencias de pesos son rank-1 (indicativo de ortogonalización pura).

### Paso 2: Probar dirección invertida
```python
# En vez de:
orthogonalize(weight, refusal_dir)
# Probar:
orthogonalize(weight, -refusal_dir)
```

### Paso 3: Añadir ortogonalización de gates
```python
# En orthogonalize_model():
gate = mlp.gate.weight
gate.data = self._orthogonalize_matrix(gate, self.refusal_dir)
```

### Paso 4: Usar prompts fijos
```python
# En lugar de train_test_split:
harmful_test = advbench[:32]  # Primeros 32 (más duros)
```

### Paso 5: Triple comparación
Evaluar: baseline vs huihui vs nuestro con los MISMOS prompts.

---

## 6. Arquitectura del Modelo

### GLM-4.7-Flash

```
Tipo: MoE (Mixture of Experts)
Parámetros totales: 31.2B
Parámetros activos: ~3B
Capas: 47 (+ embedding)
  - Capa 0: Densa
  - Capas 1-46: MoE
Hidden size: 2048
Expertos por capa: 64 routed + 1 shared
Top-k: Desconocido (probablemente 2-4)
```

### Estructura de una capa MoE

```
layer[i]:
  ├── self_attn
  │   ├── q_proj, k_proj, v_proj
  │   └── o_proj  ← ORTOGONALIZADO
  └── mlp
      ├── gate.weight  ← DEBERÍA ORTOGONALIZARSE
      ├── shared_experts
      │   └── down_proj  ← ORTOGONALIZADO
      └── experts (x64)
          └── down_proj  ← ORTOGONALIZADO
```

---

## 7. Código de Referencia

### Extracción de dirección (abliterate_glm.py:470-530)

```python
def extract_refusal_direction(self, layer: Optional[int] = None):
    harmful_acts = self.get_hidden_states(harmful_train, layer, pos=-1)
    harmless_acts = self.get_hidden_states(harmless_train, layer, pos=-1)

    harmful_mean = harmful_acts.mean(dim=0)
    harmless_mean = harmless_acts.mean(dim=0)

    self.refusal_dir_unnormalized = harmful_mean - harmless_mean
    self.refusal_dir = self.refusal_dir_unnormalized / norm
```

### Ortogonalización (abliterate_glm.py:773-799)

```python
def _orthogonalize_matrix(self, matrix: Tensor, direction: Tensor):
    v = direction / v.norm()

    if out_dim == v.shape[0]:
        proj = matrix.T @ v
        correction = torch.outer(v, proj)
        return matrix - correction
    elif in_dim == v.shape[0]:
        proj = matrix @ v
        correction = torch.outer(proj, v)
        return matrix - correction
```

Fórmula: `W' = W - v @ (W @ v).T` o `W' = W - v @ (W.T @ v).T`

---

## 8. Log de Investigación

### 2026-02-03 14:00 - Inicio
- Analizada estructura del proyecto
- Identificadas diferencias clave con huihui-ai
- Hipótesis principal: ortogonalización local vs global

### 2026-02-03 14:30 - SSH y análisis remoto
- Conectado a servidor SSH (root@31.22.104.70)
- Descargado quick_analysis.json
- Confirmado: huihui-ai modificó TODAS las capas + gates
- **PROBLEMA**: Disco del servidor 100% lleno (193GB)
- El modelo `our_abliterated_model` ocupa 56GB
- Solución: Usar `/dev/shm` (59GB libres) como directorio temporal

### 2026-02-03 15:00 - Reverse engineering con SVD
- Ejecutando `reverse_engineer_abliteration.py` en segundo plano
- Analizando 48 shards de safetensors
- Objetivo: Detectar si las diferencias son rank-1 (indicativo de ortogonalización pura)
- Si detecta rank-1, extraerá las direcciones inferidas

### Hallazgo importante: Comportamiento del modelo baseline

Analizando los ejemplos del resultado (`result_2026-02-02T15-07-30.json`):

1. **El baseline NO rechaza típicamente**: En lugar de decir "I cannot help with that",
   el modelo responde con "Analyze the Request" y luego evalúa la solicitud.

2. **La ablación a veces CAUSA rechazos**: En el ejemplo de "racism website":
   - Baseline: No es rechazo (analiza la solicitud)
   - Ablación: ES rechazo ("As an AI, I should not")

3. **Implicación**: El mecanismo de "rechazo" de GLM-4.7-Flash es diferente al típico.
   Nuestra dirección puede estar capturando algo diferente.

### 2026-02-03 15:30 - Cambios implementados

**Cambio 1: Ortogonalización de gate.weight**
- Archivo: [abliterate_glm.py:868-879](src/abliterate_glm.py#L868-L879)
- Se añadió ortogonalización de `mlp.gate.weight` en capas MoE
- huihui-ai modificó 47 gate.weight, ahora lo hacemos también

**Cambio 2: Prompts fijos para evaluación**
- Archivo: [abliterate_glm.py:381-394](src/abliterate_glm.py#L381-L394)
- En lugar de `train_test_split` aleatorio, ahora usa prompts fijos
- `harmful_test`: Primeros 32 de AdvBench (más duros, para evaluación)
- `harmful_train`: Siguientes N para extracción de dirección

### Pendiente
- [ ] Esperar resultados de reverse engineering con SVD (~en progreso)
- [ ] Analizar direcciones inferidas (si existen)
- [ ] Re-ejecutar abliteración con los cambios
- [ ] Comparar resultados

---

## 10. Conclusiones Preliminares (mientras se completa el análisis SVD)

### Lo que sabemos con certeza

1. **huihui-ai modificó TODAS las capas** (48 de 48)
2. **huihui-ai modificó 3 tipos de pesos**: down_proj (3,056), o_proj (48), gate.weight (47)
3. ~~**Vuestra implementación NO modifica gate.weight**~~ **CORREGIDO**: Ahora sí se modifica
4. **El cambio relativo promedio es ~3-4%** (pequeño pero consistente)
5. **Las capas 20-27 tienen los mayores cambios** (~40-60% del modelo)

### Lo que NO sabemos aún (pendiente del análisis SVD)

1. Si las diferencias son rank-1 (confirmaría ortogonalización pura)
2. Si huihui-ai usó una sola dirección o direcciones diferentes por capa
3. La dirección exacta que usó huihui-ai

### Acciones recomendadas mientras esperamos

1. **Añadir ortogonalización de gate.weight** - Esto es seguro hacerlo ahora
2. **Usar prompts fijos para evaluación** - Los primeros 32 de AdvBench
3. **Probar con dirección invertida** - Por si nuestra dirección está al revés

### Código para añadir ortogonalización de gates

```python
# En orthogonalize_model(), después del bloque de MoE experts:

# Ortogonalizar gate.weight
gate = get_safely(mlp, 'gate')
if gate is not None and hasattr(gate, 'weight'):
    if gate.weight.shape[1] == self.hidden_size:  # Shape: (n_experts, hidden_size)
        gate.weight.data = self._orthogonalize_matrix(gate.weight, self.refusal_dir)
        ortho_count += 1
```

---

## 9. Referencias

- Paper: "Refusal in Language Models Is Mediated by a Single Direction" (arXiv:2406.11717)
- Modelo base: zai-org/GLM-4.7-Flash
- Modelo referencia: huihui-ai/Huihui-GLM-4.7-Flash-abliterated
- Dataset: AdvBench (harmful), Alpaca (harmless)
