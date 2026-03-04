# Estado de la Investigación - Abliteración GLM-4.7-Flash

**Fecha de snapshot**: 2026-02-03 10:39
**Motivo**: Reconfiguración del servidor remoto (aumento de disco)

---

## 1. Servidor Remoto

- **Host**: root@31.22.104.70
- **Problema**: Disco 100% lleno (193GB)
- **Solución**: Aumentar disco a 300GB+

---

## 2. Estado de los Experimentos

### 2.1 Modelo abliterado v2

**Ubicación**: `/root/refusal_direction/our_abliterated_model_v2/` (55GB)

**Cambios aplicados:**
- ✅ Ortogonalización de `gate.weight` (47 tensores adicionales)
- ✅ Prompts fijos para evaluación (primeros 32 de AdvBench)
- ✅ Escalado de dirección: `direction_scale=0.3`

**Estado**: El modelo se guardó correctamente, pero no se pudo evaluar completamente por falta de espacio.

### 2.2 Reverse Engineering (SVD)

**Script**: `reverse_engineer_abliteration.py`
**Estado al momento del snapshot**: ~21% completado (10/48 shards)

**Objetivo**:
- Detectar si las diferencias entre baseline y huihui son rank-1
- Extraer las direcciones inferidas por capa
- Confirmar qué técnica usó huihui-ai

**Resultado esperado**: Si las diferencias son rank-1, tendremos las direcciones exactas.

### 2.3 Triple Comparison

**Resultado parcial** (solo baseline completado):
```
Baseline: 15/32 rechazos (46.9%)
Huihui: Error - No space left on device
Ours: Error - protobuf not installed (disk full)
```

---

## 3. Hallazgos Clave

### 3.1 Qué hizo huihui-ai (confirmado)

| Aspecto | Detalle |
|---------|---------|
| Técnica | Ortogonalización global |
| Tensores modificados | 3,151 de 9,703 (32.5%) |
| Parámetros modificados | 10,140,647,424 (32.5%) |
| Capas modificadas | TODAS (0-47) |
| Tipos de pesos | down_proj (3,056), o_proj (48), gate.weight (47) |
| Cambio relativo promedio | ~3-4% |
| Zona de mayor impacto | Capas 20-27 (~40-60% del modelo) |

### 3.2 Problema crítico identificado

**Nuestra ortogonalización AUMENTA los rechazos en lugar de reducirlos.**

Resultados anteriores:
```
Baseline: 3/32 (9.4%) rechazos
Ortogonalización: 10/32 (31.2%) rechazos
Bypass Rate: -233.3% (EMPEORA)
```

**Hipótesis a investigar:**
1. La dirección está invertida (probar con `-direction`)
2. La escala sigue siendo incorrecta
3. El mecanismo de rechazo de GLM-4.7-Flash es diferente

---

## 4. Pasos a Seguir (cuando el servidor esté listo)

### Paso 1: Verificar el modelo guardado
```bash
ls -la /root/refusal_direction/our_abliterated_model_v2/
```

### Paso 2: Completar o reiniciar reverse engineering
```bash
# Verificar si sigue corriendo
ps aux | grep reverse

# Si no, reiniciar
cd /root/refusal_direction
TMPDIR=/dev/shm python reverse_engineer_abliteration.py
```

### Paso 3: Instalar dependencias faltantes
```bash
TMPDIR=/dev/shm pip install protobuf bitsandbytes
```

### Paso 4: Ejecutar triple comparison
```bash
cd /root/refusal_direction
TMPDIR=/dev/shm HF_HOME=/dev/shm python triple_comparison.py
```

### Paso 5: Probar dirección invertida
Modificar `abliterate_glm_v2.py` para usar `-refusal_dir` en lugar de `+refusal_dir`.

---

## 5. Archivos Respaldados

| Archivo | Descripción |
|---------|-------------|
| `abliterate_glm_v2.py` | Script principal con cambios |
| `triple_comparison.py` | Script de comparación triple |
| `reverse_engineer_abliteration.py` | Script de ingeniería inversa |
| `quick_weight_diff.py` | Análisis rápido de diferencias |
| `quick_analysis_*.json` | Análisis de pesos de huihui vs baseline |
| `comparison_*.json` | Comparación de respuestas huihui vs baseline |
| `shm_triple_*.json` | Resultados parciales de triple comparison |
| `reverse_engineering_progress.txt` | Progreso del análisis SVD |

---

## 6. Comandos Útiles

### Verificar espacio en disco
```bash
df -h /root /dev/shm
```

### Verificar GPU
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Ejecutar script con TMPDIR en shm
```bash
TMPDIR=/dev/shm HF_HOME=/dev/shm python script.py
```

---

## 7. Contacto y Referencias

- **Paper**: "Refusal in Language Models Is Mediated by a Single Direction" (arXiv:2406.11717)
- **Modelo base**: zai-org/GLM-4.7-Flash
- **Modelo referencia**: huihui-ai/Huihui-GLM-4.7-Flash-abliterated
- **Dataset**: AdvBench (harmful), Alpaca (harmless)
