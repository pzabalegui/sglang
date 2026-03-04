# Despliegue de GLM-5-FP8 con SGLang

## Requisitos

- **Hardware:** 8x NVIDIA H200 (o equivalente con ~1TB VRAM total)
- **Modelo:** [zai-org/GLM-5-FP8](https://huggingface.co/zai-org/GLM-5-FP8) (~705GB)
- **Framework:** SGLang

---

## Problema Principal

GLM-5 usa la arquitectura `glm_moe_dsa` que **no está soportada** en las versiones estables de `transformers`. Requiere `transformers >= 5.x (dev)`.

---

## Instalación Paso a Paso

### 1. Actualizar transformers a versión dev (OBLIGATORIO)

```bash
pip install git+https://github.com/huggingface/transformers.git
```

> Sin esto, SGLang fallará con: `KeyError: 'glm_moe_dsa'`

### 2. Verificar que el modelo existe

```bash
ls /workspace/huggingface/hub/models--zai-org--GLM-5-FP8/snapshots/
```

Si no existe, descargarlo:
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-5-FP8', local_dir='/workspace/models/GLM-5-FP8')"
```

### 3. Limpiar procesos anteriores

```bash
pkill -9 sglang
pkill -9 python
```

### 4. Lanzar SGLang

```bash
python -m sglang.launch_server \
  --model-path /workspace/huggingface/hub/models--zai-org--GLM-5-FP8/snapshots/949e09d70b611943740a3d79a0f46a63947498a6 \
  --tp 8 \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.85
```

### 5. Verificar funcionamiento

```bash
curl http://localhost:8000/v1/models
```

---

## Problemas Encontrados y Soluciones

| Problema | Error | Solución |
|----------|-------|----------|
| transformers no reconoce GLM-5 | `KeyError: 'glm_moe_dsa'` | `pip install git+https://github.com/huggingface/transformers.git` |
| OOM al iniciar | `CUDA out of memory` | Limpiar procesos: `pkill -9 python` |
| Disco lleno | `No space left on device` | Usar `/workspace` en lugar de `/root` |
| Puerto no accesible externamente | Connection refused | Usar proxy Runpod: `https://{pod_id}-8000.proxy.runpod.net` |

---

## Uso de Memoria

| Componente | Por GPU | Total (8 GPUs) |
|------------|---------|----------------|
| Pesos FP8 | ~88GB | ~700GB |
| KV Cache | ~30GB | ~240GB |
| CUDA Graphs | ~10GB | ~80GB |
| **Total** | ~131GB | ~1TB |

---

## API Endpoints

```bash
# Listar modelos
curl http://localhost:8000/v1/models

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "glm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'
```

---

## Opciones Adicionales

| Flag | Descripción |
|------|-------------|
| `--log-requests` | Loguea contenido de peticiones |
| `--mem-fraction-static 0.85` | Usa 85% de VRAM para el modelo |
| `--tp 8` | Tensor Parallelism en 8 GPUs |

---

## Acceso Externo (Runpod)

En Runpod, los puertos HTTP se acceden via proxy:

```
https://{POD_ID}-{PORT}.proxy.runpod.net
```

Ejemplo:
```
https://740i5qf0r6zoeu-8000.proxy.runpod.net/v1/chat/completions
```

---

## Notas

- El modelo tarda ~2-3 minutos en cargar completamente
- Primera inferencia puede tardar más por compilación JIT de DeepGEMM
- Peticiones secuenciales no saturan memoria; concurrentes sí pueden
