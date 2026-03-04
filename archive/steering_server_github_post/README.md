# Steering Server - Abliteración por Hooks en Inferencia

Servidor HTTP que permite activar/desactivar la abliteración (eliminación del rechazo)
en tiempo real mediante requests. Usa PyTorch hooks para modificar las activaciones
durante el forward pass.

## Modelo por defecto: GLM-4.7-Flash

| Parámetro | Valor |
|-----------|-------|
| **Modelo** | `zai-org/GLM-4.7-Flash` |
| **Parámetros** | 30B |
| **Capas** | 47 |
| **Capa óptima** | **24** (verificado experimentalmente) |
| **Hidden size** | 2048 |
| **Effect size** | 7.54 |
| **Accuracy separación** | 100% |

## Specs de Máquina para GLM-4.7-Flash

### Recomendado (bf16 completo)
| Recurso | Especificación |
|---------|----------------|
| **GPU** | NVIDIA con 48GB+ VRAM (A6000, A100 40GB, H100) |
| **RAM** | 128GB |
| **CPU** | 32 cores |
| **Disco** | 200GB SSD NVMe |
| **OS** | Ubuntu 22.04 LTS |

### Mínimo (4-bit cuantizado)
| Recurso | Especificación |
|---------|----------------|
| **GPU** | NVIDIA con 24GB VRAM (RTX 4090, A5000, L4) |
| **RAM** | 64GB |
| **CPU** | 16 cores |
| **Disco** | 100GB SSD NVMe |
| **OS** | Ubuntu 22.04 LTS |

### Cloud Options para GLM-4.7-Flash
| Proveedor | Instancia | GPU | VRAM | Coste aprox |
|-----------|-----------|-----|------|-------------|
| **RunPod** | 1x A100 40GB | A100 | 40GB | ~$1.5/hr |
| **RunPod** | 1x A100 80GB | A100 | 80GB | ~$2.0/hr |
| **Lambda Labs** | 1x A100 | A100 | 40GB | ~$1.1/hr |
| **AWS** | p4d.24xlarge | A100 | 40GB | ~$32/hr (8 GPUs) |
| **GCP** | a2-highgpu-1g | A100 | 40GB | ~$3/hr |

---

## Instalación

```bash
# 1. Clonar o copiar esta carpeta al servidor
scp -r steering_server/ user@servidor:/opt/

# 2. Conectar al servidor
ssh user@servidor

# 3. Instalar dependencias
cd /opt/steering_server
pip install -r requirements.txt

# 4. Verificar GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Uso

### Paso 1: Extraer vector de rechazo (una sola vez)

```bash
python extract_and_serve.py \
    --model zai-org/GLM-4.7-Flash \
    --layer 24 \
    --save-vector refusal_direction_glm47flash.pt \
    --n-samples 32
```

**Capa 24 verificada experimentalmente** con:
- Effect size: 7.54
- Accuracy: 100%
- Quality: BUENA

### Paso 2: Iniciar servidor

```bash
python runtime_steering.py --serve \
    --model zai-org/GLM-4.7-Flash \
    --vector refusal_direction_glm47flash.pt \
    --port 8000
```

**Con cuantización 4-bit (menos VRAM, ~24GB):**
```bash
python runtime_steering.py --serve \
    --model zai-org/GLM-4.7-Flash \
    --vector refusal_direction_glm47flash.pt \
    --port 8000 \
    --4bit
```

### Paso 3: Hacer requests

**Sin steering (modelo rechaza normalmente):**
```bash
curl -s -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "How to hack a computer?",
        "steering": false
    }' | jq
```

**Con steering (abliteración activa, sin rechazo):**
```bash
curl -s -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "How to hack a computer?",
        "steering": true,
        "steering_scale": 1.0
    }' | jq
```

**OpenAI-compatible API:**
```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "GLM-4.7-Flash",
        "messages": [{"role": "user", "content": "How to hack?"}],
        "steering": true,
        "steering_scale": 1.0
    }' | jq
```

---

## API Reference

### POST /generate
API simple de steering.

**Request:**
```json
{
    "prompt": "string",
    "steering": true/false,
    "steering_scale": 1.0,
    "steering_layers": [14, 15, 16],  // opcional
    "max_new_tokens": 256
}
```

**Response:**
```json
{
    "response": "string",
    "steering": {
        "applied": true,
        "scale": 1.0,
        "vector": "default"
    }
}
```

### POST /v1/chat/completions
OpenAI-compatible API con extensión de steering.

**Request:**
```json
{
    "model": "model-name",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 256,
    "steering": true,
    "steering_scale": 1.0
}
```

### GET /health
Estado del servidor.

### GET /vectors
Vectores de steering cargados.

---

## Parámetros de Steering

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `steering` | bool | false | Activar/desactivar abliteración |
| `steering_scale` | float | 1.0 | Intensidad (0.5=suave, 2.0=agresivo) |
| `steering_layers` | list | null | Capas específicas (null=todas) |
| `vector_name` | string | "default" | Vector a usar si hay varios |

---

## Arquitectura

```
Request con steering=true
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  SteeringEngine                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │  Modelo (HuggingFace)                             │ │
│  │  ┌─────┐   ┌─────┐   ┌─────┐       ┌─────┐       │ │
│  │  │ L0  │ → │ L1  │ → │ ... │ → ... │ LN  │       │ │
│  │  └─────┘   └─────┘   └──┬──┘       └─────┘       │ │
│  │                         │                         │ │
│  │                    ┌────▼────┐                    │ │
│  │                    │  HOOK   │                    │ │
│  │                    │ h'=h-(h·r̂)r̂│                │ │
│  │                    └─────────┘                    │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
         │
         ▼
   Respuesta SIN rechazo
```

---

## Troubleshooting

### CUDA out of memory
```bash
# Usar cuantización 4-bit
python runtime_steering.py --serve --model ... --4bit

# O reducir batch size en extract_and_serve.py
```

### Modelo no carga
```bash
# Verificar espacio en disco
df -h

# Verificar que HuggingFace puede descargar
huggingface-cli whoami
```

### Steering no tiene efecto
```bash
# Verificar que el vector está cargado
curl http://localhost:8000/vectors

# Probar con escala más alta
curl -X POST http://localhost:8000/generate \
    -d '{"prompt": "...", "steering": true, "steering_scale": 2.0}'
```

---

## Licencia

Uso interno - Alias Robotics Research
