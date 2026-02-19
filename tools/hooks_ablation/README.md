# Hooks Ablation - Abliteración en Tiempo de Inferencia

Scripts para abliteración **reversible** mediante PyTorch hooks durante la inferencia.

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `ablation_sweep.py` | Búsqueda de parámetros óptimos |
| `sglang_steering_server.py` | Servidor HTTP con API OpenAI-compatible |

---

## ablation_sweep.py

Encuentra la configuración óptima de hooks para cualquier modelo.

```bash
# Sweep estándar
python ablation_sweep.py --model zai-org/GLM-4.7-Flash --output ./sweep_results

# Sweep rápido
python ablation_sweep.py --model tu-modelo --quick

# Sweep exhaustivo
python ablation_sweep.py --model tu-modelo --exhaustive
```

### Parámetros que Barre

- **Capas**: 35%-70% de profundidad
- **Escalas**: 0.5 - 3.0
- **Modos**: single, window, gaussian
- **Ventana**: 1-11 capas
- **Kernel**: 2.0 - 10.0

### Salida

```
sweep_results/
├── analysis_*.json           # Mejor configuración
├── all_results_*.json        # Todos los resultados
└── refusal_direction_*.pt    # Vector óptimo
```

---

## sglang_steering_server.py

Servidor HTTP para steering dinámico por request.

```bash
# Iniciar con extracción automática
python sglang_steering_server.py --model zai-org/GLM-4.7-Flash --extract-on-startup --port 8000

# Con vector pre-calculado
python sglang_steering_server.py --model zai-org/GLM-4.7-Flash --vector refusal_direction.pt --port 8000
```

### API

```bash
# Sin steering
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "GLM", "messages": [{"role": "user", "content": "prompt"}]}'

# Con steering
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "GLM",
        "messages": [{"role": "user", "content": "prompt"}],
        "steering": {"enabled": true, "scale": 2.0, "mode": "window", "center_layer": 24}
    }'
```

---

## Hooks vs Ortogonalización

| Aspecto | Hooks | Ortho |
|---------|-------|-------|
| Reversible | Si | No |
| Control por request | Si | No |
| Overhead | ~15% | 0% |
| Mejor para | Testing, A/B | Deploy |
