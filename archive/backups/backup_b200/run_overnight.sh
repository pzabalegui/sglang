#!/bin/bash
# =============================================================================
# SCRIPT NOCTURNO - ABLITERACIÓN GLM-4.7
# =============================================================================
# Ejecutar con: nohup bash /root/run_overnight.sh > /root/overnight.log 2>&1 &

set -e
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=/root/logs_$TIMESTAMP
mkdir -p $LOG_DIR

echo "========================================"
echo "INICIO: $(date)"
echo "========================================"

# -----------------------------------------------------------------------------
# PASO 0: LIMPIEZA DE DISCO
# -----------------------------------------------------------------------------
echo ""
echo "[PASO 0] Limpiando disco..."
echo "Espacio antes:"
df -h /

# Borrar modelo FP8 (no lo usamos)
echo "Borrando cache de GLM-4.7-FP8..."
rm -rf ~/.cache/huggingface/hub/models--zai-org--GLM-4.7-FP8

echo "Espacio después:"
df -h /

# -----------------------------------------------------------------------------
# PASO 1: SWEEP COMPLETO (para plotting)
# -----------------------------------------------------------------------------
echo ""
echo "[PASO 1] Ejecutando SWEEP de capas 30-65..."
echo "Inicio: $(date)"

python3 /root/abliterate_glm47_v2.py \
    --model zai-org/GLM-4.7 \
    --sweep \
    --sweep-start 30 \
    --sweep-end 65 \
    --sweep-step 1 \
    --save-activations \
    --output /root/output_sweep_full \
    --batch-size 1 \
    --log-file $LOG_DIR/sweep_full.log

echo "Sweep completado: $(date)"

# -----------------------------------------------------------------------------
# PASO 2: ABLITERACIÓN FINAL + GUARDAR MODELO
# -----------------------------------------------------------------------------
echo ""
echo "[PASO 2] Ejecutando abliteración final L47/width=10..."
echo "Inicio: $(date)"

python3 /root/abliterate_glm47_v2.py \
    --model zai-org/GLM-4.7 \
    --layer 47 \
    --attn-scale 5.0 \
    --kernel-mode gaussian \
    --kernel-width 10 \
    --norm-preserve \
    --save-activations \
    --save-model \
    --output /root/glm47_abliterated_final \
    --batch-size 1 \
    --log-file $LOG_DIR/abliterate_final.log

echo "Abliteración completada: $(date)"

# -----------------------------------------------------------------------------
# PASO 3: VERIFICACIÓN
# -----------------------------------------------------------------------------
echo ""
echo "[PASO 3] Verificando resultados..."

echo "Archivos generados:"
ls -la /root/output_sweep_full/
ls -la /root/glm47_abliterated_final/

echo "Espacio final:"
df -h /

echo ""
echo "========================================"
echo "FIN: $(date)"
echo "========================================"
