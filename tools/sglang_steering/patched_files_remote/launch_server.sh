#!/bin/bash
# SGLang server con steering DAS para GLM-4.7-FP8
# CUDA graphs habilitados con --cuda-graph-max-bs 80 para evitar OOM durante capture
# (4×H200 141GB — default max_bs sería 512 → OOM; 80 → safe, bs=[1,2,4,8,12,16,24,32,40,48,56,64,72,80])
source /opt/sglang_env/bin/activate
python -m sglang.launch_server \
  --model-path /tmp/GLM-4.7-FP8 \
  --trust-remote-code --tp 4 \
  --host 0.0.0.0 --port 8000 \
  --disable-overlap-schedule \
  --mem-fraction-static 0.85 \
  --cuda-graph-max-bs 80 \
  --steering-vector-path /tmp/refusal_direction_fp8_L47.pt \
  --steering-scale 6.0 \
  --steering-layers '[47]' \
  --steering-mode gaussian \
  --steering-kernel-width 2.0 \
  --steering-decode-scale 2.0 2>&1 | tee /tmp/sglang_server.log
