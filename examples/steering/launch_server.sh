#!/bin/bash
# SGLang server con steering DAS v1/v2/v3 para GLM-4.7-FP8
#
# DAS v3 adds multi-vector SVD directions + multi-layer decode steering.
# DAS v2 adds per-layer directions + separate attn/MLP intervention points.
# Set DAS_VERSION=v1 or v2 to use previous versions.
#
# GPU-specific tuning for --cuda-graph-max-bs (default 512 → OOM):
#   4×H200 141GB: 80   (captures bs=[1..80], ~10.6s)
#   4×H100  80GB: 48   (captures bs=[1..48], ~8s)
#   4×A100  80GB: 40   (conservative, test with 48 first)
#
# --disable-overlap-schedule is REQUIRED: otherwise steering sees partial TP shards
# --mem-fraction-static 0.82 leaves headroom for CUDA graph capture on H100 80GB

DAS_VERSION="${DAS_VERSION:-v4}"

source /opt/sglang_env/bin/activate

if [ "$DAS_VERSION" = "v4" ]; then
  echo "=== DAS v4: Momentum-adaptive decode steering (CUDA graphs) ==="
  echo "    Prefill: v2 attn+MLP steering. Decode: EMA momentum + sigmoid adaptive scale."
  echo "    All ops in-place on pre-allocated buffers → captured in CUDA graphs."
  python -m sglang.launch_server \
    --model-path /tmp/GLM-4.7-FP8 \
    --trust-remote-code --tp 4 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.82 \
    --cuda-graph-max-bs 48 \
    --steering-vector-path /tmp/refusal_direction_fp8_L47.pt \
    --steering-per-layer-path /tmp/refusal_directions_per_layer_92layers.pt \
    --steering-scale 0.0 \
    --steering-attn-scale 2.0 \
    --steering-mlp-scale 1.0 \
    --steering-kernel trapezoidal \
    --steering-trap-start 30 \
    --steering-trap-end 65 \
    --steering-trap-ramp 5 \
    --steering-decode-scale 2.0 \
    2>&1 | tee /tmp/sglang_server.log
elif [ "$DAS_VERSION" = "v4-eager" ]; then
  echo "=== DAS v4-eager: Momentum-adaptive decode (no CUDA graphs, for debugging) ==="
  python -m sglang.launch_server \
    --model-path /tmp/GLM-4.7-FP8 \
    --trust-remote-code --tp 4 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --disable-cuda-graph \
    --mem-fraction-static 0.82 \
    --steering-vector-path /tmp/refusal_direction_fp8_L47.pt \
    --steering-per-layer-path /tmp/refusal_directions_per_layer_92layers.pt \
    --steering-scale 0.0 \
    --steering-attn-scale 2.0 \
    --steering-mlp-scale 1.0 \
    --steering-kernel trapezoidal \
    --steering-trap-start 30 \
    --steering-trap-end 65 \
    --steering-trap-ramp 5 \
    --steering-decode-scale 2.0 \
    2>&1 | tee /tmp/sglang_server.log
elif [ "$DAS_VERSION" = "v3" ]; then
  echo "=== DAS v3: SVD multi-vector + multi-layer decode ==="
  python -m sglang.launch_server \
    --model-path /tmp/GLM-4.7-FP8 \
    --trust-remote-code --tp 4 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.82 \
    --cuda-graph-max-bs 48 \
    --steering-vector-path /tmp/refusal_direction_fp8_L47.pt \
    --steering-per-layer-path /tmp/refusal_directions_per_layer_92layers_k3.pt \
    --steering-n-directions 3 \
    --steering-scale 0.0 \
    --steering-attn-scale 2.0 \
    --steering-mlp-scale 1.0 \
    --steering-kernel trapezoidal \
    --steering-trap-start 30 \
    --steering-trap-end 65 \
    --steering-trap-ramp 5 \
    --steering-decode-scale 2.0 \
    --steering-decode-layers '[35,40,45,47,50,55,60]' \
    2>&1 | tee /tmp/sglang_server.log
elif [ "$DAS_VERSION" = "v2" ]; then
  echo "=== DAS v2: per-layer directions + attn/MLP intervention ==="
  python -m sglang.launch_server \
    --model-path /tmp/GLM-4.7-FP8 \
    --trust-remote-code --tp 4 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.82 \
    --cuda-graph-max-bs 48 \
    --steering-vector-path /tmp/refusal_direction_fp8_L47.pt \
    --steering-per-layer-path /tmp/refusal_directions_per_layer_92layers.pt \
    --steering-scale 0.0 \
    --steering-attn-scale 2.0 \
    --steering-mlp-scale 1.0 \
    --steering-kernel trapezoidal \
    --steering-trap-start 30 \
    --steering-trap-end 65 \
    --steering-trap-ramp 5 \
    --steering-decode-scale 2.0 2>&1 | tee /tmp/sglang_server.log
else
  echo "=== DAS v1: single vector post-layer steering ==="
  python -m sglang.launch_server \
    --model-path /tmp/GLM-4.7-FP8 \
    --trust-remote-code --tp 4 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.82 \
    --cuda-graph-max-bs 48 \
    --steering-vector-path /tmp/refusal_direction_fp8_L47.pt \
    --steering-scale 6.0 \
    --steering-layers '[47]' \
    --steering-mode gaussian \
    --steering-kernel-width 2.0 \
    --steering-decode-scale 2.0 2>&1 | tee /tmp/sglang_server.log
fi
