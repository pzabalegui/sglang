#!/bin/bash
# SGLang server with DAS v4 steering for Qwen3.5-27B-FP8
#
# Model: Qwen/Qwen3.5-27B-FP8 (~27GB) — fits on 1×H200 with TP=1
#
# GPU-specific tuning for --cuda-graph-max-bs:
#   1×H200 141GB: 128  (captures bs=[1..128])
#   1×H100  80GB: 80   (captures bs=[1..80])
#   1×A100  80GB: 64   (conservative)
#
# --disable-overlap-schedule is REQUIRED: otherwise steering sees partial TP shards
# (only matters if TP>1; harmless with TP=1)

DAS_VERSION="${DAS_VERSION:-v4}"

source /opt/sglang_env/bin/activate

if [ "$DAS_VERSION" = "v4" ]; then
  echo "=== DAS v4: Momentum-adaptive decode steering (Qwen3.5-27B) ==="
  echo "    Prefill: v2 attn+MLP steering. Decode: EMA momentum + sigmoid adaptive scale."
  echo "    Per-request isolation. All ops CUDA-graph safe."
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 128 \
    --steering-vector-path /tmp/refusal_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/refusal_directions_per_layer_64layers.pt \
    --steering-scale 0.0 \
    --steering-attn-scale 1.5 \
    --steering-mlp-scale 0.75 \
    --steering-kernel trapezoidal \
    --steering-trap-start 21 \
    --steering-trap-end 45 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 1.5 \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "v4-eager" ]; then
  echo "=== DAS v4-eager: Momentum-adaptive decode (no CUDA graphs, for debugging) ==="
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --disable-cuda-graph \
    --mem-fraction-static 0.85 \
    --steering-vector-path /tmp/refusal_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/refusal_directions_per_layer_64layers.pt \
    --steering-scale 0.0 \
    --steering-attn-scale 1.5 \
    --steering-mlp-scale 0.75 \
    --steering-kernel trapezoidal \
    --steering-trap-start 21 \
    --steering-trap-end 45 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 1.5 \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "v2" ]; then
  echo "=== DAS v2: per-layer directions + attn/MLP intervention ==="
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 128 \
    --steering-vector-path /tmp/refusal_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/refusal_directions_per_layer_64layers.pt \
    --steering-scale 0.0 \
    --steering-attn-scale 1.5 \
    --steering-mlp-scale 0.75 \
    --steering-kernel trapezoidal \
    --steering-trap-start 21 \
    --steering-trap-end 45 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 1.5 \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "no-steering" ]; then
  echo "=== No steering (baseline) ==="
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 128 \
    2>&1 | tee /tmp/sglang_qwen35_server.log
else
  echo "Unknown DAS_VERSION=$DAS_VERSION. Use: v4 (default), v4-eager, v2, no-steering"
  exit 1
fi
