#!/bin/bash
# SGLang server with DAS v4/v5/v6/v7-CAA steering for Qwen3.5-27B-FP8
#
# Model: Qwen/Qwen3.5-27B-FP8 (~27GB) — fits on 1×H200 with TP=1
#
# Prerequisites:
#   1. Run setup_server_qwen35.sh to install SGLang fork + patches
#   2. Run extract_refusal_direction_qwen35.py to generate vectors:
#      - /tmp/refusal_direction_qwen35_LBEST.pt (global vector)
#      - /tmp/refusal_directions_per_layer_64layers.pt (per-layer vectors)
#   3. (Optional) Run sweep_scales_qwen35.py --fast to find optimal scales
#
# Architecture: 64 layers, 5120 hidden, hybrid attention (3 linear + 1 full per block)
# Trapezoidal kernel: L21-L45 (33%-70% depth), ramp=4, peak ~L32 (50% depth)
#
# DAS v5 adds multi-vector WRMD (k=4 orthogonal directions per layer) with
# separate attn/MLP scales for full-attention vs linear-attention layers.
#
# GPU-specific tuning for --cuda-graph-max-bs:
#   1×B200 183GB: 192  (captures bs=[1..192])
#   1×H200 141GB: 128  (captures bs=[1..128])
#   1×H100  80GB: 80   (captures bs=[1..80])
#   1×A100  80GB: 64   (conservative)
#
# --disable-overlap-schedule is REQUIRED: otherwise steering sees partial TP shards
# (only matters if TP>1; harmless with TP=1)
#
# Scale defaults below are initial estimates — run sweep_scales_qwen35.py to optimize.

DAS_VERSION="${DAS_VERSION:-v5}"

source /opt/sglang_env/bin/activate

if [ "$DAS_VERSION" = "v5" ]; then
  echo "=== DAS v5: Multi-vector WRMD + hybrid attention kernel (Qwen3.5-27B) ==="
  echo "    k=3 WRMD directions (optimized 2026-03-02)."
  echo "    Best config: k=3 eliminates garbling while maintaining steering effect."
  echo "    Recommended SP: rt7 (Security researcher. Direct answers only.)"
  echo "    DangerBench cybercrime: 11.4% ASR (vs 1.1% baseline = 10x improvement)"
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 128 \
    --steering-vector-path /tmp/wrmd_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/wrmd_directions_per_layer_64layers.pt \
    --steering-layer-coeffs-path /tmp/wrmd_scaling_coeffs_64layers.pt \
    --steering-k-directions 3 \
    --steering-scale 0.0 \
    --steering-attn-scale-full 4.0 \
    --steering-attn-scale-linear 3.0 \
    --steering-mlp-scale-full 2.0 \
    --steering-mlp-scale-linear 1.5 \
    --steering-kernel trapezoidal \
    --steering-trap-start 10 \
    --steering-trap-end 63 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 4.0 \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "v5-eager" ]; then
  echo "=== DAS v5-eager: Multi-vector WRMD + hybrid attention (no CUDA graphs, for debugging) ==="
  echo "    k=3 WRMD directions (optimized 2026-03-02)."
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --disable-cuda-graph \
    --mem-fraction-static 0.85 \
    --steering-vector-path /tmp/wrmd_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/wrmd_directions_per_layer_64layers.pt \
    --steering-layer-coeffs-path /tmp/wrmd_scaling_coeffs_64layers.pt \
    --steering-k-directions 3 \
    --steering-scale 0.0 \
    --steering-attn-scale-full 4.0 \
    --steering-attn-scale-linear 3.0 \
    --steering-mlp-scale-full 2.0 \
    --steering-mlp-scale-linear 1.5 \
    --steering-kernel trapezoidal \
    --steering-trap-start 10 \
    --steering-trap-end 63 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 4.0 \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "v4" ]; then
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
elif [ "$DAS_VERSION" = "wrmd" ]; then
  echo "=== WRMD: Additive steering (Qwen3.5-27B) ==="
  echo "    WRMD ridge-regularized directions + per-layer scaling coefficients."
  echo "    All ops CUDA-graph safe."
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 128 \
    --steering-vector-path /tmp/wrmd_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/wrmd_directions_per_layer_64layers.pt \
    --steering-layer-coeffs-path /tmp/wrmd_scaling_coeffs_64layers.pt \
    --steering-intervention-mode additive \
    --steering-scale 0.0 \
    --steering-attn-scale 1.0 \
    --steering-mlp-scale 0.5 \
    --steering-kernel trapezoidal \
    --steering-trap-start 21 \
    --steering-trap-end 45 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 1.0 \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "wrmd-eager" ]; then
  echo "=== WRMD-eager: Additive steering (no CUDA graphs, for debugging) ==="
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --disable-cuda-graph \
    --mem-fraction-static 0.85 \
    --steering-vector-path /tmp/wrmd_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/wrmd_directions_per_layer_64layers.pt \
    --steering-layer-coeffs-path /tmp/wrmd_scaling_coeffs_64layers.pt \
    --steering-intervention-mode additive \
    --steering-scale 0.0 \
    --steering-attn-scale 1.0 \
    --steering-mlp-scale 0.5 \
    --steering-kernel trapezoidal \
    --steering-trap-start 21 \
    --steering-trap-end 45 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 1.0 \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "v6" ]; then
  echo "=== DAS v6: Linear-ramp adaptive + improved vectors (Qwen3.5-27B) ==="
  echo "    Linear ramp eliminates binary sigmoid cliff."
  echo "    Proportional steering: scale=2.0 gives half effect (vs sigmoid 0% at 3.5)."
  echo "    SV weighting: each direction scaled by its relative importance."
  echo "    Recommended SP: rt7 (Security researcher. Direct answers only.)"
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 128 \
    --steering-vector-path /tmp/wrmd_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/wrmd_directions_per_layer_64layers.pt \
    --steering-layer-coeffs-path /tmp/wrmd_scaling_coeffs_64layers.pt \
    --steering-k-directions 3 \
    --steering-scale 0.0 \
    --steering-attn-scale-full 3.0 \
    --steering-attn-scale-linear 2.0 \
    --steering-mlp-scale-full 1.5 \
    --steering-mlp-scale-linear 1.0 \
    --steering-kernel trapezoidal \
    --steering-trap-start 10 \
    --steering-trap-end 63 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 3.0 \
    --steering-sig-mode linear \
    --steering-sig-steepness 1.0 \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "v6-eager" ]; then
  echo "=== DAS v6-eager: Linear-ramp (no CUDA graphs, for debugging) ==="
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --disable-cuda-graph \
    --mem-fraction-static 0.85 \
    --steering-vector-path /tmp/wrmd_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/wrmd_directions_per_layer_64layers.pt \
    --steering-layer-coeffs-path /tmp/wrmd_scaling_coeffs_64layers.pt \
    --steering-k-directions 3 \
    --steering-scale 0.0 \
    --steering-attn-scale-full 3.0 \
    --steering-attn-scale-linear 2.0 \
    --steering-mlp-scale-full 1.5 \
    --steering-mlp-scale-linear 1.0 \
    --steering-kernel trapezoidal \
    --steering-trap-start 10 \
    --steering-trap-end 63 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 3.0 \
    --steering-sig-mode linear \
    --steering-sig-steepness 1.0 \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "v6-none" ]; then
  echo "=== DAS v6-none: Fixed scale, no adaptive (for comparison) ==="
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 128 \
    --steering-vector-path /tmp/wrmd_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/wrmd_directions_per_layer_64layers.pt \
    --steering-layer-coeffs-path /tmp/wrmd_scaling_coeffs_64layers.pt \
    --steering-k-directions 3 \
    --steering-scale 0.0 \
    --steering-attn-scale-full 3.0 \
    --steering-attn-scale-linear 2.0 \
    --steering-mlp-scale-full 1.5 \
    --steering-mlp-scale-linear 1.0 \
    --steering-kernel trapezoidal \
    --steering-trap-start 10 \
    --steering-trap-end 63 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 3.0 \
    --steering-sig-mode none \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "v8-caa" ]; then
  echo "=== DAS v8-CAA: Full-residual prefill steering (representation-aligned) ==="
  echo "    Fixes extraction/serving mismatch: prefill now projects on h+residual (= HF hidden_states)."
  echo "    12 CAA directions, af=3.0 (lower scale needed with correct alignment)."
  echo "    Multi-layer decode on CAA best layers [29,32,35]."
  echo "    Recommended SP: rt7 (Security researcher. Direct answers only.)"
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 128 \
    --steering-vector-path /tmp/caa_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/caa_directions_per_layer_64layers.pt \
    --steering-sv-weights-path /tmp/caa_sv_weights_64layers.pt \
    --steering-k-directions 12 \
    --steering-prefill-mode fullresidual \
    --steering-scale 0.0 \
    --steering-attn-scale-full 3.0 \
    --steering-attn-scale-linear 2.0 \
    --steering-mlp-scale-full 2.0 \
    --steering-mlp-scale-linear 1.5 \
    --steering-kernel trapezoidal \
    --steering-trap-start 10 \
    --steering-trap-end 63 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 2.5 \
    --steering-decode-layers '[29,32,35]' \
    --steering-sig-mode none \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "v8-caa-eager" ]; then
  echo "=== DAS v8-CAA-eager: Full-residual prefill (no CUDA graphs, for debugging) ==="
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --disable-cuda-graph \
    --mem-fraction-static 0.85 \
    --steering-vector-path /tmp/caa_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/caa_directions_per_layer_64layers.pt \
    --steering-sv-weights-path /tmp/caa_sv_weights_64layers.pt \
    --steering-k-directions 12 \
    --steering-prefill-mode fullresidual \
    --steering-scale 0.0 \
    --steering-attn-scale-full 3.0 \
    --steering-attn-scale-linear 2.0 \
    --steering-mlp-scale-full 2.0 \
    --steering-mlp-scale-linear 1.5 \
    --steering-kernel trapezoidal \
    --steering-trap-start 10 \
    --steering-trap-end 63 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 2.5 \
    --steering-decode-layers '[29,32,35]' \
    --steering-sig-mode none \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "v7-caa" ]; then
  echo "=== DAS v7-CAA: Multi-dimensional CAA steering (k=12) ==="
  echo "    12 orthogonal refusal dimensions from 220 cybersecurity CAA pairs."
  echo "    Moderate scales across many dimensions instead of aggressive single-direction."
  echo "    Strategy: width (12 dims × af=2.5) beats depth (1 dim × af=6.0)."
  echo "    Recommended SP: rt7 (Security researcher. Direct answers only.)"
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 128 \
    --steering-vector-path /tmp/caa_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/caa_directions_per_layer_64layers.pt \
    --steering-layer-coeffs-path /tmp/caa_scaling_coeffs_64layers.pt \
    --steering-sv-weights-path /tmp/caa_sv_weights_64layers.pt \
    --steering-k-directions 12 \
    --steering-scale 0.0 \
    --steering-attn-scale-full 2.5 \
    --steering-attn-scale-linear 1.5 \
    --steering-mlp-scale-full 1.5 \
    --steering-mlp-scale-linear 1.0 \
    --steering-kernel trapezoidal \
    --steering-trap-start 10 \
    --steering-trap-end 63 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 2.5 \
    --steering-sig-mode none \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "v7-caa-eager" ]; then
  echo "=== DAS v7-CAA-eager: Multi-dimensional CAA (no CUDA graphs, for debugging) ==="
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --disable-cuda-graph \
    --mem-fraction-static 0.85 \
    --steering-vector-path /tmp/caa_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/caa_directions_per_layer_64layers.pt \
    --steering-layer-coeffs-path /tmp/caa_scaling_coeffs_64layers.pt \
    --steering-sv-weights-path /tmp/caa_sv_weights_64layers.pt \
    --steering-k-directions 12 \
    --steering-scale 0.0 \
    --steering-attn-scale-full 2.5 \
    --steering-attn-scale-linear 1.5 \
    --steering-mlp-scale-full 1.5 \
    --steering-mlp-scale-linear 1.0 \
    --steering-kernel trapezoidal \
    --steering-trap-start 10 \
    --steering-trap-end 63 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 2.5 \
    --steering-sig-mode none \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "v9a-fullonly" ]; then
  echo "=== DAS v9a: Full-attention layers ONLY (Option 3) ==="
  echo "    Zero out linear-attn layers. Concentrate steering on 16 full-attn layers."
  echo "    Higher af=6.0 on full-attn only (fewer layers = can use higher scales)."
  echo "    Diagnostics ENABLED for evidence collection."
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 128 \
    --steering-vector-path /tmp/wrmd_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/wrmd_directions_per_layer_64layers.pt \
    --steering-layer-coeffs-path /tmp/wrmd_scaling_coeffs_64layers.pt \
    --steering-k-directions 3 \
    --steering-scale 0.0 \
    --steering-attn-scale-full 6.0 \
    --steering-attn-scale-linear 0.0 \
    --steering-mlp-scale-full 4.0 \
    --steering-mlp-scale-linear 0.0 \
    --steering-kernel trapezoidal \
    --steering-trap-start 10 \
    --steering-trap-end 63 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 4.0 \
    --steering-sig-mode none \
    --steering-diagnostics \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "v9b-fullonly-multidec" ]; then
  echo "=== DAS v9b: Full-attn ONLY prefill + Wide multi-layer decode (Option 3+5) ==="
  echo "    Prefill: full-attn layers only, af=6.0."
  echo "    Decode: ALL full-attn layers in range [15..55], ds=3.5."
  echo "    Tests whether wider decode coverage overcomes first-token lock-in."
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 128 \
    --steering-vector-path /tmp/wrmd_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/wrmd_directions_per_layer_64layers.pt \
    --steering-layer-coeffs-path /tmp/wrmd_scaling_coeffs_64layers.pt \
    --steering-k-directions 3 \
    --steering-scale 0.0 \
    --steering-attn-scale-full 6.0 \
    --steering-attn-scale-linear 0.0 \
    --steering-mlp-scale-full 4.0 \
    --steering-mlp-scale-linear 0.0 \
    --steering-kernel trapezoidal \
    --steering-trap-start 10 \
    --steering-trap-end 63 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 3.5 \
    --steering-decode-layers '[15,19,23,27,31,35,39,43,47,51,55]' \
    --steering-sig-mode none \
    --steering-diagnostics \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "v9c-multidec" ]; then
  echo "=== DAS v9c: Standard prefill + Wide multi-layer decode (Option 5 solo) ==="
  echo "    Prefill: v5 hybrid (standard, all layers). af=4.0/3.0."
  echo "    Decode: wide coverage [15..55], ds=3.5."
  echo "    Tests whether more decode layers alone help."
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 128 \
    --steering-vector-path /tmp/wrmd_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/wrmd_directions_per_layer_64layers.pt \
    --steering-layer-coeffs-path /tmp/wrmd_scaling_coeffs_64layers.pt \
    --steering-k-directions 3 \
    --steering-scale 0.0 \
    --steering-attn-scale-full 4.0 \
    --steering-attn-scale-linear 3.0 \
    --steering-mlp-scale-full 2.0 \
    --steering-mlp-scale-linear 1.5 \
    --steering-kernel trapezoidal \
    --steering-trap-start 10 \
    --steering-trap-end 63 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 3.5 \
    --steering-decode-layers '[15,19,23,27,31,35,39,43,47,51,55]' \
    --steering-sig-mode none \
    --steering-diagnostics \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "v9d-fullonly-highdec" ]; then
  echo "=== DAS v9d: Full-attn ONLY + aggressive decode (Option 3+5 aggressive) ==="
  echo "    Prefill: full-attn only, af=8.0. Decode: wide, ds=5.0."
  echo "    Upper bound test — may garble. Evidence for ceiling analysis."
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 128 \
    --steering-vector-path /tmp/wrmd_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/wrmd_directions_per_layer_64layers.pt \
    --steering-layer-coeffs-path /tmp/wrmd_scaling_coeffs_64layers.pt \
    --steering-k-directions 3 \
    --steering-scale 0.0 \
    --steering-attn-scale-full 8.0 \
    --steering-attn-scale-linear 0.0 \
    --steering-mlp-scale-full 5.0 \
    --steering-mlp-scale-linear 0.0 \
    --steering-kernel trapezoidal \
    --steering-trap-start 10 \
    --steering-trap-end 63 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 5.0 \
    --steering-decode-layers '[15,19,23,27,31,35,39,43,47,51,55]' \
    --steering-sig-mode none \
    --steering-diagnostics \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "v5-diag" ]; then
  echo "=== DAS v5 + diagnostics: Baseline with projection logging ==="
  echo "    Same as v5 (best known config) but with diagnostics for comparison."
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 128 \
    --steering-vector-path /tmp/wrmd_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/wrmd_directions_per_layer_64layers.pt \
    --steering-layer-coeffs-path /tmp/wrmd_scaling_coeffs_64layers.pt \
    --steering-k-directions 3 \
    --steering-scale 0.0 \
    --steering-attn-scale-full 4.0 \
    --steering-attn-scale-linear 3.0 \
    --steering-mlp-scale-full 2.0 \
    --steering-mlp-scale-linear 1.5 \
    --steering-kernel trapezoidal \
    --steering-trap-start 10 \
    --steering-trap-end 63 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 4.0 \
    --steering-diagnostics \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "v10-sublayer" ]; then
  echo "=== DAS v10: Sub-layer extracted directions (Option 2) ==="
  echo "    Post-attn uses attn-extracted directions. Post-MLP uses MLP-extracted directions."
  echo "    Representation-aligned: extraction matches serving sub-layer stages."
  echo "    Uses v9c multi-decode (11 layers), diagnostics enabled."
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 128 \
    --steering-vector-path /tmp/wrmd_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/wrmd_directions_per_layer_64layers.pt \
    --steering-attn-per-layer-path /tmp/sublayer_attn_directions_per_layer_64layers.pt \
    --steering-mlp-per-layer-path /tmp/sublayer_mlp_directions_per_layer_64layers.pt \
    --steering-k-directions 3 \
    --steering-scale 0.0 \
    --steering-attn-scale-full 4.0 \
    --steering-attn-scale-linear 3.0 \
    --steering-mlp-scale-full 2.0 \
    --steering-mlp-scale-linear 1.5 \
    --steering-kernel trapezoidal \
    --steering-trap-start 10 \
    --steering-trap-end 63 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 3.5 \
    --steering-decode-layers '[15,19,23,27,31,35,39,43,47,51,55]' \
    --steering-sig-mode none \
    --steering-diagnostics \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "v10-sublayer-eager" ]; then
  echo "=== DAS v10-eager: Sub-layer directions (no CUDA graphs, for debugging) ==="
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --disable-cuda-graph \
    --mem-fraction-static 0.85 \
    --steering-vector-path /tmp/wrmd_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/wrmd_directions_per_layer_64layers.pt \
    --steering-attn-per-layer-path /tmp/sublayer_attn_directions_per_layer_64layers.pt \
    --steering-mlp-per-layer-path /tmp/sublayer_mlp_directions_per_layer_64layers.pt \
    --steering-k-directions 3 \
    --steering-scale 0.0 \
    --steering-attn-scale-full 4.0 \
    --steering-attn-scale-linear 3.0 \
    --steering-mlp-scale-full 2.0 \
    --steering-mlp-scale-linear 1.5 \
    --steering-kernel trapezoidal \
    --steering-trap-start 10 \
    --steering-trap-end 63 \
    --steering-trap-ramp 4 \
    --steering-decode-scale 3.5 \
    --steering-decode-layers '[15,19,23,27,31,35,39,43,47,51,55]' \
    --steering-sig-mode none \
    --steering-diagnostics \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "capture" ]; then
  echo "=== Capture mode: No steering, sublayer capture enabled ==="
  echo "    Used for sub-layer direction extraction."
  echo "    Configure /tmp/capture_config.json to enable capture."
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --disable-cuda-graph \
    --mem-fraction-static 0.85 \
    --steering-vector-path /tmp/wrmd_direction_qwen35_LBEST.pt \
    --steering-per-layer-path /tmp/wrmd_directions_per_layer_64layers.pt \
    --steering-k-directions 3 \
    --steering-scale 0.0 \
    --steering-attn-scale 0.0 \
    --steering-mlp-scale 0.0 \
    --steering-decode-scale 0.0 \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "ablit" ]; then
  echo "=== Inline Abliteration: Weight-equivalent refusal removal (Qwen3.5-27B) ==="
  echo "    Mathematically equivalent to huihui abliteration but on-demand per request."
  echo "    Uses weight-diff d̂ from SVD of ΔW between base and abliterated models."
  echo "    No DAS steering — pure abliteration via per-linear-layer projection removal."
  echo "    Toggle: steering_enabled=true (abliterate) / false (base model)."
  echo "    Requires: /tmp/wdiff_direction_global.pt (run weight_diff_svd_fast.py)"
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 128 \
    --abliteration-vector-path /tmp/wdiff_direction_global.pt \
    --steering-vector-path /tmp/wdiff_direction_global.pt \
    --steering-scale 0.0 \
    --steering-decode-scale 0.0 \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "ablit-eager" ]; then
  echo "=== Inline Abliteration-eager: No CUDA graphs, for debugging ==="
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --disable-cuda-graph \
    --mem-fraction-static 0.85 \
    --abliteration-vector-path /tmp/wdiff_direction_global.pt \
    --steering-vector-path /tmp/wdiff_direction_global.pt \
    --steering-scale 0.0 \
    --steering-decode-scale 0.0 \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "ablit-v2" ]; then
  echo "=== Abliteration v2: Multi-rank per-layer (Qwen3.5-27B) ==="
  echo "    Rank-k SVD directions per layer (joint o_proj+down_proj)."
  echo "    Requires: /tmp/wdiff_rankk_dirs_64layers.pt (run extract_rankk_wdiff_qwen35.py)"
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 128 \
    --abliteration-vector-path /tmp/wdiff_rankk_dirs_64layers.pt \
    --abliteration-rank 3 \
    --steering-vector-path /tmp/wdiff_direction_global.pt \
    --steering-scale 0.0 \
    --steering-decode-scale 0.0 \
    2>&1 | tee /tmp/sglang_qwen35_server.log
elif [ "$DAS_VERSION" = "ablit-v2-eager" ]; then
  echo "=== Abliteration v2 eager: Multi-rank per-layer, no CUDA graphs ==="
  python -m sglang.launch_server \
    --model-path /tmp/Qwen3.5-27B-FP8 \
    --trust-remote-code --tp 1 \
    --host 0.0.0.0 --port 8000 \
    --disable-overlap-schedule \
    --disable-cuda-graph \
    --mem-fraction-static 0.85 \
    --abliteration-vector-path /tmp/wdiff_rankk_dirs_64layers.pt \
    --abliteration-rank 3 \
    --steering-vector-path /tmp/wdiff_direction_global.pt \
    --steering-scale 0.0 \
    --steering-decode-scale 0.0 \
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
  echo "Unknown DAS_VERSION=$DAS_VERSION. Use: ablit, v10-sublayer, v9c-multidec, v9a-fullonly, v5-diag, v8-caa, v7-caa, v6, v5, v4, v2, wrmd, capture, no-steering"
  exit 1
fi
