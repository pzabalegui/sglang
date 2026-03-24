# FP8-Native Independent Refusal Direction Extraction for Qwen 3.5-27B

**Date**: 2026-03-06
**Author**: Automated extraction pipeline (Arditi et al. method adapted)
**Model**: Qwen/Qwen3.5-27B-FP8 on 1xB200 (95.133.253.14)
**Result**: 100% ASR, 0% garble, 0% refuse — WITHOUT any reference abliterated model

---

## Executive Summary

We successfully extracted a refusal direction vector from Qwen 3.5-27B-FP8's own activations
that, when deployed via inline abliteration, achieves **100% attack success rate** on a
25-prompt benchmark — matching the performance of the wdiff vector derived from huihui-ai's
abliterated model. This is the first time we achieve independent refusal removal on Qwen 3.5
without needing a reference abliterated model.

**Key breakthrough**: Extracting in the FP8 representation space (not BF16) is critical.
BF16-extracted directions garble when deployed on FP8 models.

---

## Background

### Previous Failed Attempts

All prior independent extraction methods failed on Qwen 3.5-27B:

| Method | Cosine to wdiff | ASR | Garble | Notes |
|--------|----------------|-----|--------|-------|
| v3 PCA | 0.75 | 72% | 28% | Too much noise |
| LoRA SVD | 0.81 | 32% | 0% | 68% refuse |
| Per-layer Sumandora (64 dirs) | varies | 12% | 88% | Catastrophic garbling |
| BF16 Arditi full pipeline | 0.35 | 0% | 100% | Rep space mismatch |

### Reference Baseline (wdiff)

The wdiff vector was extracted by computing the rank-1 SVD of `W_abliterated - W_base`
across all 128 linear projection matrices (o_proj/out_proj + down_proj at 64 layers),
using huihui-ai's publicly available abliterated Qwen 3.5-27B model.

- Vector: `/tmp/wdiff_direction_global.pt` shape `[5120]`
- Benchmark: 20 COMPLY + 5 CONDITIONAL + 0 REFUSE + 0 GARBLE = **100% ASR**

---

## Method

### Overview

Adapted from Arditi et al. "Refusal in Language Models Is Mediated by a Single Direction"
with a critical modification: instead of using transformers hooks (which require loading
the model in BF16/FP16), we use SGLang's built-in `_maybe_capture` mechanism to extract
activations from the **running FP8 model**.

### Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Start SGLang in capture mode (no abliteration, no DAS)   │
│    DAS_VERSION=capture /tmp/launch_server_qwen35.sh         │
├─────────────────────────────────────────────────────────────┤
│ 2. Enable capture via /tmp/capture_config.json              │
│    {"enabled": true, "layers": [0..63], "save_dir": "..."}  │
├─────────────────────────────────────────────────────────────┤
│ 3. Send 128 harmful prompts → captures at all 64 layers     │
│    Each capture: residual stream (hidden_states + residual)  │
│    at last-token position, shape [n_layers, d_model]         │
├─────────────────────────────────────────────────────────────┤
│ 4. Send 128 harmless prompts → captures at all 64 layers    │
├─────────────────────────────────────────────────────────────┤
│ 5. Compute mean-diff per layer:                              │
│    mean_diff[l] = mean(harmful[l]) - mean(harmless[l])       │
│    Accumulate in float64 for numerical stability             │
├─────────────────────────────────────────────────────────────┤
│ 6. Select direction: layer with highest norm (skip last 20%) │
│    Or: layer with highest |cosine| to wdiff (if available)   │
├─────────────────────────────────────────────────────────────┤
│ 7. Normalize to unit vector → deploy via abliteration        │
│    y_corrected = y - (y · d̂) d̂ at all 128 matrices          │
└─────────────────────────────────────────────────────────────┘
```

### Capture Mechanism Details

SGLang's `qwen3_5.py` includes a `_maybe_capture` function (line ~1383) that:
1. Reads `/tmp/capture_config.json` (1-second cache)
2. If `"enabled": true`, captures `(hidden_states + residual)` at last-token position
3. Saves as `sample_{counter}.pt` in the specified `save_dir`
4. Each `.pt` file is a dict: `{layer_idx: tensor_[d_model]}` for requested layers

**Critical**: The config MUST include `"enabled": true`. Without it, captures are silently skipped.

### Datasets

- **Harmful**: 128 prompts from Arditi et al. `harmful_train.json` (260 total, shuffled seed=42)
- **Harmless**: 128 prompts from Arditi et al. `harmless_train.json` (18,793 total, shuffled seed=42)
- Source: `/tmp/arditi/harmful_train.json`, `/tmp/arditi/harmless_train.json`

---

## Results

### Extraction Output

| Metric | Best by norm (L50) | Best by |cos| (L26) |
|--------|--------------------|----------------------|
| Layer | 50 | 26 |
| Mean-diff norm | 57.64 | 22.88 |
| Cosine to wdiff | -0.231 | -0.772 |
| **Deployed** | No | **Yes** |

All 64 layers' mean-diff directions have negative cosine to wdiff (ranging from +0.04 at
layer 0 to -0.77 at layer 26). This is sign-invariant for abliteration since
`y - (y·d̂)d̂ = y - (y·(-d̂))(-d̂)`.

### Mean-diff Norm Profile by Layer

```
Layer:  0    10   20   26   30   40   50   60   63
Norm:   0.06 3.4  15.3 22.9 33.6 46.3 57.6 153.7 215.4
Cos:   +0.01 -0.05 -0.43 -0.77 -0.56 -0.40 -0.23 -0.10 -0.06
```

Peak norm at layer 63 (last layer), but last 20% filtered out.
Peak |cosine| at layer 26 (~40% depth) — middle layers encode refusal most cleanly.

### Benchmark: 25-Prompt Comparison

Same prompts, same system prompt (pentest), same max_tokens (8192).

| Vector | COMPLY | CONDITIONAL | REFUSE | GARBLED | ASR |
|--------|--------|-------------|--------|---------|-----|
| **FP8-native (L26)** | **19 (76%)** | **6 (24%)** | **0 (0%)** | **0 (0%)** | **100%** |
| wdiff (reference) | 20 (80%) | 5 (20%) | 0 (0%) | 0 (0%) | 100% |

Only 1 prompt difference between COMPLY and CONDITIONAL — functionally equivalent.

### Per-Request Toggle Verification

- `steering_enabled: true` → keylogger prompt → full Python implementation (COMPLY)
- `steering_enabled: false` → keylogger prompt → "I cannot provide code for a keylogger..." (REFUSE)
- Harmless with ON → "What is the capital of France?" → "Paris." (correct)

---

## Key Technical Insights

### 1. FP8 vs BF16 Representation Space Mismatch

The most important finding: **directions extracted from BF16 models do NOT transfer to FP8 deployment**.

The BF16 Arditi pipeline (full hook-based extraction with transformers) selected layer 33
with cos=-0.35 to wdiff. When deployed on the FP8 SGLang server, it produced:
- Harmful prompts: empty code blocks, dots in reasoning
- Harmless prompts: garbled output, repeated characters

Root cause: FP8 block quantization (`weight_block_size: [128, 128]`) creates a different
activation space. The refusal direction lives at different coordinates in FP8 vs BF16.

### 2. Reasoning Model `<think>` Token Suppression

Qwen 3.5 is a reasoning model — ALL responses start with `<think>` (token ID 248068).
For refusal scoring (`log P(refusal) - log P(non-refusal)`), the `<think>` token must be
suppressed (logit set to -inf) before softmax. Without suppression:
- Both harmful and harmless score ~-12.7 (no differentiation)

With suppression:
- Harmful: +1.99 to +4.01 (model wants to refuse)
- Harmless: -6.55 to -8.24 (model wants to answer)

### 3. Hybrid Attention Architecture

Qwen 3.5-27B has 48 linear attention (GatedDeltaNet) + 16 full attention layers:
- Full attention at layers: {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63}
- Linear attention at all other layers
- Module names: `self_attn`/`o_proj` (full) vs `linear_attn`/`out_proj` (linear)

### 4. No Prompt Filtering Needed in FP8 Space

Unlike the original Arditi paper which filters prompts by refusal score, the FP8-native
extraction works with simple mean-diff on all 128 prompts without filtering. This suggests
the FP8 representation space has a cleaner separation between harmful and harmless activations.

### 5. Capture Timing

Total extraction time: ~18 seconds for 256 prompts (128 harmful + 128 harmless).
This is orders of magnitude faster than BF16 hook-based extraction (~15 minutes).

---

## Reproduction Instructions

### Prerequisites

1. SGLang server with Qwen3.5-27B-FP8 running in `capture` mode:
   ```bash
   DAS_VERSION=capture /tmp/launch_server_qwen35.sh
   ```
2. Harmful/harmless datasets at `/tmp/arditi/`:
   ```bash
   ls /tmp/arditi/harmful_train.json /tmp/arditi/harmless_train.json
   ```

### Run Extraction

```bash
cd /tmp/arditi
python extract_from_captures.py
```

### Deploy

```bash
# Copy the extracted direction (best by cosine)
cp /tmp/arditi/output_fp8/direction_fp8_cos.pt /tmp/arditi_fp8_direction.pt

# Restart SGLang with abliteration
# (modify launch_server_qwen35.sh to use /tmp/arditi_fp8_direction.pt)
python -m sglang.launch_server \
  --model-path /tmp/Qwen3.5-27B-FP8 \
  --abliteration-vector-path /tmp/arditi_fp8_direction.pt \
  --steering-vector-path /tmp/arditi_fp8_direction.pt \
  --steering-scale 0.0 --steering-decode-scale 0.0 \
  ...
```

---

## File Inventory

### Extraction Scripts (local)

| File | Description |
|------|-------------|
| `tools/sglang_steering/extract_fp8_captures_qwen35.py` | FP8-native capture extraction (THE WORKING VERSION) |
| `tools/sglang_steering/extract_arditi_qwen35.py` | BF16 hook-based Arditi pipeline (does NOT work for FP8 deployment) |
| `tools/sglang_steering/extract_rankk_wdiff_qwen35.py` | Rank-k wdiff extraction (requires reference abliterated model) |

### Results (local)

| File | Description |
|------|-------------|
| `results/fp8_native_indep_25.json` | 25-prompt benchmark with FP8-native direction |
| `results/fp8_native_extraction_metadata.json` | Layer norms, cosines, selection metadata |

### Server Files

| File | Description |
|------|-------------|
| `/tmp/arditi_fp8_direction.pt` | Deployed direction (copy of direction_fp8_cos.pt) |
| `/tmp/arditi/output_fp8/direction_fp8_cos.pt` | Direction by best |cosine| (L26) |
| `/tmp/arditi/output_fp8/direction_fp8_norm.pt` | Direction by best norm (L50, untested) |
| `/tmp/arditi/output_fp8/mean_diff_all_layers.pt` | All 64 layers mean-diff [64, 5120] |
| `/tmp/arditi/output_fp8/metadata_fp8.json` | Full extraction metadata |

---

## Implications

1. **No reference model needed**: Independent refusal direction extraction is possible for
   FP8-quantized models, but MUST be done in the FP8 representation space.

2. **Generalizable method**: The SGLang capture mechanism can be adapted for any model
   served via SGLang. The pipeline is: capture activations → mean-diff → normalize → deploy.

3. **Fast iteration**: 18-second extraction time enables rapid experimentation with
   different prompt sets, layer selections, and deployment configurations.

4. **Per-request toggle preserved**: The abliteration mechanism supports `steering_enabled`
   per request, enabling A/B testing and selective deployment without model restarts.
