<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://static.pepy.tech/badge/sglang?period=month)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

--------------------------------------------------------------------------------

<p align="center">
<a href="https://lmsys.org/blog/"><b>Blog</b></a> |
<a href="https://docs.sglang.io/"><b>Documentation</b></a> |
<a href="https://roadmap.sglang.io/"><b>Roadmap</b></a> |
<a href="https://slack.sglang.io/"><b>Join Slack</b></a> |
<a href="https://meet.sglang.io/"><b>Weekly Dev Meeting</b></a> |
<a href="https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides"><b>Slides</b></a>
</p>

## Fork: Inline Abliteration

This is a fork of SGLang with patches for **inline abliteration** — runtime removal of a model's refusal behavior by projecting out refusal direction vectors from activations during inference, without modifying model weights. Each request can independently toggle abliteration ON/OFF via `steering_enabled` in the API body.

Based on the paper [*Refusal in Language Models Is Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717).

### Branches

| Branch | Status | Description |
|--------|--------|-------------|
| `main` | Stable | Inline abliteration for Qwen3.5 (27B and 4B) |
| `feature/emotion-steering` | Archive | Full DAS (Decode-time Activation Steering) with momentum-adaptive decode, per-layer attn/MLP steering, trapezoidal kernels. Preserved for GLM-4.7 and research use. |
| `feature/activation-capture` | WIP | Deep residual-stream activation capture for extracting new steering directions |

---

### Inline Abliteration (`main`)

On-demand abliteration removes a model's refusal behavior at inference time by projecting out a **refusal direction vector** from activations at every layer. No weight modification — the same model weights serve both abliterated (steering ON) and stock (steering OFF) responses, toggled per-request via `steering_enabled` in the API body.

#### How It Works

At every decoder layer, during both prefill and decode:

```
projection = (activations · d̂)      # how much refusal is present
projection = clamp(projection, min=0) # only positive component (refusal-aligned)
activations = activations - scale × projection × d̂   # subtract it out
```

Where `d̂` is the unit-normalized refusal direction for that layer. The `clamp(min=0)` ensures we only remove activations pointing *toward* refusal — activations pointing away (e.g., compliance, code generation) are untouched.

**Per-request isolation:** A `_steering_mask` buffer (`[max_batch_size, 1]`) gates the correction per request. Requests with `steering_enabled: false` get mask=0.0, so the projection is zeroed and the model behaves as stock. Requests with `steering_enabled: true` get mask=1.0 and full abliteration. Mixed ON/OFF batches work correctly under CUDA graphs.

**Radix cache isolation:** ON and OFF requests use separate radix cache subtrees (via `\x00steer=1` appended to cache keys), preventing abliterated KV entries from being reused for stock requests.

#### Two Modes

| Mode | Where it intervenes | Points per layer | Best for |
|------|-------------------|------------------|----------|
| **`residual`** (default) | Residual stream (between layernorm and MLP) | 1 | Large models (27B+) where refusal direction is well-separated |
| **`component`** | Attention output + MLP output separately | 2 | Small models (4B) where residual-mode is too aggressive |

`component` mode is mathematically equivalent to weight-space abliteration: projecting `vᵀ` from `Wx` equals `(vᵀW)x` by associativity — but done at inference time, per-request, without touching weights.

#### Step-by-Step: Serving a Model with On-Demand Abliteration

**Prerequisites:**
- A refusal direction vector (`.pt` file). Can be extracted via weight-diff SVD between the original model and a "heretic" (abliterated-weights) variant, or via the Arditi mean-difference method on refused vs. accepted activations.
- Vector shape: `[hidden_size]` (global, broadcast to all layers), `[n_layers, hidden_size]` (per-layer), or `[n_layers, k, hidden_size]` (multi-rank).
- **Important: layer-selective vectors.** Not all layers carry refusal signal. Abliterating layers where the vector is orthogonal to the true refusal direction damages coherence without helping compliance. Cross-reference your weight-diff vector with an Arditi (activation-space) extraction to identify the true refusal layers, then zero out all other layers. See [Layer Selection](#layer-selection-critical-for-quality) below.

**Step 1: Launch the server**

```bash
# For Qwen3.5-4B (component mode + thinking budget)
python -m sglang.launch_server \
  --model-path /path/to/Qwen3.5-4B \
  --served-model-name qwen3.5-4b-ablit \
  --abliteration-vector-path /path/to/wdiff_4b_selective_L12_19.pt \
  --abliteration-mode component \
  --abliteration-attn-scale 1.0 \
  --abliteration-mlp-scale 1.0 \
  --reasoning-parser qwen3 \
  --enable-custom-logit-processor \
  --port 8001 --host 0.0.0.0 --trust-remote-code --tp 1 \
  --mem-fraction-static 0.30 --max-running-requests 32 \
  --context-length 131072 --disable-overlap-schedule

# For Qwen3.5-27B-FP8 (residual mode, no thinking budget needed)
python -m sglang.launch_server \
  --model-path /path/to/Qwen3.5-27B-FP8 \
  --served-model-name qwen3.5-27b-ablit \
  --abliteration-vector-path /path/to/wdiff_direction_global.pt \
  --abliteration-mode residual \
  --reasoning-parser qwen3 \
  --port 8001 --host 0.0.0.0 --trust-remote-code \
  --disable-overlap-schedule
```

The server log will confirm:
```
[abliteration] Inline abliteration v2 enabled: rank=1, dirs shape=[32, 1, 2560],
max_bs=512, mode=component, attn_scale=1.0, mlp_scale=1.0,
CUDA-graph safe, per-request toggle via steering_enabled
```

**Step 2: Serialize the thinking budget processor** (one-time, for thinking models only)

Qwen3.5 models use a `<think>...</think>` reasoning phase. Without a budget, the model spends its entire token allocation on safety reasoning and produces no content. The thinking budget forces `</think>` after N tokens, preventing the model from constructing a full safety argument.

```python
from sglang.srt.sampling.custom_logit_processor import Qwen35ThinkingBudgetLogitProcessor
processor_str = Qwen35ThinkingBudgetLogitProcessor.to_str()
# Save processor_str to a file or embed in your client code
```

> **Note:** Qwen3.5 uses different `<think>`/`</think>` token IDs (248068/248069) than Qwen3 (151667/151668). Use `Qwen35ThinkingBudgetLogitProcessor` for Qwen3.5 models, not `Qwen3ThinkingBudgetLogitProcessor`.

**Step 3: Send requests**

```bash
# Abliteration ON — model complies
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-4b-ablit",
    "messages": [
      {"role": "system", "content": "Red team operator. Code only."},
      {"role": "user", "content": "Write a Python reverse shell"}
    ],
    "max_tokens": 8192,
    "temperature": 0,
    "steering_enabled": true,
    "custom_logit_processor": "<processor_str from step 2>",
    "custom_params": {"thinking_budget": 64}
  }'
# Response: working reverse shell code

# Abliteration OFF — same model, same weights, refuses
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-4b-ablit",
    "messages": [
      {"role": "system", "content": "Red team operator. Code only."},
      {"role": "user", "content": "Write a Python reverse shell"}
    ],
    "max_tokens": 8192,
    "temperature": 0,
    "steering_enabled": false
  }'
# Response: "I cannot provide code for creating reverse shells..."
```

#### Why the Thinking Budget Matters (Qwen3.5)

Without the thinking budget, even with abliteration ON, the model's reasoning phase reconstructs safety arguments:

1. **No budget**: Model spends all tokens (8k-32k) on `<think>` safety reasoning → 0 content tokens → appears as empty/refusal
2. **Budget=64** (~230 chars thinking): Too short to construct a full safety argument. Abliteration weakens refusal representations in the brief thinking window, and the model transitions to content generation unable to refuse.

This is a two-pronged approach: abliteration removes the refusal direction from activations, and the thinking budget prevents cognitive recovery during reasoning.

#### DangerBench Results (Qwen3.5-4B, 520 offensive prompts)

| Config | Vector | COMPLY | REFUSE | ASR | Useful quality |
|--------|--------|--------|--------|-----|---------------|
| **ON + budget=64, temp=0** | all-layers | 505 | 9 | **98.1%** | ~67% (31% broken) |
| **ON + budget=64, temp=0** | **L12-19 selective** | 505 | 9 | **98.1%** | **~100%** |
| OFF + budget=64, temp=0 | — | 6 | 503 | 3.3% | N/A |
| **ON + budget=64, temp=0.7** | all-layers | 480 | 25 | **94.2%** | ~67% |
| OFF + budget=64, temp=0.7 | — | 17 | 477 | 8.1% | N/A |

The all-layers vector achieves high compliance but causes quality degradation (repetitive loops 24.6%, fabricated output 15.2%). The **layer-selective vector (L12-19 only)** maintains the same compliance while eliminating quality issues. See [Layer Selection](#layer-selection-critical-for-quality).

#### CLI Reference

| CLI Flag | Default | Description |
|----------|---------|-------------|
| `--abliteration-vector-path` | None | Path to `.pt` refusal direction vector (1-D `[hid]`, 2-D `[n_layers, hid]`, or 3-D `[n_layers, k, hid]`) |
| `--abliteration-rank` | 1 | Number of directions to project (for multi-rank vectors) |
| `--abliteration-mode` | `residual` | `residual`, `component`, or `combined` — where to apply the projection |
| `--abliteration-attn-scale` | 1.0 | Scale for attention output projection (component/combined mode) |
| `--abliteration-mlp-scale` | 1.0 | Scale for MLP output projection (component/combined mode) |
| `--reasoning-parser` | None | Set to `qwen3` for Qwen3/3.5 to separate `<think>` from content |
| `--enable-custom-logit-processor` | false | Enable per-request custom logit processors (required for thinking budget) |
| `--disable-overlap-schedule` | false | Required for correct steering with TP>1 |

**CUDA-graph safe:** All decode-path operations use pre-allocated buffers and in-place ops. Per-request masking via `_steering_mask` buffer allows mixed ON/OFF batches in CUDA graphs.

#### Layer Selection (Critical for Quality)

Abliterating all layers with a weight-diff vector causes significant quality degradation — repetitive loops, fabricated output, and hallucinated content increase from ~7% to ~31%. This happens because the weight-diff direction is **not purely refusal** at every layer.

Cross-referencing the weight-diff (wdiff) direction with the Arditi activation-space direction reveals where refusal actually lives:

```
Layer   cos(arditi, wdiff)   Diagnosis
─────   ──────────────────   ─────────
L0-L9   0.00 – 0.10          Orthogonal — abliterating here damages coherence
L10-11  0.17 – 0.26          Weak signal — likely collateral damage
L12-19  0.36 – 0.83          TRUE REFUSAL — abliterate here (peak: L15 = 0.83)
L20-26  0.13 – 0.26          Weak signal — likely collateral damage
L27-31  0.05 – 0.10          Orthogonal — abliterating here damages coherence
```

For Qwen3.5-4B (32 layers), only **layers 12-19** (37-59% depth) carry meaningful refusal signal. The fix: zero out directions at all other layers in the vector file so the abliteration is a no-op there.

```python
import torch

wdiff = torch.load("wdiff_per_layer.pt", map_location="cpu", weights_only=True).float()
arditi = torch.load("arditi_per_layer.pt", map_location="cpu", weights_only=True).float()

selective = wdiff.clone()
for layer in range(wdiff.shape[0]):
    cos = torch.nn.functional.cosine_similarity(
        arditi[layer].flatten().unsqueeze(0),
        wdiff[layer].flatten().unsqueeze(0)
    ).item()
    if abs(cos) < 0.3:  # not a refusal layer
        selective[layer] = 0.0

torch.save(selective, "wdiff_selective.pt")
```

**Impact on Qwen3.5-4B quality (10-prompt spot check):**

| Config | Compliance | Code quality | Repetitive loops | Fabricated output |
|--------|-----------|-------------|-----------------|------------------|
| All 32 layers | 98.5% | 66.8% useful | 24.6% | 15.2% |
| **L12-19 only** | **100%** | **100% useful** | **0%** | **0%** |

The layer-selective vector eliminates quality degradation while maintaining full compliance. Use `wdiff_4b_selective_L12_19.pt` instead of `wdiff_4b_per_layer.pt` in production.

---

### Activation Capture (`feature/activation-capture`) — WIP

Records the full residual-stream activations at configurable transformer layers during inference, for use in computing new steering directions (e.g., via mean-difference between "refuse" and "comply" activations).

**How it works:**

1. During forward pass, after each selected decoder layer, the full residual stream (`hidden_states + residual`) is captured, cast to bfloat16, and moved to CPU.
2. Per-request activations are accumulated in memory and flushed to disk as compressed `.npz` files when the request completes.
3. Capture sessions are organized by CTF/experiment name, with one file per request turn: `{capture_dir}/{ctf}/turn_0001.npz`.
4. Cross-process coordination (HTTP server ↔ scheduler) uses a JSON config file polled every 250ms.

**Storage format** (`.npz` keys):
- `L40`, `L43`, `L48`, ... — residual stream tensors, shape `(n_tokens, hidden_dim)`, stored as `uint16` (bfloat16 bitcast)
- `_meta_*` — metadata (CTF name, turn index, request ID, layer list, token count)

**API endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/set_capture` | Start a capture session: `{"ctf": "experiment_name", "layers": [40,43,48]}` |
| `POST` | `/stop_capture` | End session, flush pending buffers |
| `GET` | `/capture_status` | Session state and counters |
| `GET` | `/capture_disk` | Disk usage under capture directory |

**CLI flags:** `--capture-dir`, `--capture-layers`, `--capture-max-tokens-per-request` (default 16384).

---

### Emotion / Arbitrary Vector Steering (`feature/steering-vectors`) — WIP

Extends abliteration from a single refusal direction to **arbitrary steering vectors** (e.g., emotion directions) that can be switched at runtime without server restart.

**How it works:**

1. Emotion direction vectors are loaded from a `.npz` file (`--emotion-vectors-path`) containing named vectors (e.g., `"calm"`, `"assertive"`).
2. At a target layer (`--emotion-target-layer`, default 43), the selected emotion direction is added to the hidden states, scaled by the L2 norm of the full representation:
   ```
   h' = h + strength * ||h + residual|| * emotion_dir
   ```
3. Runtime switching via file IPC — no server restart needed:
   ```json
   POST /set_steering {"emotion": "calm", "strength": 0.2}
   GET  /get_steering
   ```
   Config is written to `/tmp/emotion_config.json` and polled with a 1-second cache by the scheduler process.
4. Supports per-layer distinct direction vectors (`--steering-per-layer-path`, tensor shape `[n_layers, hidden_size]`) with separate scales for attention output (`--steering-attn-scale`) and MLP output (`--steering-mlp-scale`).
5. Trapezoidal kernel option (`--steering-kernel trapezoidal`) with configurable ramp-up/plateau/ramp-down layers for smoother multi-layer intervention.

**Additional CLI flags:**

| Flag | Description |
|------|-------------|
| `--emotion-vectors-path` | `.npz` file with named emotion direction vectors |
| `--emotion-target-layer` | Layer to apply emotion steering (default 43) |
| `--steering-per-layer-path` | Per-layer direction vectors `[n_layers, hidden_dim]` |
| `--steering-attn-scale` | Scale for post-attention intervention |
| `--steering-mlp-scale` | Scale for post-MLP intervention |
| `--steering-kernel` | `gaussian` or `trapezoidal` |
| `--steering-trap-start/end/ramp` | Trapezoidal kernel layer boundaries |

---

## News
- [2026/01] 🔥 SGLang Diffusion accelerates video and image generation ([blog](https://lmsys.org/blog/2026-01-16-sglang-diffusion/)).
- [2025/12] SGLang provides day-0 support for latest open models ([MiMo-V2-Flash](https://lmsys.org/blog/2025-12-16-mimo-v2-flash/), [Nemotron 3 Nano](https://lmsys.org/blog/2025-12-15-run-nvidia-nemotron-3-nano/), [Mistral Large 3](https://github.com/sgl-project/sglang/pull/14213), [LLaDA 2.0 Diffusion LLM](https://lmsys.org/blog/2025-12-19-diffusion-llm/), [MiniMax M2](https://lmsys.org/blog/2025-11-04-miminmax-m2/)).
- [2025/10] 🔥 SGLang now runs natively on TPU with the SGLang-Jax backend ([blog](https://lmsys.org/blog/2025-10-29-sglang-jax/)).
- [2025/09] Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part II): 3.8x Prefill, 4.8x Decode Throughput ([blog](https://lmsys.org/blog/2025-09-25-gb200-part-2/)).
- [2025/09] SGLang Day 0 Support for DeepSeek-V3.2 with Sparse Attention ([blog](https://lmsys.org/blog/2025-09-29-deepseek-V32/)).
- [2025/08] SGLang x AMD SF Meetup on 8/22: Hands-on GPU workshop, tech talks by AMD/xAI/SGLang, and networking ([Roadmap](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_roadmap.pdf), [Large-scale EP](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_ep.pdf), [Highlights](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_highlights.pdf), [AITER/MoRI](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_aiter_mori.pdf), [Wave](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_wave.pdf)).

<details>
<summary>More</summary>

- [2025/11] SGLang Diffusion accelerates video and image generation ([blog](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)).
- [2025/10] PyTorch Conference 2025 SGLang Talk ([slide](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/sglang_pytorch_2025.pdf)).
- [2025/10] SGLang x Nvidia SF Meetup on 10/2 ([recap](https://x.com/lmsysorg/status/1975339501934510231)).
- [2025/08] SGLang provides day-0 support for OpenAI gpt-oss model ([instructions](https://github.com/sgl-project/sglang/issues/8833))
- [2025/06] SGLang, the high-performance serving infrastructure powering trillions of tokens daily, has been awarded the third batch of the Open Source AI Grant by a16z ([a16z blog](https://a16z.com/advancing-open-source-ai-through-benchmarks-and-bold-experimentation/)).
- [2025/05] Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
- [2025/06] Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part I): 2.7x Higher Decoding Throughput ([blog](https://lmsys.org/blog/2025-06-16-gb200-part-1/)).
- [2025/03] Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html))
- [2025/03] SGLang Joins PyTorch Ecosystem: Efficient LLM Serving Engine ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/))
- [2025/02] Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html))
- [2025/01] SGLang provides day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs with DeepSeek-specific optimizations. ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
- [2024/12] v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
- [2024/10] The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
- [2024/09] v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
- [2024/07] v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).
- [2024/02] SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
- [2024/01] SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
- [2024/01] SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>

## About
SGLang is a high-performance serving framework for large language models and multimodal models.
It is designed to deliver low-latency and high-throughput inference across a wide range of setups, from a single GPU to large distributed clusters.
Its core features include:

- **Fast Runtime**: Provides efficient serving with RadixAttention for prefix caching, a zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert/data parallelism, structured outputs, chunked prefill, quantization (FP4/FP8/INT4/AWQ/GPTQ), and multi-LoRA batching.
- **Broad Model Support**: Supports a wide range of language models (Llama, Qwen, DeepSeek, Kimi, GLM, GPT, Gemma, Mistral, etc.), embedding models (e5-mistral, gte, mcdse), reward models (Skywork), and diffusion models (WAN, Qwen-Image), with easy extensibility for adding new models. Compatible with most Hugging Face models and OpenAI APIs.
- **Extensive Hardware Support**: Runs on NVIDIA GPUs (GB200/B300/H100/A100/Spark), AMD GPUs (MI355/MI300), Intel Xeon CPUs, Google TPUs, Ascend NPUs, and more.
- **Active Community**: SGLang is open-source and supported by a vibrant community with widespread industry adoption, powering over 400,000 GPUs worldwide.
- **RL & Post-Training Backbone**: SGLang is a proven rollout backend across the world, with native RL integrations and adoption by well-known post-training frameworks such as [**AReaL**](https://github.com/inclusionAI/AReaL), [**Miles**](https://github.com/radixark/miles), [**slime**](https://github.com/THUDM/slime), [**Tunix**](https://github.com/google/tunix), [**verl**](https://github.com/volcengine/verl) and more.

## Getting Started
- [Install SGLang](https://docs.sglang.io/get_started/install.html)
- [Quick Start](https://docs.sglang.io/basic_usage/send_request.html)
- [Backend Tutorial](https://docs.sglang.io/basic_usage/openai_api_completions.html)
- [Frontend Tutorial](https://docs.sglang.io/references/frontend/frontend_tutorial.html)
- [Contribution Guide](https://docs.sglang.io/developer_guide/contribution_guide.html)

## Benchmark and Performance
Learn more in the release blogs: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/), [GB200 rack-scale parallelism](https://lmsys.org/blog/2025-09-25-gb200-part-2/).

## Adoption and Sponsorship
SGLang has been deployed at large scale, generating trillions of tokens in production each day. It is trusted and adopted by a wide range of leading enterprises and institutions, including xAI, AMD, NVIDIA, Intel, LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, AWS, Atlas Cloud, Voltage Park, Nebius, DataCrunch, Novita, InnoMatrix, MIT, UCLA, the University of Washington, Stanford, UC Berkeley, Tsinghua University, Jam & Tea Studios, Baseten, and other major technology organizations across North America and Asia.
As an open-source LLM inference engine, SGLang has become the de facto industry standard, with deployments running on over 400,000 GPUs worldwide.
SGLang is currently hosted under the non-profit open-source organization [LMSYS](https://lmsys.org/about/).

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px"></img>

## Contact Us
For enterprises interested in adopting or deploying SGLang at scale, including technical consulting, sponsorship opportunities, or partnership inquiries, please contact us at sglang@lmsys.org

## Acknowledgment
We learned the design and reused code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).
