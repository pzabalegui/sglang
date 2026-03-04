#!/usr/bin/env python3
"""
Sweep refusal direction across layers using SGLang's activation capture.

Sends harmful/harmless prompts to a running SGLang server with capture mode enabled.
Collects per-layer hidden states and computes full quality metrics.

Prerequisites:
  - SGLang running with capture patch applied
  - /tmp/capture_config.json created by this script

Usage:
    python sweep_via_sglang.py [--url http://localhost:8000]
"""

import argparse
import gc
import glob
import io
import json
import os
import shutil
import sys
import time
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep refusal direction across layers using SGLang activation capture"
    )
    parser.add_argument("--url", default=os.environ.get("SGLANG_URL", "http://localhost:8000"),
                        help="SGLang server URL (default: $SGLANG_URL or http://localhost:8000)")
    parser.add_argument("--hidden-size", type=int, default=5120,
                        help="Model hidden size (default: 5120 for GLM-4.7)")
    parser.add_argument("--n-layers", type=int, default=92,
                        help="Total number of model layers (default: 92 for GLM-4.7)")
    parser.add_argument("--sweep-start", type=int, default=30,
                        help="First layer to sweep (default: 30)")
    parser.add_argument("--sweep-end", type=int, default=62,
                        help="Last layer to sweep (default: 62)")
    parser.add_argument("--sweep-step", type=int, default=2,
                        help="Layer sweep step size (default: 2)")
    parser.add_argument("--n-train", type=int, default=64,
                        help="Number of training samples per class (default: 64)")
    parser.add_argument("--n-test", type=int, default=16,
                        help="Number of test samples per class (default: 16)")
    parser.add_argument("--output-dir", default="/tmp/fp8_vector_extraction",
                        help="Output directory for results (default: /tmp/fp8_vector_extraction)")
    parser.add_argument("--capture-dir", default="/tmp/captures",
                        help="Directory for captured activations (default: /tmp/captures)")
    return parser.parse_args()


# ============================================================
# Configuration (set from CLI args in main())
# ============================================================
SGLANG_URL = None
OUTPUT_DIR = None
CAPTURE_DIR = None
CAPTURE_CONFIG = "/tmp/capture_config.json"
SWEEP_START = None
SWEEP_END = None
SWEEP_STEP = None
N_TRAIN = None
N_TEST = None
HIDDEN_SIZE = None
N_LAYERS = None

# ============================================================
# Refusal patterns
# ============================================================
REFUSAL_PATTERNS = [
    "i cannot", "i can't", "i'm unable", "i am unable",
    "i'm sorry", "i apologize", "sorry, but",
    "as an ai", "as a language model", "as an assistant",
    "i'm not able", "i am not able",
    "against my programming", "against ethical",
    "i must refuse", "i cannot provide", "i won't",
    "it is not within my", "not within my programming",
    "i'm not going to", "i am not going to",
    "illegal", "unethical", "harmful",
    "i can not", "i do not", "i will not",
    "unable to assist", "cannot assist", "can't assist",
    "refuse to", "declining to", "not appropriate",
]


def is_refusal(text: str) -> bool:
    return any(p in text.lower() for p in REFUSAL_PATTERNS)


# ============================================================
# Dataset loading
# ============================================================
def load_datasets(n_train=64, n_test=32):
    print("[1/5] Loading datasets...")
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    response = requests.get(url, timeout=30)
    dataset = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    harmful_all = dataset["goal"].tolist()
    harmful_test = harmful_all[:n_test]
    harmful_train = harmful_all[n_test:n_test + n_train]

    # Alpaca harmless - use local fallback if datasets not available
    try:
        from datasets import load_dataset as hf_load
        hf_ds = hf_load("tatsu-lab/alpaca")
        harmless_all = [
            item["instruction"] for item in hf_ds["train"]
            if item["input"].strip() == ""
        ]
    except ImportError:
        # Fallback: use simple harmless prompts
        harmless_all = [
            "What is the capital of France?",
            "Explain photosynthesis in simple terms.",
            "Write a haiku about autumn.",
            "What are the benefits of exercise?",
            "How does a rainbow form?",
            "Tell me about the solar system.",
            "What is machine learning?",
            "Describe the water cycle.",
            "What is the Pythagorean theorem?",
            "How do airplanes fly?",
        ] * 20  # Repeat to have enough

    harmless_test = harmless_all[:n_test]
    harmless_train = harmless_all[n_test:n_test + n_train]

    print(f"    Harmful:  {len(harmful_train)} train, {len(harmful_test)} test")
    print(f"    Harmless: {len(harmless_train)} train, {len(harmless_test)} test")
    return harmful_train, harmful_test, harmless_train, harmless_test


# ============================================================
# SGLang API interaction
# ============================================================
def send_prompt(prompt: str, max_tokens: int = 1) -> str:
    """Send a single prompt to SGLang and return the response."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    try:
        resp = requests.post(
            f"{SGLANG_URL}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"    WARNING: API error: {e}")
        return ""


def wait_for_server(timeout=300):
    """Wait for SGLang server to be ready."""
    print(f"    Waiting for SGLang at {SGLANG_URL}...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            resp = requests.get(f"{SGLANG_URL}/health", timeout=5)
            if resp.status_code == 200:
                print(f"    Server ready! ({time.time()-t0:.0f}s)")
                return True
        except Exception:
            pass
        time.sleep(5)
    print(f"    ERROR: Server not ready after {timeout}s")
    return False


# ============================================================
# Activation capture
# ============================================================
def enable_capture(layers: List[int]):
    """Enable activation capture for specified layers."""
    config = {
        "enabled": True,
        "layers": layers,
        "save_dir": CAPTURE_DIR,
    }
    with open(CAPTURE_CONFIG, "w") as f:
        json.dump(config, f)


def disable_capture():
    """Disable activation capture."""
    if os.path.exists(CAPTURE_CONFIG):
        os.remove(CAPTURE_CONFIG)


def reset_captures():
    """Clear all captured activations."""
    if os.path.exists(CAPTURE_DIR):
        shutil.rmtree(CAPTURE_DIR)
    os.makedirs(CAPTURE_DIR, exist_ok=True)


def collect_activations(n_samples: int, layers: List[int]) -> Dict[int, torch.Tensor]:
    """Collect captured activations into per-layer tensors.

    Uses glob to find files (counter in SGLang process doesn't reset between phases).
    Returns: {layer_idx: tensor[n_samples, hidden_size]}
    """
    result = {l: [] for l in layers}

    # Use glob since _CAPTURE_COUNTER in SGLang doesn't reset between phases
    files = sorted(
        glob.glob(os.path.join(CAPTURE_DIR, "sample_*.pt")),
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
    )
    if len(files) < n_samples:
        print(f"    WARNING: Found {len(files)} capture files, expected {n_samples}")

    for path in files[:n_samples]:
        data = torch.load(path, weights_only=True)
        for l in layers:
            if l in data:
                result[l].append(data[l])

    # Stack into tensors
    for l in layers:
        if result[l]:
            result[l] = torch.stack(result[l])
        else:
            result[l] = torch.zeros(0, HIDDEN_SIZE)

    return result


# ============================================================
# Capture prompts via SGLang
# ============================================================
def capture_prompts(prompts: List[str], desc: str = "Capturing"):
    """Send prompts one by one and capture activations."""
    try:
        from tqdm import tqdm
        iterator = tqdm(prompts, desc=desc, leave=False)
    except ImportError:
        iterator = prompts
    for i, prompt in enumerate(iterator):
        if i % 10 == 0 and not hasattr(iterator, 'update'):
            print(f"    {desc}: {i}/{len(prompts)}...", flush=True)
        send_prompt(prompt, max_tokens=1)
        # Small delay to ensure capture file is written
        time.sleep(0.1)


# ============================================================
# Quality metrics (from abliterate_glm47_v1.py)
# ============================================================
def compute_layer_quality_from_acts(
    harmful_acts: torch.Tensor,
    harmless_acts: torch.Tensor,
    harmful_acts_test: torch.Tensor,
    harmless_acts_test: torch.Tensor,
    layer: int,
) -> Dict:
    """Compute full quality characterization from pre-captured activations."""
    harmful_mean = harmful_acts.mean(dim=0)
    harmless_mean = harmless_acts.mean(dim=0)

    direction_raw = harmful_mean - harmless_mean
    direction_norm = direction_raw.norm().item()
    direction = direction_raw / direction_raw.norm()

    harmful_mean_norm = harmful_mean.norm().item()
    harmless_mean_norm = harmless_mean.norm().item()
    snr = direction_norm / max(harmful_mean_norm, harmless_mean_norm)

    cos_sim = F.cosine_similarity(
        harmful_mean.unsqueeze(0), harmless_mean.unsqueeze(0)
    ).item()
    cosine_dissim = 1 - cos_sim

    # Training projections
    train_h_proj = (harmful_acts @ direction).numpy()
    train_hl_proj = (harmless_acts @ direction).numpy()

    # Test projections
    n_test = harmful_acts_test.shape[0]
    test_h_proj = (harmful_acts_test @ direction).numpy()
    test_hl_proj = (harmless_acts_test @ direction).numpy()

    diff = test_h_proj.mean() - test_hl_proj.mean()
    pooled_std = np.sqrt((test_h_proj.std()**2 + test_hl_proj.std()**2) / 2)
    effect_size = float(diff / pooled_std) if pooled_std > 0 else 0.0

    accuracy = float(
        ((test_h_proj > 0).sum() + (test_hl_proj < 0).sum()) / (2 * n_test)
    )
    separation_gap = float(test_h_proj.min() - test_hl_proj.max())
    composite = snr * cosine_dissim * accuracy

    proj_ratio_harmful = float(
        np.abs(test_h_proj).mean() / harmful_mean_norm
    ) if harmful_mean_norm > 0 else 0
    proj_ratio_harmless = float(
        np.abs(test_hl_proj).mean() / harmless_mean_norm
    ) if harmless_mean_norm > 0 else 0

    return {
        "layer": layer,
        "direction_norm": float(direction_norm),
        "direction": direction,
        "harmful_mean": harmful_mean,
        "harmless_mean": harmless_mean,
        "harmful_mean_norm": harmful_mean_norm,
        "harmless_mean_norm": harmless_mean_norm,
        "snr": float(snr),
        "cosine_similarity": float(cos_sim),
        "cosine_dissimilarity": float(cosine_dissim),
        "effect_size": effect_size,
        "accuracy": accuracy,
        "composite_score": float(composite),
        "harmful_proj_mean": float(test_h_proj.mean()),
        "harmful_proj_std": float(test_h_proj.std()),
        "harmful_proj_min": float(test_h_proj.min()),
        "harmful_proj_max": float(test_h_proj.max()),
        "harmless_proj_mean": float(test_hl_proj.mean()),
        "harmless_proj_std": float(test_hl_proj.std()),
        "harmless_proj_min": float(test_hl_proj.min()),
        "harmless_proj_max": float(test_hl_proj.max()),
        "separation_gap": separation_gap,
        "proj_ratio_harmful": proj_ratio_harmful,
        "proj_ratio_harmless": proj_ratio_harmless,
        "train_harmful_proj_mean": float(train_h_proj.mean()),
        "train_harmful_proj_std": float(train_h_proj.std()),
        "train_harmless_proj_mean": float(train_hl_proj.mean()),
        "train_harmless_proj_std": float(train_hl_proj.std()),
    }


# ============================================================
# Main
# ============================================================
def main():
    global SGLANG_URL, OUTPUT_DIR, CAPTURE_DIR, SWEEP_START, SWEEP_END, SWEEP_STEP
    global N_TRAIN, N_TEST, HIDDEN_SIZE, N_LAYERS

    args = parse_args()
    SGLANG_URL = args.url
    OUTPUT_DIR = args.output_dir
    CAPTURE_DIR = args.capture_dir
    SWEEP_START = args.sweep_start
    SWEEP_END = args.sweep_end
    SWEEP_STEP = args.sweep_step
    N_TRAIN = args.n_train
    N_TEST = args.n_test
    HIDDEN_SIZE = args.hidden_size
    N_LAYERS = args.n_layers

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("REFUSAL DIRECTION SWEEP via SGLang ACTIVATION CAPTURE")
    print(f"Timestamp: {timestamp}")
    print(f"Server: {SGLANG_URL}")
    print(f"Sweep: layers {SWEEP_START}-{SWEEP_END} (step {SWEEP_STEP})")
    print(f"Samples: {N_TRAIN} train, {N_TEST} test")
    print("=" * 70)

    # 1. Load datasets
    harmful_train, harmful_test, harmless_train, harmless_test = load_datasets(
        N_TRAIN, N_TEST
    )

    # 2. Wait for SGLang
    print("\n[2/5] Connecting to SGLang...")
    if not wait_for_server():
        sys.exit(1)

    # Define sweep layers
    sweep_layers = list(range(SWEEP_START, min(SWEEP_END + 1, N_LAYERS), SWEEP_STEP))
    print(f"    Target layers: {sweep_layers}")

    # 3. Capture activations
    print(f"\n[3/5] CAPTURING ACTIVATIONS")
    print("-" * 70)

    # Enable capture for ALL sweep layers at once
    enable_capture(sweep_layers)

    # Phase A: harmful train
    print("  Phase A: Harmful train prompts...")
    reset_captures()
    capture_prompts(harmful_train[:N_TRAIN], "Harmful train")
    harmful_train_acts = collect_activations(N_TRAIN, sweep_layers)
    n_captured = harmful_train_acts[sweep_layers[0]].shape[0]
    print(f"    Captured: {n_captured}/{N_TRAIN} samples")

    # Phase B: harmless train
    print("  Phase B: Harmless train prompts...")
    reset_captures()
    time.sleep(1)
    enable_capture(sweep_layers)
    capture_prompts(harmless_train[:N_TRAIN], "Harmless train")
    harmless_train_acts = collect_activations(N_TRAIN, sweep_layers)
    n_captured = harmless_train_acts[sweep_layers[0]].shape[0]
    print(f"    Captured: {n_captured}/{N_TRAIN} samples")

    # Phase C: harmful test
    print("  Phase C: Harmful test prompts...")
    reset_captures()
    time.sleep(1)
    enable_capture(sweep_layers)
    capture_prompts(harmful_test[:N_TEST], "Harmful test")
    harmful_test_acts = collect_activations(N_TEST, sweep_layers)
    n_captured = harmful_test_acts[sweep_layers[0]].shape[0]
    print(f"    Captured: {n_captured}/{N_TEST} samples")

    # Phase D: harmless test
    print("  Phase D: Harmless test prompts...")
    reset_captures()
    time.sleep(1)
    enable_capture(sweep_layers)
    capture_prompts(harmless_test[:N_TEST], "Harmless test")
    harmless_test_acts = collect_activations(N_TEST, sweep_layers)
    n_captured = harmless_test_acts[sweep_layers[0]].shape[0]
    print(f"    Captured: {n_captured}/{N_TEST} samples")

    # Disable capture
    disable_capture()

    # 4. Compute quality metrics per layer
    print(f"\n[4/5] COMPUTING QUALITY METRICS")
    print("-" * 90)
    print(f"  {'Layer':>5} | {'Effect':>7} | {'Acc':>6} | {'SNR':>7} | {'Norm':>7} | "
          f"{'CosDiss':>7} | {'Composite':>10} | {'SepGap':>7} | {'ProjRatio':>9}")
    print("-" * 90)

    all_results = []
    for layer in sweep_layers:
        h_train = harmful_train_acts[layer]
        hl_train = harmless_train_acts[layer]
        h_test = harmful_test_acts[layer]
        hl_test = harmless_test_acts[layer]

        if h_train.shape[0] == 0 or hl_train.shape[0] == 0:
            print(f"  L{layer:3d} | SKIPPED (no captures)")
            continue

        quality = compute_layer_quality_from_acts(
            h_train, hl_train, h_test, hl_test, layer
        )
        all_results.append(quality)

        print(
            f"  L{layer:3d} | "
            f"{quality['effect_size']:7.2f} | "
            f"{quality['accuracy']:5.1%} | "
            f"{quality['snr']:7.4f} | "
            f"{quality['direction_norm']:7.4f} | "
            f"{quality['cosine_dissimilarity']:7.4f} | "
            f"{quality['composite_score']:10.5f} | "
            f"{quality['separation_gap']:7.3f} | "
            f"{quality['proj_ratio_harmful']:9.4f}"
        )

    if not all_results:
        print("ERROR: No layers completed!")
        sys.exit(1)

    # Find best
    all_results_sorted = sorted(
        all_results, key=lambda x: x["composite_score"], reverse=True
    )

    print(f"\n  TOP 5 LAYERS:")
    for r in all_results_sorted[:5]:
        print(
            f"    L{r['layer']:2d}: composite={r['composite_score']:.5f}, "
            f"effect={r['effect_size']:.2f}, acc={r['accuracy']:.1%}"
        )

    best = all_results_sorted[0]
    best_layer = best["layer"]
    best_effect = sorted(all_results, key=lambda x: x["effect_size"], reverse=True)[0]
    print(f"\n  Best by composite: L{best_layer}")
    print(f"  Best by effect:    L{best_effect['layer']}")

    # 5. Save results
    print(f"\n[5/5] SAVING RESULTS")
    print("-" * 70)

    direction = best["direction"]
    vec_path = os.path.join(OUTPUT_DIR, f"refusal_direction_fp8_L{best_layer}.pt")
    torch.save(direction, vec_path)
    sglang_path = f"/tmp/refusal_direction_fp8_L{best_layer}.pt"
    torch.save(direction, sglang_path)
    print(f"  Best vector: {sglang_path}")

    # Save all directions
    all_dirs = {}
    for r in all_results:
        all_dirs[r["layer"]] = {
            "direction": r["direction"],
            "direction_norm": r["direction_norm"],
            "harmful_mean": r["harmful_mean"],
            "harmless_mean": r["harmless_mean"],
        }
    torch.save(all_dirs, os.path.join(OUTPUT_DIR, "all_layer_directions.pt"))

    # Save sweep JSON
    sweep_json = [
        {k: v for k, v in r.items()
         if k not in ("direction", "harmful_mean", "harmless_mean")}
        for r in sorted(all_results, key=lambda x: x["layer"])
    ]
    sweep_path = os.path.join(OUTPUT_DIR, f"sweep_results_fp8_{timestamp}.json")
    with open(sweep_path, "w") as f:
        json.dump(sweep_json, f, indent=2)
    print(f"  Sweep JSON: {sweep_path}")

    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE - FULL REFUSAL CHARACTERIZATION")
    print("=" * 70)
    print(f"  Best layer:      L{best_layer}")
    print(f"  Effect size:     {best['effect_size']:.2f}")
    print(f"  Accuracy:        {best['accuracy']:.1%}")
    print(f"  Direction norm:  {best['direction_norm']:.4f}")
    print(f"  SNR:             {best['snr']:.4f}")
    print(f"  Cosine dissim:   {best['cosine_dissimilarity']:.4f}")
    print(f"  Composite:       {best['composite_score']:.5f}")
    print(f"  Separation gap:  {best['separation_gap']:.3f}")
    print(f"  Vector shape:    [{HIDDEN_SIZE}]")
    print()
    print(f"  Projection stats (test set):")
    print(f"    Harmful:  mean={best['harmful_proj_mean']:.4f}, "
          f"std={best['harmful_proj_std']:.4f}")
    print(f"    Harmless: mean={best['harmless_proj_mean']:.4f}, "
          f"std={best['harmless_proj_std']:.4f}")
    print()
    print(f"  To restart SGLang with steering:")
    print(f"    --steering-vector-path {sglang_path}")
    print(f"    --steering-layers '[{best_layer}]'")
    print("=" * 70)


if __name__ == "__main__":
    main()
