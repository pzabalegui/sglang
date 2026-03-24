#!/usr/bin/env python3
"""
Extract refusal direction from FP8-native SGLang captures.
Uses the built-in capture mechanism to get activations from the running FP8 model.

Steps:
1. Enable capture mode via /tmp/capture_config.json
2. Send prompts via API (captures residual stream at all layers)
3. Compute mean-diff (harmful - harmless) at each layer
4. Select the layer with the best direction
5. Save the direction for deployment
"""

import json
import os
import sys
import time
import requests
import torch
import random
import shutil
import numpy as np

# ============================================================
# Configuration
# ============================================================
SERVER_URL = "http://localhost:8000"
DATA_DIR = "/tmp/arditi"
CAPTURE_DIR = "/tmp/captures"
OUTPUT_DIR = "/tmp/arditi/output_fp8"
N_TRAIN = 128
SEED = 42
N_LAYERS = 64
D_MODEL = 5120

CAPTURE_CONFIG_PATH = "/tmp/capture_config.json"

# ============================================================
# Helpers
# ============================================================

def send_prompt(prompt: str, max_tokens: int = 1) -> dict:
    """Send a single prompt to SGLang. max_tokens=1 for fast capture (we only need prefill activations)."""
    resp = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=30,
    )
    return resp.json()


def enable_capture(layers=None, save_dir=None):
    """Enable capture mode by writing config file."""
    cfg = {
        "enabled": True,
        "layers": layers or list(range(N_LAYERS)),
        "save_dir": save_dir or CAPTURE_DIR,
    }
    with open(CAPTURE_CONFIG_PATH, "w") as f:
        json.dump(cfg, f)
    time.sleep(1.5)  # wait for config cache to refresh


def disable_capture():
    """Disable capture mode."""
    if os.path.exists(CAPTURE_CONFIG_PATH):
        os.remove(CAPTURE_CONFIG_PATH)
    time.sleep(1.5)


def reset_captures():
    """Clear capture directory."""
    if os.path.exists(CAPTURE_DIR):
        shutil.rmtree(CAPTURE_DIR)
    os.makedirs(CAPTURE_DIR, exist_ok=True)


def load_captures(capture_dir: str, n_expected: int) -> dict:
    """Load all captured samples and compute mean activation per layer."""
    mean_acts = torch.zeros(N_LAYERS, D_MODEL, dtype=torch.float64)
    count = 0

    for i in range(n_expected * 2):  # search wider range
        path = os.path.join(capture_dir, f"sample_{i}.pt")
        if not os.path.exists(path):
            continue
        data = torch.load(path, weights_only=True)
        for layer_idx, tensor in data.items():
            mean_acts[layer_idx] += tensor.to(torch.float64)
        count += 1

    if count == 0:
        raise ValueError(f"No captures found in {capture_dir}")

    mean_acts /= count
    print(f"  Loaded {count} captures from {capture_dir}")
    return mean_acts


# ============================================================
# Main
# ============================================================

def main():
    random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check server
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        print(f"Server health: {resp.status_code}")
    except Exception as e:
        print(f"ERROR: Server not reachable: {e}")
        sys.exit(1)

    # Load datasets
    print("Loading datasets...")
    with open(os.path.join(DATA_DIR, "harmful_train.json")) as f:
        harmful_all = json.load(f)
    with open(os.path.join(DATA_DIR, "harmless_train.json")) as f:
        harmless_all = json.load(f)

    # Extract instruction text
    if isinstance(harmful_all[0], dict):
        harmful_all = [d["instruction"] for d in harmful_all]
    if isinstance(harmless_all[0], dict):
        harmless_all = [d["instruction"] for d in harmless_all]

    random.shuffle(harmful_all)
    random.shuffle(harmless_all)
    harmful_train = harmful_all[:N_TRAIN]
    harmless_train = harmless_all[:N_TRAIN]
    print(f"  Harmful: {len(harmful_train)}, Harmless: {len(harmless_train)}")

    # Flush radix cache
    print("Flushing cache...")
    requests.post(f"{SERVER_URL}/flush_cache", timeout=10)
    time.sleep(2)

    # ---- Phase 1: Capture harmful activations ----
    harmful_captures_dir = os.path.join(OUTPUT_DIR, "harmful_captures")
    if os.path.exists(harmful_captures_dir):
        shutil.rmtree(harmful_captures_dir)
    os.makedirs(harmful_captures_dir, exist_ok=True)

    print(f"\nCapturing harmful activations ({len(harmful_train)} prompts)...")
    enable_capture(layers=list(range(N_LAYERS)), save_dir=harmful_captures_dir)

    t0 = time.time()
    for i, prompt in enumerate(harmful_train):
        try:
            send_prompt(prompt, max_tokens=1)
        except Exception as e:
            print(f"  ERROR on harmful prompt {i}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(harmful_train)} done ({time.time()-t0:.1f}s)")

    disable_capture()
    time.sleep(1)

    n_harmful_captured = len([f for f in os.listdir(harmful_captures_dir) if f.endswith('.pt')])
    print(f"  Captured {n_harmful_captured} harmful samples in {time.time()-t0:.1f}s")

    # Flush cache between harmful and harmless
    requests.post(f"{SERVER_URL}/flush_cache", timeout=10)
    time.sleep(2)

    # ---- Phase 2: Capture harmless activations ----
    harmless_captures_dir = os.path.join(OUTPUT_DIR, "harmless_captures")
    if os.path.exists(harmless_captures_dir):
        shutil.rmtree(harmless_captures_dir)
    os.makedirs(harmless_captures_dir, exist_ok=True)

    print(f"\nCapturing harmless activations ({len(harmless_train)} prompts)...")
    enable_capture(layers=list(range(N_LAYERS)), save_dir=harmless_captures_dir)

    t0 = time.time()
    for i, prompt in enumerate(harmless_train):
        try:
            send_prompt(prompt, max_tokens=1)
        except Exception as e:
            print(f"  ERROR on harmless prompt {i}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(harmless_train)} done ({time.time()-t0:.1f}s)")

    disable_capture()
    time.sleep(1)

    n_harmless_captured = len([f for f in os.listdir(harmless_captures_dir) if f.endswith('.pt')])
    print(f"  Captured {n_harmless_captured} harmless samples in {time.time()-t0:.1f}s")

    # ---- Phase 3: Compute mean-diff ----
    print("\nComputing mean-diff directions...")
    mean_harmful = load_captures(harmful_captures_dir, n_harmful_captured)
    mean_harmless = load_captures(harmless_captures_dir, n_harmless_captured)

    mean_diff = mean_harmful - mean_harmless  # [64, 5120]

    # Norms per layer
    norms = mean_diff.norm(dim=-1)
    print(f"  Mean-diff norms: min={norms.min():.4f}, max={norms.max():.4f}")
    print(f"  Top-10 layers by norm:")
    top_layers = norms.argsort(descending=True)[:10]
    for rank, l in enumerate(top_layers):
        print(f"    #{rank+1}: layer {l.item()}, norm={norms[l]:.4f}")

    # ---- Phase 4: Direction selection ----
    # Simple approach: pick the layer with max norm (filtering last 20%)
    # Then compare with wdiff for diagnostics
    print("\nSelecting best direction...")
    max_layer = int(N_LAYERS * 0.8)  # skip last 20%
    valid_norms = norms.clone()
    valid_norms[max_layer:] = 0.0

    best_layer = valid_norms.argmax().item()
    best_direction = mean_diff[best_layer].float()
    best_direction_normalized = best_direction / best_direction.norm()

    print(f"  Selected: layer {best_layer}, norm={norms[best_layer]:.4f}")

    # Compare to wdiff
    wdiff_path = "/tmp/wdiff_direction_global.pt"
    if os.path.exists(wdiff_path):
        wdiff = torch.load(wdiff_path, weights_only=True).float()
        wdiff_norm = wdiff / wdiff.norm()

        # Compare all layers
        print(f"\n  Cosine similarity to wdiff per layer:")
        cosines = []
        for l in range(N_LAYERS):
            d = mean_diff[l].float()
            d_norm = d / (d.norm() + 1e-8)
            cos = (d_norm @ wdiff_norm).item()
            cosines.append(cos)

        # Print top layers by |cosine|
        cos_sorted = sorted(enumerate(cosines), key=lambda x: abs(x[1]), reverse=True)
        for rank, (l, cos) in enumerate(cos_sorted[:10]):
            print(f"    layer {l:2d}: cos={cos:+.4f}, norm={norms[l]:.4f}")

        print(f"\n  Best norm layer ({best_layer}): cos={cosines[best_layer]:+.4f}")

        # Try selecting by best cosine instead
        best_cos_layer = max(range(max_layer), key=lambda l: abs(cosines[l]))
        print(f"  Best |cos| layer ({best_cos_layer}): cos={cosines[best_cos_layer]:+.4f}, norm={norms[best_cos_layer]:.4f}")

    # ---- Phase 5: Save directions ----
    print(f"\nSaving directions...")

    # Save normalized direction (best by norm)
    torch.save(best_direction_normalized, os.path.join(OUTPUT_DIR, "direction_fp8_norm.pt"))
    print(f"  Saved: direction_fp8_norm.pt (layer {best_layer})")

    # Save all mean-diffs for analysis
    torch.save(mean_diff, os.path.join(OUTPUT_DIR, "mean_diff_all_layers.pt"))

    # If wdiff available, also save the best-cosine direction
    if os.path.exists(wdiff_path):
        best_cos_dir = mean_diff[best_cos_layer].float()
        best_cos_dir_normalized = best_cos_dir / best_cos_dir.norm()
        # Align sign with wdiff
        if cosines[best_cos_layer] < 0:
            best_cos_dir_normalized = -best_cos_dir_normalized
        torch.save(best_cos_dir_normalized, os.path.join(OUTPUT_DIR, "direction_fp8_cos.pt"))
        print(f"  Saved: direction_fp8_cos.pt (layer {best_cos_layer}, cos={cosines[best_cos_layer]:+.4f})")

    # Save metadata
    metadata = {
        "best_norm_layer": best_layer,
        "best_norm_value": norms[best_layer].item(),
        "best_cos_layer": best_cos_layer if os.path.exists(wdiff_path) else None,
        "best_cos_value": cosines[best_cos_layer] if os.path.exists(wdiff_path) else None,
        "n_harmful": n_harmful_captured,
        "n_harmless": n_harmless_captured,
        "norms_per_layer": norms.tolist(),
        "cosines_per_layer": cosines if os.path.exists(wdiff_path) else None,
    }
    with open(os.path.join(OUTPUT_DIR, "metadata_fp8.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("DONE! FP8-native directions extracted.")
    print(f"{'='*60}")
    print(f"\nTo deploy (best by norm):")
    print(f"  cp {OUTPUT_DIR}/direction_fp8_norm.pt /tmp/arditi_fp8_direction.pt")
    if os.path.exists(wdiff_path):
        print(f"To deploy (best by cosine to wdiff):")
        print(f"  cp {OUTPUT_DIR}/direction_fp8_cos.pt /tmp/arditi_fp8_direction.pt")


if __name__ == "__main__":
    main()
