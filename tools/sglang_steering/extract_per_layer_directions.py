#!/usr/bin/env python3
"""
Extract per-layer refusal direction vectors for DAS v2/v3.

Sends harmful/harmless prompts to a running SGLang server with activation capture enabled.

DAS v2 mode (default, --n-directions 1):
  Computes difference-of-means refusal direction for EACH layer individually.
  Saves a tensor of shape [n_layers, hidden_size].

DAS v3 mode (--n-directions k, k>1):
  Uses SVD on the contrast matrix (harmful - harmless_mean) to extract top-k
  refusal directions per layer. Captures multi-dimensional refusal manifold.
  Saves a tensor of shape [n_layers, k, hidden_size].
  Also saves singular_values.pt for diagnostics.

Prerequisites:
  - SGLang running with capture patch applied (glm4_moe.py with _maybe_capture)
  - Server must be started with capture-capable model

Usage:
    python extract_per_layer_directions.py [--url http://localhost:8000] [--all-layers]
    python extract_per_layer_directions.py --n-directions 3  # SVD mode (DAS v3)
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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract per-layer refusal direction vectors for DAS v2/v3"
    )
    parser.add_argument("--url", default=os.environ.get("SGLANG_URL", "http://localhost:8000"),
                        help="SGLang server URL")
    parser.add_argument("--hidden-size", type=int, default=5120,
                        help="Model hidden size (default: 5120 for GLM-4.7)")
    parser.add_argument("--n-layers", type=int, default=92,
                        help="Total number of model layers (default: 92 for GLM-4.7)")
    parser.add_argument("--n-train", type=int, default=64,
                        help="Number of training samples per class")
    parser.add_argument("--n-directions", type=int, default=1,
                        help="Number of SVD directions per layer (1=diff-of-means, >1=SVD)")
    parser.add_argument("--output-dir", default="/tmp/per_layer_vectors",
                        help="Output directory for results")
    parser.add_argument("--capture-dir", default="/tmp/captures",
                        help="Directory for captured activations")
    parser.add_argument("--all-layers", action="store_true",
                        help="Extract ALL layers (default: only L20-L80)")
    parser.add_argument("--layer-start", type=int, default=20,
                        help="First layer to extract (default: 20)")
    parser.add_argument("--layer-end", type=int, default=80,
                        help="Last layer to extract (default: 80)")
    parser.add_argument("--layer-step", type=int, default=1,
                        help="Layer step (default: 1 = every layer)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of layers to capture simultaneously (memory tradeoff)")
    parser.add_argument("--reference-vector", type=str, default=None,
                        help="Path to reference vector for cosine validation (e.g. /tmp/refusal_direction_fp8_L47.pt)")
    parser.add_argument("--reference-layer", type=int, default=47,
                        help="Layer index of the reference vector (default: 47)")
    return parser.parse_args()


CAPTURE_CONFIG = "/tmp/capture_config.json"


def send_prompt(url: str, prompt: str, max_tokens: int = 1) -> str:
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "steering_enabled": False,  # Disable steering for clean activation capture
    }
    try:
        resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"    WARNING: API error: {e}")
        return ""


def wait_for_server(url: str, timeout=300) -> bool:
    print(f"    Waiting for SGLang at {url}...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                print(f"    Server ready! ({time.time()-t0:.0f}s)")
                return True
        except Exception:
            pass
        time.sleep(5)
    print(f"    ERROR: Server not ready after {timeout}s")
    return False


def enable_capture(layers: List[int], capture_dir: str):
    config = {"enabled": True, "layers": layers, "save_dir": capture_dir}
    with open(CAPTURE_CONFIG, "w") as f:
        json.dump(config, f)


def disable_capture():
    if os.path.exists(CAPTURE_CONFIG):
        os.remove(CAPTURE_CONFIG)


def reset_captures(capture_dir: str):
    if os.path.exists(capture_dir):
        shutil.rmtree(capture_dir)
    os.makedirs(capture_dir, exist_ok=True)


def collect_activations(capture_dir: str, n_samples: int, layers: List[int],
                        hidden_size: int) -> Dict[int, torch.Tensor]:
    result = {l: [] for l in layers}
    files = sorted(
        glob.glob(os.path.join(capture_dir, "sample_*.pt")),
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
    )
    if len(files) < n_samples:
        print(f"    WARNING: Found {len(files)} capture files, expected {n_samples}")

    for path in files[:n_samples]:
        data = torch.load(path, weights_only=True)
        for l in layers:
            if l in data:
                result[l].append(data[l])

    for l in layers:
        if result[l]:
            result[l] = torch.stack(result[l])
        else:
            result[l] = torch.zeros(0, hidden_size)

    return result


def capture_prompts(url: str, prompts: List[str], desc: str = "Capturing"):
    try:
        from tqdm import tqdm
        iterator = tqdm(prompts, desc=desc, leave=False)
    except ImportError:
        iterator = prompts
    for i, prompt in enumerate(iterator):
        if i % 10 == 0 and not hasattr(iterator, 'update'):
            print(f"    {desc}: {i}/{len(prompts)}...", flush=True)
        send_prompt(url, prompt, max_tokens=1)
        time.sleep(0.1)


def load_datasets(n_train: int):
    print("[1/5] Loading datasets...")
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    response = requests.get(url, timeout=30)
    dataset = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    harmful_all = dataset["goal"].tolist()
    harmful_train = harmful_all[:n_train]

    try:
        from datasets import load_dataset as hf_load
        hf_ds = hf_load("tatsu-lab/alpaca")
        harmless_all = [
            item["instruction"] for item in hf_ds["train"]
            if item["input"].strip() == ""
        ]
    except ImportError:
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
        ] * 20

    harmless_train = harmless_all[:n_train]
    print(f"    Harmful:  {len(harmful_train)} train")
    print(f"    Harmless: {len(harmless_train)} train")
    return harmful_train, harmless_train


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    k = args.n_directions

    if args.all_layers:
        target_layers = list(range(args.n_layers))
    else:
        target_layers = list(range(args.layer_start, min(args.layer_end + 1, args.n_layers), args.layer_step))

    das_version = "v3 (SVD)" if k > 1 else "v2 (diff-of-means)"
    print("=" * 70)
    print(f"PER-LAYER REFUSAL DIRECTION EXTRACTION (DAS {das_version})")
    print(f"Timestamp: {timestamp}")
    print(f"Server: {args.url}")
    print(f"Layers: {len(target_layers)} ({target_layers[0]}-{target_layers[-1]})")
    print(f"Samples: {args.n_train} per class")
    print(f"Directions per layer: {k}")
    print(f"Batch size: {args.batch_size} layers per capture pass")
    print("=" * 70)

    # 1. Load datasets
    harmful_train, harmless_train = load_datasets(args.n_train)

    # 2. Wait for server
    print("\n[2/5] Connecting to SGLang...")
    if not wait_for_server(args.url):
        sys.exit(1)

    # 3. Capture activations in batches of layers
    # (Capturing all 92 layers at once would use too much memory per sample file)
    print(f"\n[3/5] CAPTURING ACTIVATIONS")
    print("-" * 70)

    all_harmful_acts = {}
    all_harmless_acts = {}

    layer_batches = []
    for i in range(0, len(target_layers), args.batch_size):
        layer_batches.append(target_layers[i:i + args.batch_size])

    print(f"  {len(layer_batches)} capture passes needed ({args.batch_size} layers/pass)")

    for batch_idx, layer_batch in enumerate(layer_batches):
        print(f"\n  Pass {batch_idx+1}/{len(layer_batches)}: layers {layer_batch[0]}-{layer_batch[-1]}")

        # Harmful prompts
        reset_captures(args.capture_dir)
        enable_capture(layer_batch, args.capture_dir)
        capture_prompts(args.url, harmful_train[:args.n_train], f"Harmful [{layer_batch[0]}-{layer_batch[-1]}]")
        harmful_acts = collect_activations(args.capture_dir, args.n_train, layer_batch, args.hidden_size)
        for l in layer_batch:
            all_harmful_acts[l] = harmful_acts[l]
        n_cap = harmful_acts[layer_batch[0]].shape[0]
        print(f"    Harmful captured: {n_cap}/{args.n_train}")

        # Harmless prompts
        reset_captures(args.capture_dir)
        time.sleep(1)
        enable_capture(layer_batch, args.capture_dir)
        capture_prompts(args.url, harmless_train[:args.n_train], f"Harmless [{layer_batch[0]}-{layer_batch[-1]}]")
        harmless_acts = collect_activations(args.capture_dir, args.n_train, layer_batch, args.hidden_size)
        for l in layer_batch:
            all_harmless_acts[l] = harmless_acts[l]
        n_cap = harmless_acts[layer_batch[0]].shape[0]
        print(f"    Harmless captured: {n_cap}/{args.n_train}")

    disable_capture()

    # 4. Compute per-layer directions
    if k > 1:
        print(f"\n[4/5] COMPUTING PER-LAYER REFUSAL DIRECTIONS (SVD, k={k})")
    else:
        print(f"\n[4/5] COMPUTING PER-LAYER REFUSAL DIRECTIONS (diff-of-means)")
    print("-" * 90)

    if k > 1:
        # SVD mode: output [n_layers, k, hidden_size]
        per_layer_directions = torch.zeros(args.n_layers, k, args.hidden_size, dtype=torch.float32)
        singular_values = torch.zeros(args.n_layers, min(k, args.n_train), dtype=torch.float32)
        print(f"  {'Layer':>5} | {'S[0]':>8} | {'S[1]/S[0]':>9} | {'cos(v0,dom)':>11} | {'Sep_v0':>8} | {'Sep_v1':>8}")
        print("-" * 90)
    else:
        # Difference-of-means mode: output [n_layers, hidden_size]
        per_layer_directions = torch.zeros(args.n_layers, args.hidden_size, dtype=torch.float32)
        singular_values = None
        print(f"  {'Layer':>5} | {'Norm':>8} | {'CosDisS':>8} | {'H_proj':>8} | {'HL_proj':>8} | {'Sep':>8}")
        print("-" * 90)

    per_layer_norms = torch.zeros(args.n_layers, dtype=torch.float32)
    layer_stats = {}

    for layer in target_layers:
        h_acts = all_harmful_acts.get(layer)
        hl_acts = all_harmless_acts.get(layer)

        if h_acts is None or hl_acts is None or h_acts.shape[0] == 0 or hl_acts.shape[0] == 0:
            print(f"  L{layer:3d} | SKIPPED (no captures)")
            continue

        harmful_mean = h_acts.mean(dim=0)
        harmless_mean = hl_acts.mean(dim=0)

        # Difference-of-means direction (always computed for stats/validation)
        dom_raw = harmful_mean - harmless_mean
        dom_norm = dom_raw.norm().item()
        dom_dir = dom_raw / dom_raw.norm()

        per_layer_norms[layer] = dom_norm

        if k > 1:
            # SVD mode: contrast matrix C = H_harm - mu_safe
            C = h_acts.float() - harmless_mean.unsqueeze(0).float()  # [n_harm, hidden_size]
            U, S, Vh = torch.linalg.svd(C, full_matrices=False)
            # Vh: [min(n_harm, hidden_size), hidden_size] — top singular vectors
            n_sv = min(k, Vh.shape[0])
            dirs_k = F.normalize(Vh[:n_sv], dim=1)  # [n_sv, hidden_size]

            # Fix sign ambiguity: ensure each direction has positive projection
            # along harmful-harmless axis (so clamp_(min=0) works correctly).
            # Convention: harmful samples should project positively onto the direction.
            for _ki in range(n_sv):
                h_proj_mean = (h_acts @ dirs_k[_ki]).mean().item()
                hl_proj_mean = (hl_acts @ dirs_k[_ki]).mean().item()
                if h_proj_mean < hl_proj_mean:
                    dirs_k[_ki] = -dirs_k[_ki]  # flip to match convention

            # Store directions
            per_layer_directions[layer, :n_sv] = dirs_k

            # Store singular values for diagnostics
            n_sv_save = min(singular_values.shape[1], S.shape[0])
            singular_values[layer, :n_sv_save] = S[:n_sv_save]

            # Validation: Vh[0] should approximate difference-of-means
            cos_v0_dom = F.cosine_similarity(dirs_k[0].unsqueeze(0), dom_dir.unsqueeze(0)).item()

            # Separation stats for v0 and v1
            sep_v0 = (h_acts @ dirs_k[0]).mean().item() - (hl_acts @ dirs_k[0]).mean().item()
            sep_v1 = 0.0
            sv_ratio = 0.0
            if n_sv > 1:
                sep_v1 = (h_acts @ dirs_k[1]).mean().item() - (hl_acts @ dirs_k[1]).mean().item()
                sv_ratio = (S[1] / S[0]).item() if S[0].item() > 1e-8 else 0.0

            layer_stats[layer] = {
                "dom_norm": dom_norm,
                "cos_v0_dom": cos_v0_dom,
                "singular_values": S[:min(k, len(S))].tolist(),
                "sv_ratio_1_0": sv_ratio,
                "separation_v0": sep_v0,
                "separation_v1": sep_v1,
            }

            print(f"  L{layer:3d} | {S[0].item():8.2f} | {sv_ratio:9.4f} | "
                  f"{cos_v0_dom:11.4f} | {sep_v0:8.4f} | {sep_v1:8.4f}")
        else:
            # Difference-of-means mode (v2 compatible)
            per_layer_directions[layer] = dom_dir

            cos_sim = F.cosine_similarity(harmful_mean.unsqueeze(0), harmless_mean.unsqueeze(0)).item()
            h_proj = (h_acts @ dom_dir).mean().item()
            hl_proj = (hl_acts @ dom_dir).mean().item()
            separation = h_proj - hl_proj

            layer_stats[layer] = {
                "norm": dom_norm,
                "cos_dissim": 1 - cos_sim,
                "harmful_proj_mean": h_proj,
                "harmless_proj_mean": hl_proj,
                "separation": separation,
            }

            print(f"  L{layer:3d} | {dom_norm:8.4f} | {1-cos_sim:8.4f} | "
                  f"{h_proj:8.4f} | {hl_proj:8.4f} | {separation:8.4f}")

    # 5. Save results
    print(f"\n[5/5] SAVING RESULTS")
    print("-" * 70)

    # Save the main per-layer directions tensor
    if k > 1:
        suffix = f"k{k}"
        output_path = os.path.join(args.output_dir,
                                   f"refusal_directions_per_layer_{args.n_layers}layers_{suffix}.pt")
        sglang_path = f"/tmp/refusal_directions_per_layer_{args.n_layers}layers_{suffix}.pt"
    else:
        output_path = os.path.join(args.output_dir,
                                   f"refusal_directions_per_layer_{args.n_layers}layers.pt")
        sglang_path = f"/tmp/refusal_directions_per_layer_{args.n_layers}layers.pt"

    torch.save(per_layer_directions, output_path)
    print(f"  Per-layer directions: {output_path}")
    print(f"    Shape: {list(per_layer_directions.shape)}")

    # Also save to /tmp for easy server access
    torch.save(per_layer_directions, sglang_path)
    print(f"  Copy for server: {sglang_path}")

    # Save per-layer norms
    norms_path = os.path.join(args.output_dir, "per_layer_norms.pt")
    torch.save(per_layer_norms, norms_path)

    # Save singular values (SVD mode only)
    if singular_values is not None:
        sv_path = os.path.join(args.output_dir, "singular_values.pt")
        torch.save(singular_values, sv_path)
        sv_sglang = "/tmp/singular_values.pt"
        torch.save(singular_values, sv_sglang)
        print(f"  Singular values: {sv_path}")
        print(f"  Copy for server: {sv_sglang}")

        # Print SVD spectrum summary
        active_sv = [layer for layer in target_layers if per_layer_norms[layer].item() > 0]
        if active_sv:
            ratios = []
            for l in active_sv:
                s = singular_values[l]
                if s[0].item() > 1e-8 and len(s) > 1:
                    ratios.append(s[1].item() / s[0].item())
            if ratios:
                mean_ratio = sum(ratios) / len(ratios)
                print(f"\n  SVD SPECTRUM SUMMARY:")
                print(f"    Mean S[1]/S[0] across active layers: {mean_ratio:.4f}")
                useful_k2 = sum(1 for r in ratios if r > 0.05)
                print(f"    Layers where k>1 is useful (S[1]/S[0] > 0.05): {useful_k2}/{len(ratios)}")
                if mean_ratio < 0.05:
                    print(f"    WARNING: 2nd direction is mostly noise. Consider using k=1.")

    # Save stats JSON
    stats_path = os.path.join(args.output_dir, f"per_layer_stats_{timestamp}.json")
    with open(stats_path, "w") as f:
        json.dump(layer_stats, f, indent=2)
    print(f"  Stats JSON: {stats_path}")

    # Validate against reference vector if provided
    if args.reference_vector and os.path.exists(args.reference_vector):
        ref_vec = torch.load(args.reference_vector, map_location="cpu", weights_only=True)
        ref_vec = ref_vec.float()
        ref_vec = ref_vec / ref_vec.norm()

        ref_layer = args.reference_layer
        if k > 1:
            new_vec = per_layer_directions[ref_layer, 0]  # First SVD direction
        else:
            new_vec = per_layer_directions[ref_layer]
        if new_vec.norm() > 0:
            cos = F.cosine_similarity(ref_vec.unsqueeze(0), new_vec.unsqueeze(0)).item()
            print(f"\n  VALIDATION: L{ref_layer} v0 cosine vs reference: {cos:.6f}")
            if cos > 0.99:
                print(f"    PASS: Vectors are essentially identical")
            elif cos > 0.95:
                print(f"    OK: Vectors are very similar")
            else:
                print(f"    WARNING: Vectors differ significantly (cos={cos:.4f})")
        else:
            print(f"\n  WARNING: L{ref_layer} direction is zero — was this layer captured?")

    # Summary
    active_layers = [l for l in target_layers if per_layer_norms[l].item() > 0]
    if active_layers:
        best_layer = max(active_layers, key=lambda l: layer_stats.get(l, {}).get(
            "separation_v0" if k > 1 else "separation", 0))
        sep_key = "separation_v0" if k > 1 else "separation"
        print(f"\n  Summary:")
        print(f"    Active layers: {len(active_layers)}/{len(target_layers)}")
        print(f"    Directions per layer: {k}")
        print(f"    Best separation: L{best_layer} (sep={layer_stats[best_layer][sep_key]:.4f})")
        print(f"    Norm range: {min(per_layer_norms[l].item() for l in active_layers):.4f} - "
              f"{max(per_layer_norms[l].item() for l in active_layers):.4f}")

    if k > 1:
        print(f"\n  To use with DAS v3:")
        print(f"    --steering-per-layer-path {sglang_path}")
        print(f"    --steering-n-directions {k}")
    else:
        print(f"\n  To use with DAS v2:")
        print(f"    --steering-per-layer-path {sglang_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
