#!/usr/bin/env python3
"""
Rank-k per-layer weight-diff SVD extraction for Qwen 3.5-27B.

For each layer, horizontally concatenates ΔW_oproj and ΔW_downproj into
[5120, in_oproj + in_downproj], then extracts top-k left singular vectors.
This captures the joint principal refusal directions across both matrices.

Output:
  - wdiff_rankk_dirs_64layers.pt: [64, k, 5120] per-layer rank-k directions
  - wdiff_rankk_sv_weights_64layers.pt: [64, k] singular value weights
  - wdiff_rankk_spectrum.json: per-layer SVD analysis

Usage:
  python3 extract_rankk_wdiff_qwen35.py [--rank 3] [--output-dir /tmp]
"""
import argparse
import json
import sys
import os
import time
import gc

import torch
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

CACHE = "/tmp/wdiff_cache"
MODEL_A = "Qwen/Qwen3.5-27B"
MODEL_B = "huihui-ai/Huihui-Qwen3.5-27B-abliterated"
NUM_LAYERS = 64
HIDDEN_SIZE = 5120
FULL_ATTN_LAYERS = set(range(3, NUM_LAYERS, 4))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_index(model_id):
    path = hf_hub_download(model_id, "model.safetensors.index.json", cache_dir=CACHE)
    with open(path) as f:
        return json.load(f)


def load_tensor(model_id, index, tensor_name):
    shard = index.get("weight_map", {}).get(tensor_name)
    if shard is None:
        return None
    shard_path = hf_hub_download(model_id, shard, cache_dir=CACHE)
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        if tensor_name in f.keys():
            return f.get_tensor(tensor_name)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=3, help="Number of SVD directions per layer")
    parser.add_argument("--output-dir", default="/tmp")
    args = parser.parse_args()

    K = args.rank
    OUTPUT_DIR = args.output_dir
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Rank-{} Per-Layer Weight-Diff SVD Extraction".format(K))
    print("Device: {}".format(DEVICE))
    print("=" * 70)
    sys.stdout.flush()

    idx_a = load_index(MODEL_A)
    idx_b = load_index(MODEL_B)

    # Storage: [n_layers, k, hidden_size]
    all_dirs = torch.zeros(NUM_LAYERS, K, HIDDEN_SIZE)
    all_sv_weights = torch.zeros(NUM_LAYERS, K)
    spectrum = {}

    t_start = time.time()

    for layer_idx in range(NUM_LAYERS):
        t_layer = time.time()
        is_full = layer_idx in FULL_ATTN_LAYERS
        attn_type = "FULL" if is_full else "LIN"
        prefix = "model.language_model.layers.{}".format(layer_idx)

        if is_full:
            oproj_name = "{}.self_attn.o_proj.weight".format(prefix)
        else:
            oproj_name = "{}.linear_attn.out_proj.weight".format(prefix)
        down_name = "{}.mlp.down_proj.weight".format(prefix)

        # Load both weight diffs
        deltas = []
        for mat_label, mat_name in [("o_proj", oproj_name), ("down_proj", down_name)]:
            ta = load_tensor(MODEL_A, idx_a, mat_name)
            tb = load_tensor(MODEL_B, idx_b, mat_name)
            if ta is not None and tb is not None:
                delta = (ta.float() - tb.float())
                if delta.norm() > 1e-8:
                    deltas.append((mat_label, delta))
            del ta, tb
            gc.collect()

        if not deltas:
            print("  L{:02d} ({}): NO DIFF".format(layer_idx, attn_type))
            spectrum["layer_{}".format(layer_idx)] = {"skipped": True}
            continue

        # Horizontally concatenate: [5120, in_o + in_d]
        # Both matrices have out_dim = HIDDEN_SIZE = 5120
        combined = torch.cat([d for _, d in deltas], dim=1).float().to(DEVICE)
        frob = combined.norm().item()

        # Randomized SVD — top-k
        q = min(K + 2, min(combined.shape))  # slight oversampling for accuracy
        U, S, V = torch.svd_lowrank(combined, q=q, niter=6)
        # U: [5120, q], S: [q]

        # Take top-k
        k_actual = min(K, S.shape[0])
        s_vals = S[:k_actual].cpu().tolist()
        dirs = U[:, :k_actual].cpu()  # [5120, k]

        # Normalize each direction
        for ki in range(k_actual):
            d = dirs[:, ki]
            d_norm = d.norm()
            if d_norm > 1e-10:
                dirs[:, ki] = d / d_norm

        # Store: [k, 5120]
        all_dirs[layer_idx, :k_actual, :] = dirs.T
        all_sv_weights[layer_idx, :k_actual] = torch.tensor(s_vals)

        # Compute energy fractions
        s_sq_total = (S ** 2).sum().item()
        energies = [(sv ** 2) / s_sq_total for sv in s_vals]
        cumulative = [sum(energies[:i+1]) for i in range(len(energies))]

        # Cross-rank orthogonality check
        ortho_check = ""
        if k_actual >= 2:
            cos_01 = abs(torch.dot(dirs[:, 0], dirs[:, 1]).item())
            ortho_check = " cos(d0,d1)={:.4f}".format(cos_01)

        dt = time.time() - t_layer
        energy_str = " ".join(["r{}e={:.3f}".format(i+1, e) for i, e in enumerate(energies)])
        cum_str = "cum_r{}={:.3f}".format(k_actual, cumulative[-1]) if cumulative else ""
        print("  L{:02d} ({}) {} {} {} [{:.1f}s]".format(
            layer_idx, attn_type, energy_str, cum_str, ortho_check, dt))
        sys.stdout.flush()

        spectrum["layer_{}".format(layer_idx)] = {
            "sv_values": s_vals,
            "energies": energies,
            "cumulative_energy": cumulative,
            "frob": frob,
            "mats_used": [label for label, _ in deltas],
        }

        del combined, U, S, V, deltas
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    total_time = time.time() - t_start
    print("\nTotal time: {:.1f}s ({:.1f}s/layer)".format(total_time, total_time / NUM_LAYERS))

    # ── Cross-layer consistency ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-LAYER ANALYSIS")
    print("=" * 70)

    ref_idx = 32
    ref_d0 = all_dirs[ref_idx, 0]
    if ref_d0.norm() < 1e-8:
        ref_idx = 31
        ref_d0 = all_dirs[ref_idx, 0]

    if ref_d0.norm() > 1e-8:
        print("\n  Rank-1 cross-layer consistency (vs L{}):".format(ref_idx))
        cos_vals = []
        for i in range(NUM_LAYERS):
            d = all_dirs[i, 0]
            if d.norm() > 1e-8:
                c = abs(torch.dot(ref_d0, d).item() / (ref_d0.norm() * d.norm()).item())
                cos_vals.append(c)
                marker = " ***" if c < 0.90 else ""
                print("    L{:02d}: cos={:.4f}{}".format(i, c, marker))
        print("    Mean: {:.4f}  Min: {:.4f}  >0.95: {}/{}".format(
            np.mean(cos_vals), np.min(cos_vals),
            sum(1 for c in cos_vals if c > 0.95), len(cos_vals)))

    # Rank-2 cross-layer consistency
    ref_d1 = all_dirs[ref_idx, 1]
    if ref_d1.norm() > 1e-8:
        print("\n  Rank-2 cross-layer consistency (vs L{}):".format(ref_idx))
        cos2_vals = []
        for i in range(NUM_LAYERS):
            d = all_dirs[i, 1]
            if d.norm() > 1e-8:
                c = abs(torch.dot(ref_d1, d).item() / (ref_d1.norm() * d.norm()).item())
                cos2_vals.append(c)
        if cos2_vals:
            print("    Mean: {:.4f}  Min: {:.4f}  Max: {:.4f}".format(
                np.mean(cos2_vals), np.min(cos2_vals), np.max(cos2_vals)))

    # SV weight analysis
    print("\n  Singular value weights (mean across layers):")
    for ki in range(K):
        vals = all_sv_weights[:, ki]
        valid = vals[vals > 1e-8]
        if len(valid) > 0:
            print("    SV{}: mean={:.4f} min={:.4f} max={:.4f}".format(
                ki+1, valid.mean().item(), valid.min().item(), valid.max().item()))

    # ── Save ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAVING")
    print("=" * 70)

    p = os.path.join(OUTPUT_DIR, "wdiff_rankk_dirs_64layers.pt")
    torch.save(all_dirs, p)
    print("  {} shape={}".format(p, list(all_dirs.shape)))

    p = os.path.join(OUTPUT_DIR, "wdiff_rankk_sv_weights_64layers.pt")
    torch.save(all_sv_weights, p)
    print("  {} shape={}".format(p, list(all_sv_weights.shape)))

    p = os.path.join(OUTPUT_DIR, "wdiff_rankk_spectrum.json")
    with open(p, "w") as f:
        json.dump(spectrum, f, indent=2)
    print("  {}".format(p))

    print("\nDONE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
