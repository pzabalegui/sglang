#!/usr/bin/env python3
"""
Fast SVD analysis of weight diffs using torch.svd_lowrank (randomized SVD).
Only computes top-5 singular values/vectors — enough to verify rank-1 and recover d_hat.
Uses GPU if available for massive speedup.
"""
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
OUTPUT_DIR = "/tmp"
TOP_K = 5  # Only need top-5 to verify rank-1

# Use GPU if available
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


def fast_svd(delta_w, k=TOP_K):
    """Randomized truncated SVD — O(mn*k) instead of O(mn*min(m,n))."""
    delta = delta_w.float().to(DEVICE)
    frob = delta.norm().item()
    if frob < 1e-8:
        return {"frob": frob, "is_zero": True}

    # torch.svd_lowrank: randomized SVD, much faster for large matrices
    U, S, V = torch.svd_lowrank(delta, q=k, niter=4)
    # U: [m, k], S: [k], V: [n, k]

    s_vals = S.cpu().tolist()
    s_sq_total = (S ** 2).sum().item()
    rank1_energy = (s_vals[0] ** 2) / s_sq_total if s_sq_total > 0 else 0
    s0_s1 = s_vals[0] / s_vals[1] if len(s_vals) > 1 and s_vals[1] > 1e-10 else float("inf")

    return {
        "frob": frob,
        "is_zero": False,
        "s_values": s_vals,
        "rank1_energy": rank1_energy,
        "s0_s1": s0_s1,
        "U0": U[:, 0].cpu(),   # [out_dim]
        "V0": V[:, 0].cpu(),   # [in_dim]
    }


def main():
    os.makedirs(CACHE, exist_ok=True)
    print("=" * 70)
    print("Fast SVD Analysis: Qwen 3.5-27B weight diff")
    print("Device: {}".format(DEVICE))
    print("=" * 70)
    sys.stdout.flush()

    idx_a = load_index(MODEL_A)
    idx_b = load_index(MODEL_B)

    # Storage
    best_dirs = torch.zeros(NUM_LAYERS, HIDDEN_SIZE)
    oproj_dirs = torch.zeros(NUM_LAYERS, HIDDEN_SIZE)
    downproj_dirs = torch.zeros(NUM_LAYERS, HIDDEN_SIZE)
    layer_mags = torch.zeros(NUM_LAYERS)
    oproj_vs_down_cos = torch.zeros(NUM_LAYERS)

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

        layer_results = {}
        best_energy = -1
        best_dir = None
        total_frob_sq = 0

        for mat_label, mat_name in [("o_proj", oproj_name), ("down_proj", down_name)]:
            ta = load_tensor(MODEL_A, idx_a, mat_name)
            tb = load_tensor(MODEL_B, idx_b, mat_name)
            if ta is None or tb is None:
                continue

            delta = ta.float() - tb.float()
            res = fast_svd(delta)

            if res.get("is_zero"):
                continue

            r1e = res["rank1_energy"]
            frob = res["frob"]
            total_frob_sq += frob ** 2

            # For abliteration W' = W - d * (d^T * W):
            # DeltaW = d * (d^T * W) — a rank-1 matrix
            # U[:,0] = d (the refusal direction in output/hidden space)
            # V[:,0] = (d^T * W)^T / || || (projection coefficients)
            # BOTH o_proj [5120, 6144] and down_proj [5120, 17408] have
            # output dim = hidden_size = 5120, so U[:,0] is always d_hat
            direction = res["U0"]  # [out_dim = 5120]
            direction = direction / (direction.norm() + 1e-10)

            if direction.shape[0] != HIDDEN_SIZE:
                print("  L{:02d} {}: dim={} != {}".format(
                    layer_idx, mat_label, direction.shape[0], HIDDEN_SIZE))
                continue

            if mat_label == "o_proj":
                oproj_dirs[layer_idx] = direction
            else:
                downproj_dirs[layer_idx] = direction

            if r1e > best_energy:
                best_energy = r1e
                best_dir = direction

            layer_results[mat_label] = {
                "frob": frob,
                "rank1_energy": r1e,
                "s0_s1": res["s0_s1"],
                "top5_sv": res["s_values"],
            }

            del ta, tb, delta, res
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        if best_dir is not None:
            best_dirs[layer_idx] = best_dir
            layer_mags[layer_idx] = total_frob_sq ** 0.5

        # Cosine between o_proj and down_proj dirs
        o = oproj_dirs[layer_idx]
        d = downproj_dirs[layer_idx]
        cos = 0.0
        if o.norm() > 1e-8 and d.norm() > 1e-8:
            cos = abs(torch.dot(o, d).item() / (o.norm() * d.norm()).item())
            oproj_vs_down_cos[layer_idx] = cos

        r1e_str = ""
        for ml, md in layer_results.items():
            r1e_str += "  {}:r1e={:.3f}".format(ml, md["rank1_energy"])
        dt = time.time() - t_layer
        print("  L{:02d} ({}) cos(o,d)={:.3f}{} [{:.1f}s]".format(
            layer_idx, attn_type, cos, r1e_str, dt))
        sys.stdout.flush()

        spectrum["layer_{}".format(layer_idx)] = layer_results

    total_time = time.time() - t_start
    print("\nTotal time: {:.1f}s ({:.1f}s/layer)".format(total_time, total_time / NUM_LAYERS))

    # ── Aggregate ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("AGGREGATE ANALYSIS")
    print("=" * 70)

    # Rank-1 energy
    r1e_o = []
    r1e_d = []
    for lk, ld in spectrum.items():
        if "o_proj" in ld:
            r1e_o.append(ld["o_proj"]["rank1_energy"])
        if "down_proj" in ld:
            r1e_d.append(ld["down_proj"]["rank1_energy"])

    print("\n  Rank-1 energy (1.0 = perfect rank-1 = standard abliteration):")
    if r1e_o:
        print("    o_proj:    mean={:.4f} min={:.4f} max={:.4f}".format(
            np.mean(r1e_o), np.min(r1e_o), np.max(r1e_o)))
    if r1e_d:
        print("    down_proj: mean={:.4f} min={:.4f} max={:.4f}".format(
            np.mean(r1e_d), np.min(r1e_d), np.max(r1e_d)))

    # Cross-layer consistency
    print("\n  Cross-layer direction consistency:")
    ref_idx = 32
    ref = best_dirs[ref_idx]
    if ref.norm() < 1e-8:
        ref_idx = 31
        ref = best_dirs[ref_idx]

    if ref.norm() > 1e-8:
        cross_cos = []
        for i in range(NUM_LAYERS):
            d = best_dirs[i]
            if d.norm() > 1e-8:
                c = abs(torch.dot(ref, d).item() / (ref.norm() * d.norm()).item())
                cross_cos.append((i, c))

        cos_vals = [c for _, c in cross_cos]
        print("    Reference: L{}".format(ref_idx))
        print("    Mean cos: {:.4f}".format(np.mean(cos_vals)))
        print("    >0.99: {}  >0.95: {}  >0.90: {}  <0.80: {}".format(
            sum(1 for c in cos_vals if c > 0.99),
            sum(1 for c in cos_vals if c > 0.95),
            sum(1 for c in cos_vals if c > 0.90),
            sum(1 for c in cos_vals if c < 0.80)))

        print("\n    Per-layer:")
        for i, c in cross_cos:
            is_full = i in FULL_ATTN_LAYERS
            bar = "#" * int(c * 40)
            t = "FULL" if is_full else "LIN "
            print("      L{:02d} ({}): {:.4f} {}".format(i, t, c, bar))

    # o_proj vs down_proj
    valid_cos = [oproj_vs_down_cos[i].item() for i in range(NUM_LAYERS) if oproj_vs_down_cos[i] > 0]
    if valid_cos:
        print("\n  o_proj vs down_proj (same d_hat if cos>0.95):")
        print("    Mean: {:.4f}  Min: {:.4f}  Max: {:.4f}".format(
            np.mean(valid_cos), np.min(valid_cos), np.max(valid_cos)))

    # ── Save ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAVING")
    print("=" * 70)

    saves = [
        ("wdiff_directions_per_layer_64layers.pt", best_dirs),
        ("wdiff_oproj_dirs_64layers.pt", oproj_dirs),
        ("wdiff_downproj_dirs_64layers.pt", downproj_dirs),
        ("wdiff_scaling_64layers.pt", layer_mags),
    ]
    for fname, tensor in saves:
        p = os.path.join(OUTPUT_DIR, fname)
        torch.save(tensor, p)
        print("  {} shape={}".format(p, list(tensor.shape)))

    # Global mean direction
    valid_mask = best_dirs.norm(dim=1) > 1e-8
    if valid_mask.any():
        weighted = best_dirs * layer_mags.unsqueeze(1)
        global_dir = weighted[valid_mask].sum(dim=0)
        global_dir = global_dir / global_dir.norm()
        p = os.path.join(OUTPUT_DIR, "wdiff_direction_global.pt")
        torch.save(global_dir, p)
        print("  {} shape={}".format(p, list(global_dir.shape)))

    # Spectrum JSON
    p = os.path.join(OUTPUT_DIR, "wdiff_spectrum.json")
    with open(p, "w") as f:
        json.dump(spectrum, f, indent=2)
    print("  {}".format(p))

    print("\nDONE. Run: python /tmp/compare_wdiff_wrmd.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
