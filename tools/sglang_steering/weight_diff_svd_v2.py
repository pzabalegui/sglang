#!/usr/bin/env python3
"""
Full SVD analysis of weight differences between Qwen 3.5-27B base and abliterated.
Recovers the refusal direction d_hat from rank-1 structure of DeltaW.

Correct tensor naming for Qwen 3.5:
  - model.language_model.layers.{i}.self_attn.o_proj       (full-attn, i%4==3)
  - model.language_model.layers.{i}.linear_attn.out_proj   (linear-attn)
  - model.language_model.layers.{i}.mlp.down_proj          (all layers)

Only o_proj/out_proj and down_proj are modified (write to residual stream).
For abliteration: DeltaW = (W * d_hat) x d_hat^T  (rank-1).
SVD recovery: d_hat = Vh[0] for o_proj/out_proj, U[:,0] for down_proj.
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


def svd_analysis(delta_w, top_k=10):
    """SVD of DeltaW. Returns dict with metrics + raw vectors."""
    delta = delta_w.float()
    frob = delta.norm().item()
    if frob < 1e-8:
        return {"frob": frob, "is_zero": True}

    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    S = S[:top_k]
    s_vals = S.tolist()
    s_sq_total = (S ** 2).sum().item()
    rank1_energy = (s_vals[0] ** 2) / s_sq_total if s_sq_total > 0 else 0
    s0_over_s1 = s_vals[0] / s_vals[1] if len(s_vals) > 1 and s_vals[1] > 1e-10 else float("inf")

    # Effective rank (99% energy)
    cum = torch.cumsum(S[:top_k] ** 2, dim=0) / s_sq_total
    eff_rank = int((cum < 0.99).sum().item()) + 1

    return {
        "frob": frob,
        "is_zero": False,
        "s_values": s_vals,
        "rank1_energy": rank1_energy,
        "s0_over_s1": s0_over_s1,
        "eff_rank_99": eff_rank,
        "U0": U[:, 0],   # left singular vector [out_dim]
        "Vh0": Vh[0],    # right singular vector [in_dim]
    }


def main():
    os.makedirs(CACHE, exist_ok=True)
    print("=" * 70)
    print("Full SVD Analysis: Qwen 3.5-27B weight diff")
    print("=" * 70)

    idx_a = load_index(MODEL_A)
    idx_b = load_index(MODEL_B)

    # Storage for recovered directions
    # Per-layer: best direction in hidden_size space (from whichever matrix has highest rank-1 energy)
    best_dirs = torch.zeros(NUM_LAYERS, HIDDEN_SIZE)
    # Per-layer: direction from o_proj/out_proj specifically
    oproj_dirs = torch.zeros(NUM_LAYERS, HIDDEN_SIZE)
    # Per-layer: direction from down_proj specifically
    downproj_dirs = torch.zeros(NUM_LAYERS, HIDDEN_SIZE)
    # Magnitude per layer
    layer_mags = torch.zeros(NUM_LAYERS)
    # Cosine similarity between o_proj and down_proj directions per layer
    oproj_vs_down_cos = torch.zeros(NUM_LAYERS)

    spectrum = {}

    for layer_idx in range(NUM_LAYERS):
        is_full = layer_idx in FULL_ATTN_LAYERS
        attn_type = "FULL" if is_full else "LIN"
        prefix = "model.language_model.layers.{}".format(layer_idx)

        # Two matrices to analyze per layer
        if is_full:
            oproj_name = "{}.self_attn.o_proj.weight".format(prefix)
        else:
            oproj_name = "{}.linear_attn.out_proj.weight".format(prefix)
        down_name = "{}.mlp.down_proj.weight".format(prefix)

        layer_results = {}
        best_energy = -1
        best_dir = None
        total_frob_sq = 0

        for mat_label, mat_name, use_U in [("o_proj", oproj_name, False), ("down_proj", down_name, True)]:
            ta = load_tensor(MODEL_A, idx_a, mat_name)
            tb = load_tensor(MODEL_B, idx_b, mat_name)
            if ta is None or tb is None:
                print("  L{:02d} {}: NOT FOUND".format(layer_idx, mat_label))
                continue

            delta = ta.float() - tb.float()
            res = svd_analysis(delta)

            if res.get("is_zero"):
                print("  L{:02d} {}: ZERO".format(layer_idx, mat_label))
                continue

            r1e = res["rank1_energy"]
            s0s1 = res["s0_over_s1"]
            frob = res["frob"]
            total_frob_sq += frob ** 2

            # Extract direction in hidden_size space
            if use_U:
                # down_proj: [hidden, intermediate] -> output dim is hidden
                direction = res["U0"]  # [hidden_size]
            else:
                # o_proj/out_proj: [hidden, head_dim*num_heads] -> for Qwen it's [5120, 6144]
                # The refusal direction d_hat was applied as W' = W - (W*d_hat)*d_hat^T
                # d_hat is in hidden_size space. For o_proj [out=5120, in=6144]:
                #   DeltaW = (W*d_hat) * d_hat^T, so d_hat = Vh[0] which is [in_dim=6144]
                # BUT hidden_size=5120, and in_dim=6144 (num_heads*head_dim = 48*128 = 6144)
                # So Vh[0] is NOT in hidden_size space for o_proj!
                # Actually: standard abliteration orthogonalizes in the OUTPUT dimension
                # W' = W - d_hat * (d_hat^T * W) where d_hat is [out_dim]
                # So DeltaW = d_hat * (d_hat^T * W), and U[:,0] = d_hat [out_dim=5120]
                direction = res["U0"]  # [out_dim = 5120 = hidden_size]

            direction = direction / (direction.norm() + 1e-10)

            # Verify dimension
            if direction.shape[0] != HIDDEN_SIZE:
                print("  L{:02d} {}: WRONG DIM {} (expected {})".format(
                    layer_idx, mat_label, direction.shape[0], HIDDEN_SIZE))
                continue

            # Store
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
                "s0_over_s1": s0s1,
                "eff_rank": res["eff_rank_99"],
                "top5_sv": res["s_values"][:5],
            }

            del ta, tb, delta, res
            gc.collect()

        if best_dir is not None:
            best_dirs[layer_idx] = best_dir
            layer_mags[layer_idx] = total_frob_sq ** 0.5

        # Cosine between o_proj and down_proj directions
        o = oproj_dirs[layer_idx]
        d = downproj_dirs[layer_idx]
        if o.norm() > 1e-8 and d.norm() > 1e-8:
            cos = abs(torch.dot(o, d).item() / (o.norm() * d.norm()).item())
            oproj_vs_down_cos[layer_idx] = cos
        else:
            cos = 0.0

        # Print summary line
        r1e_str = ""
        for ml, md in layer_results.items():
            r1e_str += "  {}:r1e={:.3f}".format(ml, md["rank1_energy"])
        print("  L{:02d} ({}) cos(o,d)={:.3f}{}".format(
            layer_idx, attn_type, cos, r1e_str))

        spectrum["layer_{}".format(layer_idx)] = layer_results

    # ── Aggregate analysis ──────────────────────────────────────────────
    print()
    print("=" * 70)
    print("AGGREGATE ANALYSIS")
    print("=" * 70)

    # 1. Rank-1 energy across layers
    r1e_oproj = []
    r1e_down = []
    for lk, ld in spectrum.items():
        if "o_proj" in ld:
            r1e_oproj.append(ld["o_proj"]["rank1_energy"])
        if "down_proj" in ld:
            r1e_down.append(ld["down_proj"]["rank1_energy"])

    print("\n  Rank-1 energy (higher = cleaner rank-1 = standard abliteration):")
    if r1e_oproj:
        print("    o_proj:    mean={:.4f} min={:.4f} max={:.4f}".format(
            np.mean(r1e_oproj), np.min(r1e_oproj), np.max(r1e_oproj)))
    if r1e_down:
        print("    down_proj: mean={:.4f} min={:.4f} max={:.4f}".format(
            np.mean(r1e_down), np.min(r1e_down), np.max(r1e_down)))

    # 2. Cross-layer direction consistency
    print("\n  Cross-layer direction consistency (cosine between L_i and L_j):")
    # Pick a reference layer (middle of model)
    ref_layer = 32
    ref_dir = best_dirs[ref_layer]
    if ref_dir.norm() < 1e-8:
        ref_layer = 31
        ref_dir = best_dirs[ref_layer]

    if ref_dir.norm() > 1e-8:
        cross_cos = []
        for i in range(NUM_LAYERS):
            d = best_dirs[i]
            if d.norm() > 1e-8:
                c = abs(torch.dot(ref_dir, d).item() / (ref_dir.norm() * d.norm()).item())
                cross_cos.append(c)
            else:
                cross_cos.append(0.0)
        print("    Reference: L{}".format(ref_layer))
        print("    Mean cos(L{}, L_i): {:.4f}".format(ref_layer, np.mean([c for c in cross_cos if c > 0])))
        print("    Min cos:  {:.4f}".format(np.min([c for c in cross_cos if c > 0]) if any(c > 0 for c in cross_cos) else 0))
        print("    Layers with cos > 0.99: {}".format(sum(1 for c in cross_cos if c > 0.99)))
        print("    Layers with cos > 0.95: {}".format(sum(1 for c in cross_cos if c > 0.95)))
        print("    Layers with cos > 0.90: {}".format(sum(1 for c in cross_cos if c > 0.90)))

        # Per-layer
        print("\n    Per-layer cos(L{}, L_i):".format(ref_layer))
        for i in range(NUM_LAYERS):
            if cross_cos[i] > 0:
                is_full = i in FULL_ATTN_LAYERS
                bar = "#" * int(cross_cos[i] * 40)
                star = " ***" if cross_cos[i] > 0.99 else (" **" if cross_cos[i] > 0.95 else (" *" if cross_cos[i] > 0.9 else ""))
                print("      L{:02d} ({:4s}): {:.4f} {}{}".format(
                    i, "FULL" if is_full else "LIN", cross_cos[i], bar, star))

    # 3. o_proj vs down_proj consistency
    valid_cos = [oproj_vs_down_cos[i].item() for i in range(NUM_LAYERS) if oproj_vs_down_cos[i] > 0]
    if valid_cos:
        print("\n  o_proj vs down_proj direction consistency:")
        print("    Mean cos: {:.4f}".format(np.mean(valid_cos)))
        print("    This tells us if the same d_hat was used for both matrices.")
        print("    cos > 0.95 = same direction; cos < 0.8 = different directions")

    # ── Save outputs ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    # 1. Best direction per layer [64, 5120]
    p1 = os.path.join(OUTPUT_DIR, "wdiff_directions_per_layer_64layers.pt")
    torch.save(best_dirs, p1)
    print("  Saved: {} shape={}".format(p1, list(best_dirs.shape)))

    # 2. o_proj directions per layer [64, 5120]
    p2 = os.path.join(OUTPUT_DIR, "wdiff_oproj_dirs_64layers.pt")
    torch.save(oproj_dirs, p2)
    print("  Saved: {} shape={}".format(p2, list(oproj_dirs.shape)))

    # 3. down_proj directions per layer [64, 5120]
    p3 = os.path.join(OUTPUT_DIR, "wdiff_downproj_dirs_64layers.pt")
    torch.save(downproj_dirs, p3)
    print("  Saved: {} shape={}".format(p3, list(downproj_dirs.shape)))

    # 4. Layer magnitudes [64]
    p4 = os.path.join(OUTPUT_DIR, "wdiff_scaling_64layers.pt")
    torch.save(layer_mags, p4)
    print("  Saved: {} shape={}".format(p4, list(layer_mags.shape)))

    # 5. Spectrum JSON
    p5 = os.path.join(OUTPUT_DIR, "wdiff_spectrum.json")
    # Make JSON serializable
    clean_spectrum = {}
    for lk, ld in spectrum.items():
        clean_spectrum[lk] = {}
        for mk, md in ld.items():
            clean_spectrum[lk][mk] = {k: v for k, v in md.items()}
    with open(p5, "w") as f:
        json.dump(clean_spectrum, f, indent=2)
    print("  Saved: {}".format(p5))

    # 6. Global mean direction (average across all layers, normalized)
    valid_mask = best_dirs.norm(dim=1) > 1e-8
    if valid_mask.any():
        # Weight by layer magnitude
        weighted = best_dirs * layer_mags.unsqueeze(1)
        global_dir = weighted[valid_mask].sum(dim=0)
        global_dir = global_dir / global_dir.norm()
        p6 = os.path.join(OUTPUT_DIR, "wdiff_direction_global.pt")
        torch.save(global_dir, p6)
        print("  Saved global direction: {} shape={}".format(p6, list(global_dir.shape)))

    print()
    print("DONE. Next: compare with WRMD using compare_vectors.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
