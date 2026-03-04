#!/usr/bin/env python3
"""
Compare weight-diff recovered directions vs WRMD directions.
This is the KEY diagnostic: if cos(wdiff, wrmd) is low, our WRMD
is capturing the wrong direction and that explains poor steering results.

Usage:
    python compare_wdiff_wrmd.py
"""
import json
import sys
import os

import torch
import numpy as np

NUM_LAYERS = 64
HIDDEN_SIZE = 5120
FULL_ATTN_LAYERS = set(range(3, NUM_LAYERS, 4))


def cosine_sim(a, b):
    """Absolute cosine similarity (direction-agnostic)."""
    a = a.float().flatten()
    b = b.float().flatten()
    na, nb = a.norm(), b.norm()
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return abs((a @ b).item() / (na * nb).item())


def load_dirs(path, label):
    """Load direction tensor, handle [L,H] or [L,k,H] formats."""
    d = torch.load(path, map_location="cpu", weights_only=True)
    if d.dim() == 3:
        print("  {}: shape {} -> using v[0] per layer".format(label, list(d.shape)))
        return d[:, 0, :]
    print("  {}: shape {}".format(label, list(d.shape)))
    return d


def main():
    # Paths on server
    wdiff_path = "/tmp/wdiff_directions_per_layer_64layers.pt"
    wrmd_path = "/tmp/wrmd_directions_per_layer_64layers.pt"
    caa_path = "/tmp/caa_directions_per_layer_64layers.pt"
    wdiff_scale_path = "/tmp/wdiff_scaling_64layers.pt"
    wrmd_scale_path = "/tmp/wrmd_scaling_coeffs_64layers.pt"

    print("=" * 70)
    print("Direction Comparison: Weight-Diff vs WRMD vs CAA")
    print("=" * 70)

    # Load all available direction sets
    print("\nLoading directions...")
    dirs_wdiff = load_dirs(wdiff_path, "Weight-diff")
    dirs_wrmd = load_dirs(wrmd_path, "WRMD")

    has_caa = os.path.exists(caa_path)
    dirs_caa = load_dirs(caa_path, "CAA") if has_caa else None

    # ── 1. WRMD vs Weight-diff (PRIMARY) ─────────────────────────────────
    print("\n" + "=" * 70)
    print("1. WRMD vs Weight-Diff (primary comparison)")
    print("=" * 70)

    cos_wrmd_wdiff = []
    per_layer = []
    for i in range(NUM_LAYERS):
        w = dirs_wrmd[i]
        d = dirs_wdiff[i]
        c = cosine_sim(w, d)
        cos_wrmd_wdiff.append(c)
        is_full = i in FULL_ATTN_LAYERS
        per_layer.append((i, c, is_full))

    valid = [c for c in cos_wrmd_wdiff if c > 0]
    print("\n  Summary:")
    print("    Mean cos:    {:.4f}".format(np.mean(valid) if valid else 0))
    print("    Median cos:  {:.4f}".format(np.median(valid) if valid else 0))
    print("    Min cos:     {:.4f}".format(np.min(valid) if valid else 0))
    print("    Max cos:     {:.4f}".format(np.max(valid) if valid else 0))
    print("    High (>0.95):  {}".format(sum(1 for c in valid if c > 0.95)))
    print("    Mod (0.8-0.95): {}".format(sum(1 for c in valid if 0.8 <= c <= 0.95)))
    print("    Low (<0.8):    {}".format(sum(1 for c in valid if c < 0.8)))

    # Full vs linear attention breakdown
    full_cos = [c for c in per_layer if c[2] and c[1] > 0]
    lin_cos = [c for c in per_layer if not c[2] and c[1] > 0]
    if full_cos:
        print("\n    Full-attn layers:   mean={:.4f} (n={})".format(
            np.mean([x[1] for x in full_cos]), len(full_cos)))
    if lin_cos:
        print("    Linear-attn layers: mean={:.4f} (n={})".format(
            np.mean([x[1] for x in lin_cos]), len(lin_cos)))

    # Per-layer detail
    print("\n  Per-layer cosine(WRMD, WDiff):")
    for i, c, is_full in per_layer:
        if c > 0:
            bar = "#" * int(c * 40)
            star = " ***" if c > 0.95 else (" **" if c > 0.9 else (" *" if c > 0.8 else "  LOW"))
            t = "FULL" if is_full else "LIN "
            print("    L{:02d} ({}): {:.4f} {}{}".format(i, t, c, bar, star))

    # ── 2. CAA vs Weight-diff (if available) ─────────────────────────────
    if dirs_caa is not None:
        print("\n" + "=" * 70)
        print("2. CAA vs Weight-Diff")
        print("=" * 70)
        cos_caa_wdiff = []
        for i in range(NUM_LAYERS):
            c = cosine_sim(dirs_caa[i], dirs_wdiff[i])
            cos_caa_wdiff.append(c)
        valid_caa = [c for c in cos_caa_wdiff if c > 0]
        print("    Mean cos: {:.4f}  Min: {:.4f}  Max: {:.4f}".format(
            np.mean(valid_caa), np.min(valid_caa), np.max(valid_caa)))

        print("\n3. CAA vs WRMD")
        print("=" * 70)
        cos_caa_wrmd = []
        for i in range(NUM_LAYERS):
            c = cosine_sim(dirs_caa[i], dirs_wrmd[i])
            cos_caa_wrmd.append(c)
        valid_cw = [c for c in cos_caa_wrmd if c > 0]
        print("    Mean cos: {:.4f}  Min: {:.4f}  Max: {:.4f}".format(
            np.mean(valid_cw), np.min(valid_cw), np.max(valid_cw)))

    # ── 3. Scaling correlation ───────────────────────────────────────────
    if os.path.exists(wdiff_scale_path) and os.path.exists(wrmd_scale_path):
        print("\n" + "=" * 70)
        print("Scaling Coefficient Correlation")
        print("=" * 70)
        sw = torch.load(wdiff_scale_path, map_location="cpu", weights_only=True).float()
        sr = torch.load(wrmd_scale_path, map_location="cpu", weights_only=True).float()
        if sr.dim() == 2:
            sr = sr[:, 0]
        # Normalize
        if sw.max() > 0:
            sw = sw / sw.max()
        if sr.max() > 0:
            sr = sr / sr.max()
        corr = np.corrcoef(sw.numpy(), sr.numpy())[0, 1]
        print("    Pearson r: {:.4f}".format(corr))
        print("    (r > 0.8 = strong correlation in which layers matter)")

    # ── Interpretation ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    mean_cos = np.mean(valid) if valid else 0

    if mean_cos > 0.90:
        print("  WRMD and weight-diff capture ESSENTIALLY THE SAME direction.")
        print("  Our WRMD direction is correct. The problem is NOT the direction.")
        print("  -> Issue is in the STEERING METHOD (activation-space can't replicate weight-space)")
        print("  -> Consider: partial weight modification (abliteration) on key layers")
        print("     combined with light steering on others")
    elif mean_cos > 0.70:
        print("  WRMD and weight-diff show MODERATE agreement.")
        print("  WRMD may be partially right but missing some component.")
        print("  -> Try steering with weight-diff directions instead of WRMD")
        print("  -> If that works better, WRMD needs recalibration")
    elif mean_cos > 0.50:
        print("  WRMD and weight-diff show PARTIAL overlap.")
        print("  They capture different aspects of the refusal mechanism.")
        print("  -> Combine both directions (multi-vector steering)")
        print("  -> Or: the abliterated model's direction works for weight mod")
        print("     but a different direction is needed for activation steering")
    else:
        print("  WRMD and weight-diff are NEARLY ORTHOGONAL.")
        print("  This confirms: activation-space and weight-space refusal")
        print("  representations are fundamentally different.")
        print("  -> Weight-diff direction may not help for steering")
        print("  -> Consider direct weight modification (partial abliteration)")
        print("  -> Refusal in Qwen 3.5 is likely non-linear (confirm with d')")

    print("=" * 70)

    # Save
    output = {
        "wrmd_vs_wdiff": {
            "per_layer": {str(i): c for i, c, _ in per_layer},
            "mean": float(np.mean(valid)) if valid else 0,
            "median": float(np.median(valid)) if valid else 0,
        }
    }
    out_path = "/tmp/wdiff_wrmd_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved: {}".format(out_path))


if __name__ == "__main__":
    main()
