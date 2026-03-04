#!/usr/bin/env python3
"""
Weight-diff quick check for Qwen 3.5-27B: correct tensor naming.
Qwen 3.5 uses:
  - model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj  (full-attn: i % 4 == 3)
  - model.language_model.layers.{i}.linear_attn.{out_proj, in_proj_*}  (linear-attn)
  - model.language_model.layers.{i}.mlp.{gate,up,down}_proj  (all layers)
"""
import json
import sys
import os
import time
import gc

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

CACHE = "/tmp/wdiff_cache"
MODEL_A = "Qwen/Qwen3.5-27B"
MODEL_B = "huihui-ai/Huihui-Qwen3.5-27B-abliterated"
NUM_LAYERS = 64
HIDDEN_SIZE = 5120
FULL_ATTN_LAYERS = set(range(3, NUM_LAYERS, 4))


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


def get_matrices_for_layer(layer_idx):
    """Return list of (short_name, full_tensor_name) for a layer."""
    prefix = "model.language_model.layers.{}".format(layer_idx)
    is_full = layer_idx in FULL_ATTN_LAYERS
    matrices = []

    if is_full:
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            full = "{}.self_attn.{}.weight".format(prefix, name)
            matrices.append(("self_attn." + name, full))
    else:
        for name in ["out_proj", "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z"]:
            full = "{}.linear_attn.{}.weight".format(prefix, name)
            matrices.append(("linear_attn." + name, full))

    for name in ["gate_proj", "up_proj", "down_proj"]:
        full = "{}.mlp.{}.weight".format(prefix, name)
        matrices.append(("mlp." + name, full))

    return matrices


def main():
    os.makedirs(CACHE, exist_ok=True)

    print("=" * 70)
    print("Weight Diff Check v2: Qwen 3.5-27B base vs abliterated")
    print("=" * 70)
    print("  Model A: {}".format(MODEL_A))
    print("  Model B: {}".format(MODEL_B))
    print()

    # Download indices
    print("[1/3] Loading indices...")
    t0 = time.time()
    idx_a = load_index(MODEL_A)
    idx_b = load_index(MODEL_B)
    print("  A: {} tensors, B: {} tensors".format(
        len(idx_a["weight_map"]), len(idx_b["weight_map"])))
    print("  Done in {:.1f}s".format(time.time() - t0))

    # Check for naming differences
    keys_a = set(idx_a["weight_map"].keys())
    keys_b = set(idx_b["weight_map"].keys())
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    if only_a or only_b:
        print("  WARNING: {} tensors only in A, {} only in B".format(len(only_a), len(only_b)))
        if only_a:
            for k in sorted(only_a)[:5]:
                print("    A only: {}".format(k))
        if only_b:
            for k in sorted(only_b)[:5]:
                print("    B only: {}".format(k))

    # Check layers
    check_layers = [0, 3, 7, 10, 15, 20, 25, 31, 32, 35, 38, 42, 47, 50, 55, 59, 60, 63]
    print()
    print("[2/3] Comparing weight matrices...")

    n_modified = 0
    n_identical = 0
    n_missing = 0
    all_results = {}

    for layer_idx in check_layers:
        is_full = layer_idx in FULL_ATTN_LAYERS
        attn_type = "FULL" if is_full else "LINEAR"
        print("\n  Layer {} ({}):".format(layer_idx, attn_type))

        matrices = get_matrices_for_layer(layer_idx)
        layer_results = {}

        for short_name, full_name in matrices:
            ta = load_tensor(MODEL_A, idx_a, full_name)
            tb = load_tensor(MODEL_B, idx_b, full_name)

            if ta is None or tb is None:
                print("    {}: NOT FOUND".format(short_name))
                n_missing += 1
                continue

            a = ta.float()
            b = tb.float()
            delta = a - b
            frob = delta.norm().item()
            rel = frob / (a.norm().item() + 1e-10)
            maxd = delta.abs().max().item()
            is_same = torch.allclose(a, b, atol=1e-6, rtol=1e-5)

            status = "IDENTICAL" if is_same else "MODIFIED"
            if is_same:
                n_identical += 1
            else:
                n_modified += 1

            print("    {:35s} {:10s} ||D||={:.6f} rel={:.2e} max={:.2e} shape={}".format(
                short_name, status, frob, rel, maxd, list(ta.shape)))

            layer_results[short_name] = {
                "status": status,
                "frobenius_norm": frob,
                "relative_frobenius": rel,
                "max_abs_diff": maxd,
                "shape": list(ta.shape),
            }

            del ta, tb, a, b, delta
            gc.collect()

        all_results["layer_{}".format(layer_idx)] = layer_results

    # Summary
    print()
    print("[3/3] Summary")
    print("=" * 70)
    total = n_modified + n_identical
    print("  Tensors compared: {}".format(total))
    print("  MODIFIED:         {}".format(n_modified))
    print("  IDENTICAL:        {}".format(n_identical))
    print("  Missing/Error:    {}".format(n_missing))

    if n_modified > 0:
        print()
        print("  RESULT: WEIGHT MODIFICATION DETECTED")
        print("  The abliterated model has actual weight changes.")
        print("  -> Proceed to full SVD analysis")

        # List modified matrices
        print()
        print("  Modified matrices by type:")
        type_counts = {}
        for lk, ld in all_results.items():
            for mk, md in ld.items():
                if md["status"] == "MODIFIED":
                    base = mk.split(".")[0] + "." + mk.split(".")[1] if "." in mk else mk
                    type_counts[base] = type_counts.get(base, 0) + 1
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print("    {}: {} layers modified".format(t, c))

    elif n_identical > 0 and n_modified == 0:
        print()
        print("  RESULT: NO WEIGHT MODIFICATION DETECTED")
        print("  Model may use runtime hooks. Weight-diff is NOT viable.")
    else:
        print()
        print("  RESULT: NO DATA")

    # Save
    output_path = "/tmp/weight_diff_check_v2.json"
    with open(output_path, "w") as f:
        json.dump({
            "model_a": MODEL_A,
            "model_b": MODEL_B,
            "n_modified": n_modified,
            "n_identical": n_identical,
            "n_missing": n_missing,
            "results": all_results,
        }, f, indent=2)
    print("\n  Saved: {}".format(output_path))
    print("=" * 70)


if __name__ == "__main__":
    main()
