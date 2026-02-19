"""Inspect model parameters to understand FP8 weight structure."""
import torch
import sys
import glob

from safetensors import safe_open

files = sorted(glob.glob("/tmp/GLM-4.7-FP8/model-*.safetensors"))[:1]
if not files:
    print("No safetensors files found")
    sys.exit(1)

print(f"Inspecting: {files[0]}")
with safe_open(files[0], framework="pt", device="cpu") as f:
    keys = list(f.keys())
    # Find attention-related weights for layer 47 or layer 62
    for k in sorted(keys):
        if any(f"layers.{l}." in k for l in [0, 1, 47, 62]):
            tensor = f.get_tensor(k)
            print(f"  {k}: dtype={tensor.dtype}, shape={tensor.shape}")
