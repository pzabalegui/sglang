"""Inspect MoE layer parameters to understand weight structure."""
import glob
from safetensors import safe_open

# Layer 47 is MoE - find which shard has it
files = sorted(glob.glob("/tmp/GLM-4.7-FP8/model-*.safetensors"))
for f in files:
    with safe_open(f, framework="pt", device="cpu") as sf:
        keys = list(sf.keys())
        layer47_keys = [k for k in keys if "layers.47." in k]
        if layer47_keys:
            print(f"Found layer 47 in: {f}")
            for k in sorted(layer47_keys):
                tensor = sf.get_tensor(k)
                print(f"  {k}: dtype={tensor.dtype}, shape={tensor.shape}")
            break
