"""Patch SGLang's glm4_moe.py to add activation capture mode.

This adds a file-triggered capture mechanism that saves per-layer hidden states
during prefill, alongside the existing steering injection point.

When /tmp/capture_config.json exists and is enabled, the forward loop saves
hidden_states[-1, :] (last token) at each specified layer to /tmp/captures/.
"""
import sys

SGLANG_DIR = "/tmp/sglang_steering/python/sglang/srt"

# ============================================================
# 1. Patch glm4_moe.py: Add capture imports and function
# ============================================================
path = f"{SGLANG_DIR}/models/glm4_moe.py"
with open(path, "r") as f:
    content = f.read()

# Add capture code AFTER the existing imports
# Find the apply_steering import line
# Find the line that imports apply_steering (may have other imports on same line)
import_marker = None
for line in content.split('\n'):
    if 'import' in line and 'apply_steering' in line and 'forward_batch_info' in line:
        import_marker = line.strip()
        break

if import_marker is None:
    print(f"ERROR: Could not find apply_steering import in glm4_moe.py")
    sys.exit(1)

print(f"  Found import marker: {import_marker}")

capture_code = import_marker + '''

# ============================================================
# ACTIVATION CAPTURE for refusal direction extraction
# ============================================================
import json as _json
import os as _os

_CAPTURE_STORE = {}
_CAPTURE_COUNTER = [0]
_CAPTURE_CONFIG_CACHE = [None, 0]  # [config, last_check_time]

def _get_capture_config():
    """Read capture config with 1-second cache."""
    import time
    now = time.time()
    if now - _CAPTURE_CONFIG_CACHE[1] < 1.0 and _CAPTURE_CONFIG_CACHE[0] is not None:
        return _CAPTURE_CONFIG_CACHE[0]
    _CAPTURE_CONFIG_CACHE[1] = now

    config_path = "/tmp/capture_config.json"
    if not _os.path.exists(config_path):
        _CAPTURE_CONFIG_CACHE[0] = None
        return None
    try:
        with open(config_path, "r") as f:
            cfg = _json.load(f)
        if not cfg.get("enabled", False):
            _CAPTURE_CONFIG_CACHE[0] = None
            return None
        _CAPTURE_CONFIG_CACHE[0] = cfg
        return cfg
    except Exception:
        _CAPTURE_CONFIG_CACHE[0] = None
        return None


def _maybe_capture(hidden_states, layer_idx, forward_batch, n_layers=92):
    """Capture hidden states during prefill for refusal direction extraction."""
    import torch

    cfg = _get_capture_config()
    if cfg is None:
        return

    # Only capture during prefill (not decode)
    # During decode, hidden_states has very few tokens (usually 1 per request)
    # During prefill, it has many tokens (full prompt)
    if hidden_states.shape[0] <= 2:
        return

    # Only capture on rank 0 (all ranks have identical hidden_states after allreduce)
    try:
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass

    target_layers = cfg.get("layers", list(range(n_layers)))
    if layer_idx not in target_layers:
        return

    # Save last token's hidden state (where refusal signal is strongest)
    _CAPTURE_STORE[layer_idx] = hidden_states[-1, :].detach().cpu().to(torch.float32)

    # When we reach the last target layer, save everything to disk
    max_target = max(target_layers)
    if layer_idx == max_target:
        save_dir = cfg.get("save_dir", "/tmp/captures")
        _os.makedirs(save_dir, exist_ok=True)
        sample_id = _CAPTURE_COUNTER[0]
        save_path = _os.path.join(save_dir, f"sample_{sample_id}.pt")
        torch.save(dict(_CAPTURE_STORE), save_path)
        _CAPTURE_STORE.clear()
        _CAPTURE_COUNTER[0] += 1'''

content = content.replace(import_marker, capture_code, 1)

# Now add _maybe_capture call in the forward loop, AFTER apply_steering
# Find the steering call
steering_call = "hidden_states = apply_steering(hidden_states, forward_batch.steering_config, i)"
if steering_call not in content:
    print(f"ERROR: Could not find apply_steering call in forward loop")
    sys.exit(1)

capture_call = f"""{steering_call}
                _maybe_capture(hidden_states, i, forward_batch, n_layers=len(self.layers))"""

content = content.replace(steering_call, capture_call, 1)

with open(path, "w") as f:
    f.write(content)
print(f"[OK] Patched {path} with activation capture")

print("\n=== Capture patch applied successfully ===")
print("Usage:")
print("  1. Start SGLang normally (no steering vector needed)")
print("  2. Create /tmp/capture_config.json:")
print('     {"enabled": true, "layers": [30,32,...,62], "save_dir": "/tmp/captures"}')
print("  3. Send prompts via API - activations saved to /tmp/captures/sample_N.pt")
print("  4. Delete /tmp/capture_config.json to disable capture")
