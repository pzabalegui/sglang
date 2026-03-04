"""
Patch glm4_moe.py for per-request steering toggle and decode scale override.

Works with DAS v1, v2, and v3 code. Adds:
1. Prefill: skip steering when forward_batch.steering_disabled is True
2. Decode: zero _steer_dec_scale when steering_disabled is True
3. Decode: per-request decode_scale override via steering_decode_scale_override

Note: This patch is for the per-request toggle ONLY. The main steering
code (prefill projection, decode clamped projective, v2 attn/MLP) must
already be present in glm4_moe.py from the base steering patch.
"""
import sys

SGLANG_DIR = "/tmp/sglang_steering/python/sglang"
GLM_FILE = f"{SGLANG_DIR}/srt/models/glm4_moe.py"

with open(GLM_FILE, "r") as f:
    content = f.read()

# Check if already has v2 patch (decode_scale_override)
if "steering_decode_scale_override" in content:
    print("glm4_moe.py: Already has v2 toggle (decode_scale_override). Skipping.")
    sys.exit(0)

# Check if has v1 patch (steering_disabled only) — upgrade to v2
if "steering_disabled" in content:
    # Add decode_scale_override logic after the _has_decode_steering block
    # Find the decode steering setup section
    old_decode_setup = """        _has_multi_layer_decode = getattr(self, '_has_multi_layer_decode', False)

        aux_hidden_states = []"""

    if old_decode_setup in content:
        new_decode_setup = """        _has_multi_layer_decode = getattr(self, '_has_multi_layer_decode', False)

        # Per-request decode scale override (CUDA-graph safe: modifies buffer before layer loop)
        _saved_dec_scale_for_override = None
        _saved_dec_scales_for_override = None
        _ds_override = getattr(forward_batch, 'steering_decode_scale_override', None)
        if _has_decode_steering and _ds_override is not None:
            if _has_multi_layer_decode:
                # Scale proportionally: override / global_default ratio applied to all layers
                _old_global = self._steer_dec_scale.item()
                if _old_global > 0:
                    _ratio = float(_ds_override) / _old_global
                    _saved_dec_scales_for_override = self._steer_dec_scales.clone()
                    self._steer_dec_scales.mul_(_ratio)
                _saved_dec_scale_for_override = _old_global
                self._steer_dec_scale.fill_(float(_ds_override))
            else:
                _saved_dec_scale_for_override = self._steer_dec_scale.item()
                self._steer_dec_scale.fill_(float(_ds_override))

        aux_hidden_states = []"""

        content = content.replace(old_decode_setup, new_decode_setup, 1)
        print("Added decode_scale_override logic before layer loop")
    else:
        print("WARNING: Could not find _has_multi_layer_decode + aux_hidden_states block")

    # Add restore after layer loop (before normal_end_layer check)
    old_restore = "        if normal_end_layer != self.end_layer:"
    new_restore = """        # Restore decode scale after per-request override
        if _saved_dec_scale_for_override is not None:
            self._steer_dec_scale.fill_(_saved_dec_scale_for_override)
        if _saved_dec_scales_for_override is not None:
            self._steer_dec_scales.copy_(_saved_dec_scales_for_override)

        if normal_end_layer != self.end_layer:"""

    # Only add if not already present
    if "_saved_dec_scale_for_override" not in content:
        content = content.replace(old_restore, new_restore, 1)
        print("Added decode_scale restore after layer loop")

    with open(GLM_FILE, "w") as f:
        f.write(content)
    print("glm4_moe.py v1 → v2 toggle upgrade complete.")
    sys.exit(0)

# Fresh patch: add both steering_disabled and decode_scale_override

# Part 1: Add steering_disabled check to prefill steering condition
old_prefill = """        _has_steering = (
            hasattr(self, '_steering_dir')
            and self._steering_dir is not None
            and _is_prefill
        )"""

if old_prefill not in content:
    # Try v1 pattern
    old_prefill = """        _has_steering = (
            hasattr(self, '_steering_dir')
            and self._steering_dir is not None
            and not forward_batch.forward_mode.is_decode()
        )"""

if old_prefill in content:
    new_prefill = old_prefill.rstrip(")") + "\n            and not getattr(forward_batch, 'steering_disabled', False)\n        )"
    content = content.replace(old_prefill, new_prefill, 1)
    print("Added steering_disabled check to prefill steering")
else:
    print("WARNING: Could not find _has_steering block for prefill")

# Part 1b: Also disable v2 steering context when steering_disabled
old_v2 = "        _steering_v2 = getattr(self, '_steering_v2', False) and _is_prefill"
if old_v2 in content:
    new_v2 = "        _steering_v2 = getattr(self, '_steering_v2', False) and _is_prefill and not getattr(forward_batch, 'steering_disabled', False)"
    content = content.replace(old_v2, new_v2, 1)
    print("Added steering_disabled check to v2 steering context")

# Part 2: Add _steering_off variable + decode control
old_decode = """        _has_decode_steering = (
            forward_batch.forward_mode.is_decode()
            and getattr(self, '_steer_dec_scale', None) is not None
        )
        _decode_peak_layer = getattr(self, '_decode_steer_peak_layer', -1)"""

if old_decode in content:
    new_decode = """        _has_decode_steering = (
            forward_batch.forward_mode.is_decode()
            and getattr(self, '_steer_dec_scale', None) is not None
            and not getattr(forward_batch, 'steering_disabled', False)
        )
        _decode_peak_layer = getattr(self, '_decode_steer_peak_layer', -1)"""
    content = content.replace(old_decode, new_decode, 1)
    print("Added steering_disabled check to decode steering")
else:
    print("WARNING: Could not find _has_decode_steering block")

# Part 3: Add decode_scale_override before layer loop
old_loop = """        _has_multi_layer_decode = getattr(self, '_has_multi_layer_decode', False)

        aux_hidden_states = []"""

if old_loop in content:
    new_loop = """        _has_multi_layer_decode = getattr(self, '_has_multi_layer_decode', False)

        # Per-request decode scale override (CUDA-graph safe: modifies buffer before layer loop)
        _saved_dec_scale_for_override = None
        _saved_dec_scales_for_override = None
        _ds_override = getattr(forward_batch, 'steering_decode_scale_override', None)
        if _has_decode_steering and _ds_override is not None:
            if _has_multi_layer_decode:
                _old_global = self._steer_dec_scale.item()
                if _old_global > 0:
                    _ratio = float(_ds_override) / _old_global
                    _saved_dec_scales_for_override = self._steer_dec_scales.clone()
                    self._steer_dec_scales.mul_(_ratio)
                _saved_dec_scale_for_override = _old_global
                self._steer_dec_scale.fill_(float(_ds_override))
            else:
                _saved_dec_scale_for_override = self._steer_dec_scale.item()
                self._steer_dec_scale.fill_(float(_ds_override))

        aux_hidden_states = []"""
    content = content.replace(old_loop, new_loop, 1)
    print("Added decode_scale_override logic before layer loop")
else:
    print("WARNING: Could not find layer loop start for decode_scale_override")

# Part 4: Add scale restoration after the layer loop
old_restore = "        if normal_end_layer != self.end_layer:"
if old_restore in content:
    new_restore = """        # Restore decode scale if it was overridden
        if _saved_dec_scale_for_override is not None:
            self._steer_dec_scale.fill_(_saved_dec_scale_for_override)
        if _saved_dec_scales_for_override is not None:
            self._steer_dec_scales.copy_(_saved_dec_scales_for_override)

        if normal_end_layer != self.end_layer:"""
    content = content.replace(old_restore, new_restore, 1)
    print("Added decode scale restoration after layer loop")
else:
    print("WARNING: Could not find restoration point")

with open(GLM_FILE, "w") as f:
    f.write(content)

print("glm4_moe.py per-request toggle + decode_scale_override patch complete.")
