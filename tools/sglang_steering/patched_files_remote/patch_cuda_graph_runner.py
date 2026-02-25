"""
Patch cuda_graph_runner.py to support per-request steering toggle.

During CUDA graph replay, Python forward() does NOT run, so buffer
modifications in glm4_moe.py forward() are ineffective. This patch
zeros the steering buffers BEFORE graph replay and restores them after,
enabling per-request steering disable even during CUDA graph execution.

Supports DAS v1 (post-layer scales), v2 (attn/MLP scales), and
v3 (per-layer decode scales via _steer_dec_scales).
"""
import sys

SGLANG_DIR = "/tmp/sglang_steering/python/sglang"
CGR_FILE = f"{SGLANG_DIR}/srt/model_executor/cuda_graph_runner.py"

with open(CGR_FILE, "r") as f:
    content = f.read()

# Check if already patched
if "Per-request steering toggle" in content:
    print("cuda_graph_runner.py: Already patched. Skipping.")
    sys.exit(0)

# Find the replay point: self.graphs[graph_key].replay()
# Insert buffer zeroing before and restoration after
target = "        self.graphs[graph_key].replay()\n\n        output = self.output_buffers[graph_key]"
if target not in content:
    target = "        self.graphs[graph_key].replay()\n        output = self.output_buffers[graph_key]"
    if target not in content:
        print("ERROR: Could not find 'self.graphs[graph_key].replay()' followed by 'output = self.output_buffers[graph_key]'")
        sys.exit(1)

replacement = """        # Per-request steering toggle (patched): zero all steering scales before replay
        # Buffers are on model.model (Glm4MoeModel), not model (Glm4MoeForCausalLM)
        _steer_restore = {}
        if getattr(forward_batch, 'steering_disabled', False):
            _inner = getattr(self.model_runner.model, 'model', self.model_runner.model)
            # Zero decode scale (global toggle)
            if hasattr(_inner, '_steer_dec_scale') and _inner._steer_dec_scale is not None:
                _steer_restore['dec_scale'] = _inner._steer_dec_scale.item()
                _inner._steer_dec_scale.fill_(0.0)
            # Zero per-layer decode scales (v3)
            if hasattr(_inner, '_steer_dec_scales') and _inner._steer_dec_scales is not None:
                _steer_restore['dec_scales'] = _inner._steer_dec_scales.clone()
                _inner._steer_dec_scales.zero_()
            # Zero post-layer scales (v1)
            if hasattr(_inner, '_steering_scales') and _inner._steering_scales is not None:
                _steer_restore['scales'] = _inner._steering_scales.clone()
                _inner._steering_scales.zero_()
            # Zero attn scales (v2)
            if hasattr(_inner, '_steering_attn_scales') and _inner._steering_attn_scales is not None:
                _steer_restore['attn_scales'] = _inner._steering_attn_scales.clone()
                _inner._steering_attn_scales.zero_()
            # Zero MLP scales (v2)
            if hasattr(_inner, '_steering_mlp_scales') and _inner._steering_mlp_scales is not None:
                _steer_restore['mlp_scales'] = _inner._steering_mlp_scales.clone()
                _inner._steering_mlp_scales.zero_()

        self.graphs[graph_key].replay()

        # Restore all steering scales after replay
        if _steer_restore:
            _inner = getattr(self.model_runner.model, 'model', self.model_runner.model)
            if 'dec_scale' in _steer_restore:
                _inner._steer_dec_scale.fill_(_steer_restore['dec_scale'])
            if 'dec_scales' in _steer_restore:
                _inner._steer_dec_scales.copy_(_steer_restore['dec_scales'])
            if 'scales' in _steer_restore:
                _inner._steering_scales.copy_(_steer_restore['scales'])
            if 'attn_scales' in _steer_restore:
                _inner._steering_attn_scales.copy_(_steer_restore['attn_scales'])
            if 'mlp_scales' in _steer_restore:
                _inner._steering_mlp_scales.copy_(_steer_restore['mlp_scales'])

        output = self.output_buffers[graph_key]"""

content = content.replace(target, replacement, 1)
print("Added steering buffer zeroing around graph replay (v1 + v2)")

with open(CGR_FILE, "w") as f:
    f.write(content)

print("cuda_graph_runner.py patch complete.")
