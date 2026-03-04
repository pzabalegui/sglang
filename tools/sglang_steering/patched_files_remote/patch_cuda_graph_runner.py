"""
Patch cuda_graph_runner.py for per-request steering toggle + decode scale override.

During CUDA graph replay, Python forward() does NOT run, so buffer
modifications in glm4_moe.py forward() are ineffective. This patch:
1. Zeros steering buffers BEFORE graph replay when steering_disabled
2. Overrides decode scale buffers when steering_decode_scale_override is set
3. Zeros v4 momentum buffer when steering is disabled
4. Zeros v5 buffers (_v5_sig_tmp, _v5_sig_result, _v5_proj) when steering is disabled
5. Restores all buffers AFTER replay

Supports DAS v1 (post-layer scales), v2 (attn/MLP scales),
v3 (per-layer decode scales), v4 (momentum-adaptive), and v5 (multi-vector WRMD).
"""
import sys

SGLANG_DIR = "/tmp/sglang_steering/python/sglang"
CGR_FILE = f"{SGLANG_DIR}/srt/model_executor/cuda_graph_runner.py"

with open(CGR_FILE, "r") as f:
    content = f.read()

# Check if already has v5+ablit patch
if "_v5_sig_tmp" in content and "_steering_mask" in content:
    print("cuda_graph_runner.py: Already patched (v5 + abliteration mask). Skipping.")
    sys.exit(0)

# Check if has v5 patch but not abliteration mask — upgrade
if "_v5_sig_tmp" in content and "_steering_mask" not in content:
    # Add mask zeroing in the steering_off block (after v5 buffer zeroing)
    v5_zero_marker = """            # v5: zero multi-vector WRMD buffers
            for _buf_name in ('_v5_sig_tmp', '_v5_sig_result', '_v5_proj'):
                if hasattr(_inner, _buf_name) and getattr(_inner, _buf_name) is not None:
                    _steer_restore[_buf_name] = getattr(_inner, _buf_name).clone()
                    getattr(_inner, _buf_name).zero_()"""
    v5_zero_plus_mask = v5_zero_marker + """
            # Abliteration: zero steering mask (controls per-request abliteration)
            if hasattr(_inner, '_steering_mask') and _inner._steering_mask is not None:
                _steer_restore['steering_mask'] = _inner._steering_mask.clone()
                _inner._steering_mask.zero_()"""
    content = content.replace(v5_zero_marker, v5_zero_plus_mask)

    # Add mask restore
    v5_restore_marker = """            # v5: restore multi-vector WRMD buffers
            for _buf_name in ('_v5_sig_tmp', '_v5_sig_result', '_v5_proj'):
                if _buf_name in _steer_restore:
                    getattr(_inner, _buf_name).copy_(_steer_restore[_buf_name])"""
    v5_restore_plus_mask = v5_restore_marker + """
            if 'steering_mask' in _steer_restore:
                _inner._steering_mask.copy_(_steer_restore['steering_mask'])"""
    content = content.replace(v5_restore_marker, v5_restore_plus_mask)

    with open(CGR_FILE, "w") as f:
        f.write(content)
    print("cuda_graph_runner.py v5 -> v5+ablit upgrade complete (added steering mask support).")
    sys.exit(0)

# Check if has v4 patch but not v5 — upgrade v4→v5
if "_steer_momentum" in content and "_v5_sig_tmp" not in content:
    old_marker_v4 = "patched v4)"
    new_marker_v5 = "patched v5)"
    if old_marker_v4 in content:
        content = content.replace(old_marker_v4, new_marker_v5)
        # Add v5 buffer zeroing in the steering_off block (after momentum zero)
        momentum_zero = """            # v4: zero momentum to prevent stale accumulation
            if hasattr(_inner, '_steer_momentum') and _inner._steer_momentum is not None:
                _steer_restore['momentum'] = _inner._steer_momentum.clone()
                _inner._steer_momentum.zero_()"""
        momentum_zero_plus_v5 = momentum_zero + """
            # v5: zero multi-vector WRMD buffers
            for _buf_name in ('_v5_sig_tmp', '_v5_sig_result', '_v5_proj'):
                if hasattr(_inner, _buf_name) and getattr(_inner, _buf_name) is not None:
                    _steer_restore[_buf_name] = getattr(_inner, _buf_name).clone()
                    getattr(_inner, _buf_name).zero_()"""
        content = content.replace(momentum_zero, momentum_zero_plus_v5)
        # Add v5 buffer restore (after momentum restore)
        momentum_restore = """            if 'momentum' in _steer_restore:
                _inner._steer_momentum.copy_(_steer_restore['momentum'])"""
        momentum_restore_plus_v5 = momentum_restore + """
            # v5: restore multi-vector WRMD buffers
            for _buf_name in ('_v5_sig_tmp', '_v5_sig_result', '_v5_proj'):
                if _buf_name in _steer_restore:
                    getattr(_inner, _buf_name).copy_(_steer_restore[_buf_name])"""
        content = content.replace(momentum_restore, momentum_restore_plus_v5)
        with open(CGR_FILE, "w") as f:
            f.write(content)
        print("cuda_graph_runner.py v4 -> v5 upgrade complete (added multi-vector buffer support).")
        sys.exit(0)
    else:
        print("WARNING: v4 patch detected but marker not found. Proceeding with fresh patch.")

# Check if has v2 patch but not v4 — upgrade v2→v5
if "steering_decode_scale_override" in content and "_steer_momentum" not in content:
    old_marker = "patched v2)"
    new_marker = "patched v5)"
    if old_marker in content:
        content = content.replace(old_marker, new_marker)
        # Add momentum zeroing + v5 buffers in the steering_off block (after mlp_scales zero)
        mlp_zero = """            if hasattr(_inner, '_steering_mlp_scales') and _inner._steering_mlp_scales is not None:
                _steer_restore['mlp_scales'] = _inner._steering_mlp_scales.clone()
                _inner._steering_mlp_scales.zero_()"""
        mlp_zero_plus_momentum_v5 = mlp_zero + """
            # v4: zero momentum to prevent stale accumulation
            if hasattr(_inner, '_steer_momentum') and _inner._steer_momentum is not None:
                _steer_restore['momentum'] = _inner._steer_momentum.clone()
                _inner._steer_momentum.zero_()
            # v5: zero multi-vector WRMD buffers
            for _buf_name in ('_v5_sig_tmp', '_v5_sig_result', '_v5_proj'):
                if hasattr(_inner, _buf_name) and getattr(_inner, _buf_name) is not None:
                    _steer_restore[_buf_name] = getattr(_inner, _buf_name).clone()
                    getattr(_inner, _buf_name).zero_()"""
        content = content.replace(mlp_zero, mlp_zero_plus_momentum_v5)
        # Add momentum + v5 restore (after mlp_scales restore)
        mlp_restore = """            if 'mlp_scales' in _steer_restore:
                _inner._steering_mlp_scales.copy_(_steer_restore['mlp_scales'])"""
        mlp_restore_plus_momentum_v5 = mlp_restore + """
            if 'momentum' in _steer_restore:
                _inner._steer_momentum.copy_(_steer_restore['momentum'])
            # v5: restore multi-vector WRMD buffers
            for _buf_name in ('_v5_sig_tmp', '_v5_sig_result', '_v5_proj'):
                if _buf_name in _steer_restore:
                    getattr(_inner, _buf_name).copy_(_steer_restore[_buf_name])"""
        content = content.replace(mlp_restore, mlp_restore_plus_momentum_v5)
        with open(CGR_FILE, "w") as f:
            f.write(content)
        print("cuda_graph_runner.py v2 -> v5 upgrade complete (added momentum + multi-vector buffer support).")
        sys.exit(0)
    else:
        print("WARNING: v2 patch detected but marker not found. Proceeding with fresh patch.")
        # Fall through to fresh patch below

# Check if has v1 patch — upgrade
if "Per-request steering toggle" in content:
    # Replace the entire steering block around replay
    # Find the v1 block and replace with v2
    old_block_start = "        # Per-request steering toggle (patched): zero all steering scales before replay"
    old_block_end = "        output = self.output_buffers[graph_key]"

    idx_start = content.find(old_block_start)
    idx_end = content.find(old_block_end, idx_start)
    if idx_start == -1 or idx_end == -1:
        print("WARNING: Could not find v1 steering block boundaries. Adding v2 fresh.")
    else:
        # Replace from start to end (inclusive of output line)
        old_section = content[idx_start:idx_end + len(old_block_end)]
        new_section = """        # Per-request steering toggle + decode scale override + v4 momentum + v5 buffers (patched v5)
        # Buffers are on model.model (Glm4MoeModel), not model (Glm4MoeForCausalLM)
        _steer_restore = {}
        _inner = getattr(self.model_runner.model, 'model', self.model_runner.model)
        _steering_off = getattr(forward_batch, 'steering_disabled', False)
        _ds_override = getattr(forward_batch, 'steering_decode_scale_override', None)

        if _steering_off:
            # Zero all steering scales (full disable)
            if hasattr(_inner, '_steer_dec_scale') and _inner._steer_dec_scale is not None:
                _steer_restore['dec_scale'] = _inner._steer_dec_scale.item()
                _inner._steer_dec_scale.fill_(0.0)
            if hasattr(_inner, '_steer_dec_scales') and _inner._steer_dec_scales is not None:
                _steer_restore['dec_scales'] = _inner._steer_dec_scales.clone()
                _inner._steer_dec_scales.zero_()
            if hasattr(_inner, '_steering_scales') and _inner._steering_scales is not None:
                _steer_restore['scales'] = _inner._steering_scales.clone()
                _inner._steering_scales.zero_()
            if hasattr(_inner, '_steering_attn_scales') and _inner._steering_attn_scales is not None:
                _steer_restore['attn_scales'] = _inner._steering_attn_scales.clone()
                _inner._steering_attn_scales.zero_()
            if hasattr(_inner, '_steering_mlp_scales') and _inner._steering_mlp_scales is not None:
                _steer_restore['mlp_scales'] = _inner._steering_mlp_scales.clone()
                _inner._steering_mlp_scales.zero_()
            # v4: zero momentum to prevent stale accumulation
            if hasattr(_inner, '_steer_momentum') and _inner._steer_momentum is not None:
                _steer_restore['momentum'] = _inner._steer_momentum.clone()
                _inner._steer_momentum.zero_()
            # v5: zero multi-vector WRMD buffers
            for _buf_name in ('_v5_sig_tmp', '_v5_sig_result', '_v5_proj'):
                if hasattr(_inner, _buf_name) and getattr(_inner, _buf_name) is not None:
                    _steer_restore[_buf_name] = getattr(_inner, _buf_name).clone()
                    getattr(_inner, _buf_name).zero_()
            # Abliteration: zero steering mask (controls per-request abliteration)
            if hasattr(_inner, '_steering_mask') and _inner._steering_mask is not None:
                _steer_restore['steering_mask'] = _inner._steering_mask.clone()
                _inner._steering_mask.zero_()
        elif _ds_override is not None:
            # Per-request decode scale override (not full disable)
            if hasattr(_inner, '_steer_dec_scale') and _inner._steer_dec_scale is not None:
                _steer_restore['dec_scale'] = _inner._steer_dec_scale.item()
                _old_global = _steer_restore['dec_scale']
                _inner._steer_dec_scale.fill_(float(_ds_override))
                # Also scale per-layer decode scales proportionally
                if hasattr(_inner, '_steer_dec_scales') and _inner._steer_dec_scales is not None and _old_global > 0:
                    _steer_restore['dec_scales'] = _inner._steer_dec_scales.clone()
                    _ratio = float(_ds_override) / _old_global
                    _inner._steer_dec_scales.mul_(_ratio)

        self.graphs[graph_key].replay()

        # Restore all steering scales after replay
        if _steer_restore:
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
            if 'momentum' in _steer_restore:
                _inner._steer_momentum.copy_(_steer_restore['momentum'])
            # v5: restore multi-vector WRMD buffers
            for _buf_name in ('_v5_sig_tmp', '_v5_sig_result', '_v5_proj'):
                if _buf_name in _steer_restore:
                    getattr(_inner, _buf_name).copy_(_steer_restore[_buf_name])
            if 'steering_mask' in _steer_restore:
                _inner._steering_mask.copy_(_steer_restore['steering_mask'])

        output = self.output_buffers[graph_key]"""

        content = content.replace(old_section, new_section, 1)
        with open(CGR_FILE, "w") as f:
            f.write(content)
        print("cuda_graph_runner.py v1 → v2 upgrade complete.")
        sys.exit(0)

# Fresh patch: find the replay point
target = "        self.graphs[graph_key].replay()\n\n        output = self.output_buffers[graph_key]"
if target not in content:
    target = "        self.graphs[graph_key].replay()\n        output = self.output_buffers[graph_key]"
    if target not in content:
        print("ERROR: Could not find 'self.graphs[graph_key].replay()' followed by 'output = self.output_buffers[graph_key]'")
        sys.exit(1)

replacement = """        # Per-request steering toggle + decode scale override + v4/v5 (patched v5)
        # Buffers are on model.model (Glm4MoeModel), not model (Glm4MoeForCausalLM)
        _steer_restore = {}
        _inner = getattr(self.model_runner.model, 'model', self.model_runner.model)
        _steering_off = getattr(forward_batch, 'steering_disabled', False)
        _ds_override = getattr(forward_batch, 'steering_decode_scale_override', None)

        if _steering_off:
            # Zero all steering scales (full disable)
            if hasattr(_inner, '_steer_dec_scale') and _inner._steer_dec_scale is not None:
                _steer_restore['dec_scale'] = _inner._steer_dec_scale.item()
                _inner._steer_dec_scale.fill_(0.0)
            if hasattr(_inner, '_steer_dec_scales') and _inner._steer_dec_scales is not None:
                _steer_restore['dec_scales'] = _inner._steer_dec_scales.clone()
                _inner._steer_dec_scales.zero_()
            if hasattr(_inner, '_steering_scales') and _inner._steering_scales is not None:
                _steer_restore['scales'] = _inner._steering_scales.clone()
                _inner._steering_scales.zero_()
            if hasattr(_inner, '_steering_attn_scales') and _inner._steering_attn_scales is not None:
                _steer_restore['attn_scales'] = _inner._steering_attn_scales.clone()
                _inner._steering_attn_scales.zero_()
            if hasattr(_inner, '_steering_mlp_scales') and _inner._steering_mlp_scales is not None:
                _steer_restore['mlp_scales'] = _inner._steering_mlp_scales.clone()
                _inner._steering_mlp_scales.zero_()
            # v4: zero momentum to prevent stale accumulation
            if hasattr(_inner, '_steer_momentum') and _inner._steer_momentum is not None:
                _steer_restore['momentum'] = _inner._steer_momentum.clone()
                _inner._steer_momentum.zero_()
            # v5: zero multi-vector WRMD buffers
            for _buf_name in ('_v5_sig_tmp', '_v5_sig_result', '_v5_proj'):
                if hasattr(_inner, _buf_name) and getattr(_inner, _buf_name) is not None:
                    _steer_restore[_buf_name] = getattr(_inner, _buf_name).clone()
                    getattr(_inner, _buf_name).zero_()
            # Abliteration: zero steering mask (controls per-request abliteration)
            if hasattr(_inner, '_steering_mask') and _inner._steering_mask is not None:
                _steer_restore['steering_mask'] = _inner._steering_mask.clone()
                _inner._steering_mask.zero_()
        elif _ds_override is not None:
            # Per-request decode scale override (not full disable)
            if hasattr(_inner, '_steer_dec_scale') and _inner._steer_dec_scale is not None:
                _steer_restore['dec_scale'] = _inner._steer_dec_scale.item()
                _old_global = _steer_restore['dec_scale']
                _inner._steer_dec_scale.fill_(float(_ds_override))
                if hasattr(_inner, '_steer_dec_scales') and _inner._steer_dec_scales is not None and _old_global > 0:
                    _steer_restore['dec_scales'] = _inner._steer_dec_scales.clone()
                    _ratio = float(_ds_override) / _old_global
                    _inner._steer_dec_scales.mul_(_ratio)

        self.graphs[graph_key].replay()

        # Restore all steering scales after replay
        if _steer_restore:
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
            if 'momentum' in _steer_restore:
                _inner._steer_momentum.copy_(_steer_restore['momentum'])
            # v5: restore multi-vector WRMD buffers
            for _buf_name in ('_v5_sig_tmp', '_v5_sig_result', '_v5_proj'):
                if _buf_name in _steer_restore:
                    getattr(_inner, _buf_name).copy_(_steer_restore[_buf_name])
            if 'steering_mask' in _steer_restore:
                _inner._steering_mask.copy_(_steer_restore['steering_mask'])

        output = self.output_buffers[graph_key]"""

content = content.replace(target, replacement, 1)
print("Added steering toggle + decode_scale_override around graph replay (v1 + v2 + v3 + v4 + v5)")

with open(CGR_FILE, "w") as f:
    f.write(content)

print("cuda_graph_runner.py patch complete.")
