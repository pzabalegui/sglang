path = '/tmp/sglang_steering/python/sglang/srt/model_executor/forward_batch_info.py'
with open(path, 'r') as f:
    content = f.read()

# Replace the projection-based steering with additive
old = '''    # Use FULL representation for projection (hidden_states + residual)
    # hidden_states is MLP delta, residual is accumulated stream
    # The refusal direction lives in the full representation space
    if residual is not None:
        full_repr = hidden_states + residual
    else:
        full_repr = hidden_states

    # Project full representation onto direction: scalar = (h+r) · r\u0302
    proj_scalar = (full_repr * direction).sum(dim=-1, keepdim=True)

    # CLAMPED STEERING: only steer positive projections
    proj_scalar = proj_scalar.clamp(min=0)

    # Subtract the projection from hidden_states only
    # The correction propagates to residual via the next layer's prepare_attn
    effective_scale = steering_config.get_effective_scale(layer_idx)
    modified = hidden_states - effective_scale * proj_scalar * direction'''

new = '''    # ADDITIVE STEERING (CAA - Contrastive Activation Addition)
    # h' = h - alpha * direction (constant shift, no projection)
    # Much more powerful than projective for deeply safety-trained models
    effective_scale = steering_config.get_effective_scale(layer_idx)
    modified = hidden_states - effective_scale * direction'''

if old not in content:
    # The unicode r-hat might be different, try without it
    old2 = content[content.find('    # Use FULL representation'):content.find('    return modified')]
    if old2:
        content = content.replace(old2, new + '\n\n', 1)
        print('[OK] Replaced via substring match')
    else:
        print('ERROR: Cannot find code block')
        import sys; sys.exit(1)
else:
    content = content.replace(old, new, 1)
    print('[OK] Replaced exact match')

with open(path, 'w') as f:
    f.write(content)
print('[OK] Switched to ADDITIVE steering')
