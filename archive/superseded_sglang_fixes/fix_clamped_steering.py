"""Fix steering to use directional clamping.

Problem: Most tokens have NEGATIVE projection onto refusal direction.
When we subtract scale * negative * direction, we ADD refusal to these tokens.
This corrupts harmless tokens and causes garbled output at higher scales.

Fix: Only steer when projection is positive (token is refusal-aligned).
h' = h - scale * max(0, h.r) * r

This way:
- Refusal-aligned tokens (proj > 0): refusal gets removed
- Normal tokens (proj < 0): no modification
"""

path = '/tmp/sglang_steering/python/sglang/srt/model_executor/forward_batch_info.py'
with open(path, 'r') as f:
    content = f.read()

# Fix the main apply_steering function - change proj computation to clamped
old_proj = '''    # Project full representation onto direction: scalar = (h+r) . r_hat
    proj_scalar = (full_repr * direction).sum(dim=-1, keepdim=True)

    # Subtract the projection from hidden_states only
    # The correction propagates to residual via the next layer's prepare_attn
    effective_scale = steering_config.get_effective_scale(layer_idx)
    modified = hidden_states - effective_scale * proj_scalar * direction'''

new_proj = '''    # Project full representation onto direction: scalar = (h+r) . r_hat
    proj_scalar = (full_repr * direction).sum(dim=-1, keepdim=True)

    # CLAMPED STEERING: only steer tokens with positive projection (refusal-aligned)
    # Tokens with negative projection (normal) are left unchanged
    # This prevents adding refusal direction to harmless tokens
    proj_scalar = proj_scalar.clamp(min=0)

    # Subtract the projection from hidden_states only
    # The correction propagates to residual via the next layer's prepare_attn
    effective_scale = steering_config.get_effective_scale(layer_idx)
    modified = hidden_states - effective_scale * proj_scalar * direction'''

if old_proj not in content:
    print('ERROR: Could not find projection code in apply_steering')
    # Try to find the proj_scalar line
    if 'proj_scalar = (full_repr * direction)' in content:
        print('Found proj_scalar line, trying flexible match')
        import re
        content = re.sub(
            r'(    proj_scalar = \(full_repr \* direction\)\.sum\(dim=-1, keepdim=True\))\n',
            r'\1\n\n    # CLAMPED STEERING: only steer positive projections\n    proj_scalar = proj_scalar.clamp(min=0)\n',
            content,
            count=1
        )
        print('[OK] Applied clamping via regex')
    else:
        print('ERROR: Cannot find proj_scalar computation at all')
        import sys; sys.exit(1)
else:
    content = content.replace(old_proj, new_proj, 1)
    print('[OK] Applied clamped steering to apply_steering()')

# Also fix the pre-layer steering function if it exists
old_pre = '''    # Project residual onto refusal direction
    proj_scalar = (residual * direction).sum(dim=-1, keepdim=True)
    
    # Remove the refusal component from residual
    effective_scale = steering_config.get_effective_scale(layer_idx)
    return residual - effective_scale * proj_scalar * direction'''

new_pre = '''    # Project residual onto refusal direction
    proj_scalar = (residual * direction).sum(dim=-1, keepdim=True)
    
    # CLAMPED: only steer tokens with positive projection (refusal-aligned)
    proj_scalar = proj_scalar.clamp(min=0)
    
    # Remove the refusal component from residual
    effective_scale = steering_config.get_effective_scale(layer_idx)
    return residual - effective_scale * proj_scalar * direction'''

if old_pre in content:
    content = content.replace(old_pre, new_pre, 1)
    print('[OK] Applied clamped steering to apply_steering_to_residual()')
else:
    print('[SKIP] apply_steering_to_residual not found or already modified')

with open(path, 'w') as f:
    f.write(content)

print('\nClamped steering applied: max(0, proj) instead of proj')
print('This prevents adding refusal to non-refusal-aligned tokens')
