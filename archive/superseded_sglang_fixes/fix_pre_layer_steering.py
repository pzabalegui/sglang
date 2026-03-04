"""Move steering injection BEFORE each layer instead of after.

Current (post-layer):
  h, res = layer(pos, h, batch, res)
  h = apply_steering(h, config, i, residual=res)  # Too late - next layer uses unsteered attention

New (pre-layer - steer residual):
  if steering: res = steer_residual(res, config, i)  # Clean the residual BEFORE layer
  h, res = layer(pos, h, batch, res)

This way, prepare_attn sees a de-refused residual, so attention computes on 
cleaned representations. Much closer to orthogonalization.
"""

path = '/tmp/sglang_steering/python/sglang/srt/models/glm4_moe.py'
with open(path, 'r') as f:
    content = f.read()

# Find the current steering injection (post-layer)
old_block = '''                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                )
                # Apply steering vector if configured
                hidden_states = apply_steering(hidden_states, forward_batch.steering_config, i, residual=residual)'''

# New: steer residual PRE-layer, remove post-layer steering
new_block = '''                # PRE-LAYER steering: clean the residual before the layer processes it
                # This way attention and MLP both see de-refused representations
                if forward_batch.steering_config is not None and residual is not None:
                    from sglang.srt.model_executor.forward_batch_info import apply_steering_to_residual
                    residual = apply_steering_to_residual(residual, forward_batch.steering_config, i)
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                )'''

if old_block not in content:
    print('ERROR: Could not find post-layer steering block')
    # Try to see what's there
    import re
    m = re.search(r'hidden_states, residual = layer\(.*?\)', content, re.DOTALL)
    if m:
        print(f'Found layer call at position {m.start()}: {m.group()[:100]}')
    import sys; sys.exit(1)

content = content.replace(old_block, new_block, 1)

with open(path, 'w') as f:
    f.write(content)
print('[OK] Moved steering to PRE-layer (residual injection)')


# Now add apply_steering_to_residual function to forward_batch_info.py
path2 = '/tmp/sglang_steering/python/sglang/srt/model_executor/forward_batch_info.py'
with open(path2, 'r') as f:
    content2 = f.read()

# Add the new function right after the existing apply_steering function
marker = '''    return modified


class ForwardMode(IntEnum):'''

new_func = '''    return modified


def apply_steering_to_residual(
    residual: torch.Tensor,
    steering_config: Optional[SteeringConfig],
    layer_idx: int,
) -> torch.Tensor:
    """Apply steering to the residual stream BEFORE layer computation.
    
    The residual holds the accumulated representation. By removing the
    refusal direction from the residual before prepare_attn, both the
    attention and MLP see cleaned inputs. This approximates orthogonalization.
    
    Formula: res' = res - scale * (res . r_hat) * r_hat
    """
    if steering_config is None:
        return residual
    if not steering_config.should_apply_to_layer(layer_idx):
        return residual
    
    direction = steering_config.direction.to(
        device=residual.device, dtype=residual.dtype
    )
    direction = direction / direction.norm()
    
    # Project residual onto refusal direction
    proj_scalar = (residual * direction).sum(dim=-1, keepdim=True)
    
    # Remove the refusal component from residual
    effective_scale = steering_config.get_effective_scale(layer_idx)
    return residual - effective_scale * proj_scalar * direction


class ForwardMode(IntEnum):'''

if marker not in content2:
    print('ERROR: Could not find insertion marker in forward_batch_info.py')
    import sys; sys.exit(1)

content2 = content2.replace(marker, new_func, 1)

with open(path2, 'w') as f:
    f.write(content2)
print('[OK] Added apply_steering_to_residual function')
print()
print('Pre-layer steering: modifies residual before prepare_attn')
print('Attention and MLP both see de-refused representations')
