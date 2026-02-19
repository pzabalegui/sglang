path = '/tmp/sglang_steering/python/sglang/srt/model_executor/forward_batch_info.py'
with open(path, 'r') as f:
    content = f.read()

old = '''            elif override_scale is not None:
                # Create modified config with new scale
                ret.steering_config = SteeringConfig(
                    direction=base_config.direction,
                    scale=override_scale,
                    layers=base_config.layers,
                    enabled=base_config.enabled
                )'''

new = '''            elif override_scale is not None:
                # Create modified config with new scale
                # Recompute layer_weights if base has Gaussian kernel
                if base_config.layer_weights is not None:
                    # Scale all layer weights proportionally
                    scale_ratio = override_scale / max(base_config.layer_weights.values())
                    new_weights = {l: w * scale_ratio for l, w in base_config.layer_weights.items()}
                    ret.steering_config = SteeringConfig(
                        direction=base_config.direction,
                        scale=override_scale,
                        layers=base_config.layers,
                        enabled=base_config.enabled,
                        layer_weights=new_weights,
                    )
                else:
                    ret.steering_config = SteeringConfig(
                        direction=base_config.direction,
                        scale=override_scale,
                        layers=base_config.layers,
                        enabled=base_config.enabled,
                    )'''

if old not in content:
    print('ERROR: Could not find override code')
    import sys; sys.exit(1)

content = content.replace(old, new, 1)
with open(path, 'w') as f:
    f.write(content)
print('[OK] Fixed BUG 3: per-request override now preserves Gaussian layer_weights')
