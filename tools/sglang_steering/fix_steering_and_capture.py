"""Fix two critical bugs in SGLang steering implementation:

BUG 1: _maybe_capture captures hidden_states (MLP delta) instead of full representation
       (hidden_states + residual). This means the extracted refusal vector is in the wrong space.

BUG 2: apply_steering projects hidden_states (MLP delta) onto the refusal direction.
       It should project the FULL representation (hidden_states + residual) to correctly
       measure the refusal signal, then apply the correction to hidden_states only.

After fix:
- capture: saves (hidden_states + residual)[-1, :] = full layer output representation
- apply_steering: computes projection on (hidden_states + residual), modifies hidden_states
"""
import sys

SGLANG_DIR = "/tmp/sglang_steering/python/sglang/srt"

# ============================================================
# 1. Fix apply_steering in forward_batch_info.py
# ============================================================
path_fbi = f"{SGLANG_DIR}/model_executor/forward_batch_info.py"
with open(path_fbi, "r") as f:
    content_fbi = f.read()

# Find the apply_steering function signature and replace it
old_signature = """def apply_steering(
    hidden_states: torch.Tensor,
    steering_config: Optional[SteeringConfig],
    layer_idx: int,
) -> torch.Tensor:"""

new_signature = """def apply_steering(
    hidden_states: torch.Tensor,
    steering_config: Optional[SteeringConfig],
    layer_idx: int,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:"""

if old_signature not in content_fbi:
    print(f"ERROR: Could not find apply_steering signature in forward_batch_info.py")
    sys.exit(1)

content_fbi = content_fbi.replace(old_signature, new_signature, 1)

# Now fix the projection computation to use full representation
# Find the projection computation block
old_proj = """    # Project hidden states onto direction: scalar = h · r̂
    proj_scalar = (hidden_states * direction).sum(dim=-1, keepdim=True)

    # Subtract the projection: h' = h - effective_scale * (h·r̂) * r̂
    effective_scale = steering_config.get_effective_scale(layer_idx)
    modified = hidden_states - effective_scale * proj_scalar * direction"""

new_proj = """    # Use FULL representation for projection (hidden_states + residual)
    # hidden_states is MLP delta, residual is accumulated stream
    # The refusal direction lives in the full representation space
    if residual is not None:
        full_repr = hidden_states + residual
    else:
        full_repr = hidden_states

    # Project full representation onto direction: scalar = (h+r) · r̂
    proj_scalar = (full_repr * direction).sum(dim=-1, keepdim=True)

    # Subtract the projection from hidden_states only
    # The correction propagates to residual via the next layer's prepare_attn
    effective_scale = steering_config.get_effective_scale(layer_idx)
    modified = hidden_states - effective_scale * proj_scalar * direction"""

if old_proj not in content_fbi:
    print(f"ERROR: Could not find projection computation in forward_batch_info.py")
    print("Looking for the projection code pattern...")
    # Try a more flexible match
    import re
    pattern = r"proj_scalar = \(hidden_states \* direction\)\.sum\(dim=-1, keepdim=True\)"
    if re.search(pattern, content_fbi):
        print("Found proj_scalar line, attempting targeted fix...")
        content_fbi = re.sub(
            r"    # Project hidden states onto direction:.*?\n"
            r"    proj_scalar = \(hidden_states \* direction\)\.sum\(dim=-1, keepdim=True\)\n\n"
            r"    # Subtract the projection:.*?\n"
            r"    effective_scale = steering_config\.get_effective_scale\(layer_idx\)\n"
            r"    modified = hidden_states - effective_scale \* proj_scalar \* direction",
            new_proj.lstrip(),
            content_fbi,
            count=1,
            flags=re.DOTALL
        )
    else:
        print("ERROR: Could not find projection code at all")
        sys.exit(1)
else:
    content_fbi = content_fbi.replace(old_proj, new_proj, 1)

# Also fix the debug logging to show full repr projection
old_debug_proj = "        proj = (hidden_states * direction).sum(dim=-1, keepdim=True)"
new_debug_proj = """        if residual is not None:
            debug_full = hidden_states + residual
        else:
            debug_full = hidden_states
        proj = (debug_full * direction).sum(dim=-1, keepdim=True)"""

if old_debug_proj in content_fbi:
    content_fbi = content_fbi.replace(old_debug_proj, new_debug_proj, 1)
    print("  [OK] Fixed debug projection too")

with open(path_fbi, "w") as f:
    f.write(content_fbi)
print(f"[OK] Fixed apply_steering in {path_fbi}")


# ============================================================
# 2. Fix the forward loop call in glm4_moe.py
# ============================================================
path_model = f"{SGLANG_DIR}/models/glm4_moe.py"
with open(path_model, "r") as f:
    content_model = f.read()

# Fix the steering call to pass residual
old_call = "                hidden_states = apply_steering(hidden_states, forward_batch.steering_config, i)"
new_call = "                hidden_states = apply_steering(hidden_states, forward_batch.steering_config, i, residual=residual)"

if old_call not in content_model:
    print(f"ERROR: Could not find apply_steering call in glm4_moe.py")
    sys.exit(1)

content_model = content_model.replace(old_call, new_call, 1)

# Fix the capture to use full representation
old_capture = "                _maybe_capture(hidden_states, i, forward_batch, n_layers=len(self.layers))"
new_capture = "                _maybe_capture(hidden_states + residual if residual is not None else hidden_states, i, forward_batch, n_layers=len(self.layers))"

if old_capture not in content_model:
    print(f"ERROR: Could not find _maybe_capture call in glm4_moe.py")
    sys.exit(1)

content_model = content_model.replace(old_capture, new_capture, 1)

with open(path_model, "w") as f:
    f.write(content_model)
print(f"[OK] Fixed forward loop in {path_model}")


# ============================================================
# Summary
# ============================================================
print("\n=== FIXES APPLIED ===")
print("1. apply_steering now takes 'residual' parameter")
print("   - Projects FULL representation (h + residual) onto refusal direction")
print("   - Applies correction to hidden_states only (propagates via residual)")
print("2. _maybe_capture now saves (hidden_states + residual)")
print("   - Captures the full layer output representation")
print("   - Matches what HuggingFace output_hidden_states provides")
print("3. Forward loop passes residual to both functions")
print("\nNext steps:")
print("  1. Kill current SGLang server")
print("  2. Restart without steering (just capture)")
print("  3. Re-run sweep to extract new vector in correct space")
print("  4. Restart with new vector and validate")
