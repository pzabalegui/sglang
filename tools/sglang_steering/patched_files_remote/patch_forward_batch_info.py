"""
Patch forward_batch_info.py to add steering_disabled and steering_decode_scale_override
fields to ForwardBatch.

- steering_disabled: True when any request has steering_enabled=False
- steering_decode_scale_override: per-request decode scale (None = use server default)

The model (glm4_moe.py) and cuda_graph_runner.py read these fields.
"""
import sys

SGLANG_DIR = "/tmp/sglang_steering/python/sglang"
FBI_FILE = f"{SGLANG_DIR}/srt/model_executor/forward_batch_info.py"

with open(FBI_FILE, "r") as f:
    content = f.read()

# Check if v2 (decode_scale_override) already patched
if "steering_decode_scale_override" in content:
    print("forward_batch_info.py: Already patched (v2 with decode_scale_override). Skipping.")
    sys.exit(0)

# Check if v1 (steering_disabled only) is patched — upgrade to v2
if "steering_disabled" in content:
    # Add steering_decode_scale_override field after steering_disabled
    old_field = """    # Per-request steering toggle (patched)
    steering_disabled: bool = False"""
    new_field = """    # Per-request steering toggle (patched)
    steering_disabled: bool = False
    # Per-request decode scale override (None = use server default)
    steering_decode_scale_override: float = None"""

    if old_field in content:
        content = content.replace(old_field, new_field, 1)
        print("Added steering_decode_scale_override field to ForwardBatch")
    else:
        # Try alternate format
        content = content.replace(
            "    steering_disabled: bool = False",
            "    steering_disabled: bool = False\n    # Per-request decode scale override (None = use server default)\n    steering_decode_scale_override: float = None",
            1,
        )
        print("Added steering_decode_scale_override field (alt format)")

    # Add decode_scale_override logic to init_new
    # Find existing steering_disabled logic and add decode_scale after it
    old_toggle = """                if getattr(req, "steering_enabled", None) is False:
                    ret.steering_disabled = True
                    break"""
    new_toggle = """                if getattr(req, "steering_enabled", None) is False:
                    ret.steering_disabled = True
                # Per-request decode scale override (first non-None wins)
                _ds = getattr(req, "steering_decode_scale", None)
                if _ds is not None and ret.steering_decode_scale_override is None:
                    ret.steering_decode_scale_override = float(_ds)"""

    if old_toggle in content:
        content = content.replace(old_toggle, new_toggle, 1)
        print("Added decode_scale_override logic to init_new")
    else:
        print("WARNING: Could not find steering_disabled toggle block to upgrade. Manual edit needed.")

    with open(FBI_FILE, "w") as f:
        f.write(content)
    print("forward_batch_info.py v1 → v2 upgrade complete.")
    sys.exit(0)

# Fresh patch: add both fields
# Part 1: Add fields to ForwardBatch dataclass
target_field = "    mrope_positions: torch.Tensor = None"
if target_field not in content:
    print("ERROR: Could not find 'mrope_positions: torch.Tensor = None' in ForwardBatch")
    sys.exit(1)

replacement_field = """    mrope_positions: torch.Tensor = None

    # Per-request steering toggle (patched)
    steering_disabled: bool = False
    # Per-request decode scale override (None = use server default)
    steering_decode_scale_override: float = None"""

content = content.replace(target_field, replacement_field, 1)
print("Added steering_disabled + steering_decode_scale_override fields to ForwardBatch")

# Part 2: Set fields from batch.reqs in init_new()
target_init = "        device = model_runner.device\n"
if target_init not in content:
    print("ERROR: Could not find 'device = model_runner.device' in init_new")
    sys.exit(1)

insert_code = """
        # Per-request steering toggle + decode scale override (patched)
        if hasattr(batch, "reqs") and batch.reqs:
            for req in batch.reqs:
                if getattr(req, "steering_enabled", None) is False:
                    ret.steering_disabled = True
                # Per-request decode scale override (first non-None wins)
                _ds = getattr(req, "steering_decode_scale", None)
                if _ds is not None and ret.steering_decode_scale_override is None:
                    ret.steering_decode_scale_override = float(_ds)

"""

# Count occurrences to patch the right one
if content.count(target_init) > 1:
    idx_init_new = content.find("def init_new(")
    idx_device = content.find(target_init, idx_init_new)
    if idx_device == -1:
        print("ERROR: Could not find 'device = model_runner.device' after init_new")
        sys.exit(1)
    insert_pos = idx_device + len(target_init)
    content = content[:insert_pos] + insert_code + content[insert_pos:]
    print("Added steering toggle + decode_scale logic to init_new (position-based)")
else:
    content = content.replace(target_init, target_init + insert_code, 1)
    print("Added steering toggle + decode_scale logic to init_new")

with open(FBI_FILE, "w") as f:
    f.write(content)

print("forward_batch_info.py patch complete.")
