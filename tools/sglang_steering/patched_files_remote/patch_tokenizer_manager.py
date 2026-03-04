"""
Patch tokenizer_manager.py to propagate steering_enabled/steering_scale/steering_decode_scale
from GenerateReqInput to TokenizedGenerateReqInput.

Without this patch, per-request steering toggle and decode scale override
do not work because the fields are lost during tokenization.
"""
import sys

SGLANG_DIR = "/tmp/sglang_steering/python/sglang"
TM_FILE = f"{SGLANG_DIR}/srt/managers/tokenizer_manager.py"

with open(TM_FILE, "r") as f:
    content = f.read()

# Check if already patched (v2: includes decode_scale)
if "steering_decode_scale=obj.steering_decode_scale" in content:
    print("tokenizer_manager.py: Already patched (v2 with decode_scale). Nothing to do.")
    sys.exit(0)

# If v1 patch exists (only steering_enabled/steering_scale), upgrade to v2
if "steering_enabled=obj.steering_enabled" in content:
    # Add steering_decode_scale after existing steering_scale line
    old_v1 = """                steering_enabled=obj.steering_enabled,
                steering_scale=obj.steering_scale,"""
    new_v2 = """                steering_enabled=obj.steering_enabled,
                steering_scale=obj.steering_scale,
                steering_decode_scale=obj.steering_decode_scale,"""
    if old_v1 in content:
        content = content.replace(old_v1, new_v2, 1)
        print("Upgraded v1 patch to v2: added steering_decode_scale")
    else:
        print("WARNING: v1 patch found but format unexpected. Adding steering_decode_scale manually.")
        # Try inserting after steering_scale
        target = "steering_scale=obj.steering_scale,"
        content = content.replace(target, target + "\n                steering_decode_scale=obj.steering_decode_scale,", 1)
else:
    # Fresh patch: add all three fields
    target = "                num_items_assigned=obj.num_items_assigned,"
    if target not in content:
        print("ERROR: Could not find 'num_items_assigned=obj.num_items_assigned,' in tokenizer_manager.py")
        sys.exit(1)

    replacement = target + """
                steering_enabled=obj.steering_enabled,
                steering_scale=obj.steering_scale,
                steering_decode_scale=obj.steering_decode_scale,"""

    content = content.replace(target, replacement, 1)
    print("Added steering_enabled/steering_scale/steering_decode_scale to TokenizedGenerateReqInput constructor")

with open(TM_FILE, "w") as f:
    f.write(content)

print("tokenizer_manager.py patch complete.")
