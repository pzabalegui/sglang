"""
Patch schedule_batch.py to add steering_enabled/steering_scale/steering_decode_scale
to the Req class, and patch scheduler.py to copy these fields when creating Req objects.

Without this patch, per-request steering toggle and decode scale override
do not work because the Req class doesn't have steering fields.
"""
import sys

SGLANG_DIR = "/tmp/sglang_steering/python/sglang"

# =============================================
# Part 1: Patch schedule_batch.py (Req class)
# =============================================
SB_FILE = f"{SGLANG_DIR}/srt/managers/schedule_batch.py"

with open(SB_FILE, "r") as f:
    content = f.read()

if "self.steering_decode_scale" in content:
    print("schedule_batch.py: Already patched (v2 with decode_scale). Skipping.")
elif "self.steering_enabled" in content:
    # v1 patch exists, upgrade to v2: add steering_decode_scale
    # Add to __init__ signature
    old_sig = """        steering_enabled: Optional[bool] = None,
        steering_scale: Optional[float] = None,
    ):"""
    new_sig = """        steering_enabled: Optional[bool] = None,
        steering_scale: Optional[float] = None,
        steering_decode_scale: Optional[float] = None,
    ):"""
    if old_sig in content:
        content = content.replace(old_sig, new_sig, 1)
    else:
        print("WARNING: v1 signature format unexpected, inserting decode_scale manually")
        content = content.replace(
            "        steering_scale: Optional[float] = None,",
            "        steering_scale: Optional[float] = None,\n        steering_decode_scale: Optional[float] = None,",
            1,
        )

    # Add to __init__ body
    old_body = """        self.steering_enabled = steering_enabled
        self.steering_scale = steering_scale"""
    new_body = """        self.steering_enabled = steering_enabled
        self.steering_scale = steering_scale
        self.steering_decode_scale = steering_decode_scale"""
    if old_body in content:
        content = content.replace(old_body, new_body, 1)
    else:
        content = content.replace(
            "        self.steering_scale = steering_scale",
            "        self.steering_scale = steering_scale\n        self.steering_decode_scale = steering_decode_scale",
            1,
        )

    with open(SB_FILE, "w") as f:
        f.write(content)
    print("Upgraded schedule_batch.py v1 → v2: added steering_decode_scale")
else:
    # Fresh patch: add all three fields
    # 1A: Add steering params to Req.__init__ signature
    target_param = "        http_worker_ipc: Optional[str] = None,\n    ):"
    if target_param not in content:
        print("ERROR: Could not find 'http_worker_ipc: Optional[str] = None,' closing in Req.__init__")
        sys.exit(1)

    replacement_param = """        http_worker_ipc: Optional[str] = None,
        steering_enabled: Optional[bool] = None,
        steering_scale: Optional[float] = None,
        steering_decode_scale: Optional[float] = None,
    ):"""

    content = content.replace(target_param, replacement_param, 1)
    print("Added steering params to Req.__init__ signature")

    # 1B: Add assignments in __init__ body
    target_body = "        self.rid = rid"
    if target_body not in content:
        print("ERROR: Could not find 'self.rid = rid' in Req.__init__")
        sys.exit(1)

    replacement_body = """        self.rid = rid
        # Per-request steering override (patched)
        self.steering_enabled = steering_enabled
        self.steering_scale = steering_scale
        self.steering_decode_scale = steering_decode_scale"""

    content = content.replace(target_body, replacement_body, 1)
    print("Added self.steering_enabled/steering_scale/steering_decode_scale to Req.__init__ body")

    with open(SB_FILE, "w") as f:
        f.write(content)

    print("schedule_batch.py patch complete.")

# =============================================
# Part 2: Patch scheduler.py (Req creation)
# =============================================
SC_FILE = f"{SGLANG_DIR}/srt/managers/scheduler.py"

with open(SC_FILE, "r") as f:
    content = f.read()

if "steering_decode_scale=recv_req.steering_decode_scale" in content:
    print("scheduler.py: Already patched (v2 with decode_scale). Skipping.")
elif "steering_enabled=recv_req.steering_enabled" in content:
    # v1 patch exists, upgrade to v2
    old_sched = """                steering_enabled=recv_req.steering_enabled,
                steering_scale=recv_req.steering_scale,"""
    new_sched = """                steering_enabled=recv_req.steering_enabled,
                steering_scale=recv_req.steering_scale,
                steering_decode_scale=recv_req.steering_decode_scale,"""
    if old_sched in content:
        content = content.replace(old_sched, new_sched, 1)
    else:
        content = content.replace(
            "                steering_scale=recv_req.steering_scale,",
            "                steering_scale=recv_req.steering_scale,\n                steering_decode_scale=recv_req.steering_decode_scale,",
            1,
        )
    with open(SC_FILE, "w") as f:
        f.write(content)
    print("Upgraded scheduler.py v1 → v2: added steering_decode_scale")
else:
    # Fresh patch
    target_req = "                dllm_config=self.dllm_config,\n            )\n            req.tokenizer = self.tokenizer"
    if target_req not in content:
        print("ERROR: Could not find Req() constructor closing in scheduler.py")
        sys.exit(1)

    replacement_req = """                dllm_config=self.dllm_config,
                steering_enabled=recv_req.steering_enabled,
                steering_scale=recv_req.steering_scale,
                steering_decode_scale=recv_req.steering_decode_scale,
            )
            req.tokenizer = self.tokenizer"""

    content = content.replace(target_req, replacement_req, 1)
    print("Added steering fields to Req() constructor call in scheduler.py")

    with open(SC_FILE, "w") as f:
        f.write(content)

    print("scheduler.py patch complete.")
