"""
Patch server_args.py to add DAS v1 + v2 + v3 steering CLI arguments.

Adds dataclass fields and argparse CLI args for:
- v1: steering_vector_path, steering_scale, steering_layers, steering_mode,
      steering_kernel_width, steering_decode_scale
- v2: steering_per_layer_path, steering_attn_scale, steering_mlp_scale,
      steering_kernel, steering_trap_start, steering_trap_end, steering_trap_ramp
- v3: steering_n_directions, steering_decode_layers

IMPORTANT: CLI args MUST be inside add_cli_args() method, NOT at module level.
"""
import sys

SGLANG_DIR = "/tmp/sglang_steering/python/sglang"
SA_FILE = f"{SGLANG_DIR}/srt/server_args.py"

with open(SA_FILE, "r") as f:
    content = f.read()

# =============================================
# Part 1: Add dataclass fields to ServerArgs
# =============================================
if "steering_vector_path" in content and "steering_attn_scale" in content and "steering_n_directions" in content:
    print("server_args.py: Already fully patched (v1 + v2 + v3). Skipping.")
    sys.exit(0)

# Determine if v1 fields already exist
has_v1 = "steering_vector_path" in content

# Find insertion point for dataclass fields
# Look for a field near the end of the ServerArgs dataclass
# We'll insert before the first method definition
if not has_v1:
    # Insert all fields (v1 + v2)
    target_field = "    enable_multimodal: bool = True"
    if target_field not in content:
        # Try alternative insertion points
        target_field = "    disable_overlap_schedule: bool = False"
        if target_field not in content:
            print("ERROR: Could not find suitable insertion point in ServerArgs dataclass")
            sys.exit(1)

    steering_fields = """
    # === DAS Steering (v1 + v2 + v3) ===
    steering_vector_path: Optional[str] = None
    steering_per_layer_path: Optional[str] = None
    steering_scale: float = 5.0
    steering_attn_scale: float = 0.0
    steering_mlp_scale: float = 0.0
    steering_decode_scale: float = 0.0
    steering_n_directions: int = 1
    steering_decode_layers: Optional[str] = None  # JSON list, e.g. '[35,40,47,55,60]'
    steering_layers: Optional[str] = None  # JSON list, e.g. '[47]'
    steering_mode: str = "gaussian"
    steering_kernel: str = "gaussian"  # 'gaussian' or 'trapezoidal'
    steering_kernel_width: float = 2.0
    steering_trap_start: int = 30
    steering_trap_end: int = 65
    steering_trap_ramp: int = 5"""

    content = content.replace(target_field, target_field + steering_fields, 1)
    print("Added v1 + v2 steering dataclass fields")
elif "steering_attn_scale" not in content:
    # v1 exists, add v2+v3 fields after steering_decode_scale
    target_field = "    steering_decode_scale: float = 0.0"
    if target_field not in content:
        # Try finding the last v1 field
        target_field = "    steering_kernel_width: float = 2.0"

    if target_field in content:
        v2_fields = """
    # === DAS v2 + v3 additions ===
    steering_per_layer_path: Optional[str] = None
    steering_attn_scale: float = 0.0
    steering_mlp_scale: float = 0.0
    steering_n_directions: int = 1
    steering_decode_layers: Optional[str] = None  # JSON list, e.g. '[35,40,47,55,60]'
    steering_kernel: str = "gaussian"  # 'gaussian' or 'trapezoidal'
    steering_trap_start: int = 30
    steering_trap_end: int = 65
    steering_trap_ramp: int = 5"""
        content = content.replace(target_field, target_field + v2_fields, 1)
        print("Added v2+v3 steering dataclass fields (v1 already present)")
    else:
        print("WARNING: Could not find v1 steering fields to extend")
elif "steering_n_directions" not in content:
    # v1+v2 exist, add v3 fields
    target_field = "    steering_trap_ramp: int = 5"
    if target_field in content:
        v3_fields = """
    # === DAS v3 additions ===
    steering_n_directions: int = 1
    steering_decode_layers: Optional[str] = None  # JSON list, e.g. '[35,40,47,55,60]'"""
        content = content.replace(target_field, target_field + v3_fields, 1)
        print("Added v3 steering dataclass fields (v1+v2 already present)")
    else:
        print("WARNING: Could not find v2 steering fields to extend")

# =============================================
# Part 2: Add CLI args inside add_cli_args()
# =============================================
# Find the add_cli_args method and a suitable insertion point inside it
if "steering-vector-path" in content and "steering-attn-scale" in content and "steering-n-directions" in content:
    print("CLI args already fully patched (v1+v2+v3). Skipping.")
else:
    # Find the end of add_cli_args() - look for the return statement
    target_cli = '        return parser'
    if target_cli not in content:
        # Some versions might not have return parser
        target_cli = "        # DP attention"
        if target_cli not in content:
            print("WARNING: Could not find insertion point in add_cli_args()")
            sys.exit(1)

    has_v1_cli = "steering-vector-path" in content

    if not has_v1_cli:
        cli_args = """
        # === DAS Steering CLI args (v1 + v2 + v3) ===
        parser.add_argument("--steering-vector-path", type=str, default=None,
            help="Path to refusal direction vector (.pt file, shape [hidden_size])")
        parser.add_argument("--steering-per-layer-path", type=str, default=None,
            help="Path to per-layer refusal directions (.pt, shape [n_layers, (k,) hidden_size])")
        parser.add_argument("--steering-scale", type=float, default=5.0,
            help="Post-layer steering scale (v1 compatible)")
        parser.add_argument("--steering-attn-scale", type=float, default=0.0,
            help="DAS v2: post-attention steering scale (0 = disabled)")
        parser.add_argument("--steering-mlp-scale", type=float, default=0.0,
            help="DAS v2: post-MoE steering scale (0 = disabled)")
        parser.add_argument("--steering-decode-scale", type=float, default=0.0,
            help="Clamped projective decode steering scale (0 = disabled)")
        parser.add_argument("--steering-n-directions", type=int, default=1,
            help="DAS v3: number of SVD directions per layer (1=v2 compat)")
        parser.add_argument("--steering-decode-layers", type=str, default=None,
            help="DAS v3: JSON list of decode steering layers, e.g. '[35,40,47,55,60]'")
        parser.add_argument("--steering-layers", type=str, default=None,
            help="JSON list of center layers, e.g. '[47]'")
        parser.add_argument("--steering-mode", type=str, default="gaussian",
            choices=["gaussian", "single"],
            help="Steering mode: gaussian kernel or single layer")
        parser.add_argument("--steering-kernel", type=str, default="gaussian",
            choices=["gaussian", "trapezoidal"],
            help="DAS v2: layer weight kernel type")
        parser.add_argument("--steering-kernel-width", type=float, default=2.0,
            help="Gaussian kernel sigma (width)")
        parser.add_argument("--steering-trap-start", type=int, default=30,
            help="DAS v2: trapezoidal kernel start layer")
        parser.add_argument("--steering-trap-end", type=int, default=65,
            help="DAS v2: trapezoidal kernel end layer")
        parser.add_argument("--steering-trap-ramp", type=int, default=5,
            help="DAS v2: trapezoidal kernel ramp width")

"""
        content = content.replace(target_cli, cli_args + target_cli, 1)
        print("Added all DAS v1 + v2 + v3 CLI args to add_cli_args()")
    elif "steering-attn-scale" not in content:
        # v1 CLI exists, add v2+v3 CLI args after the last v1 arg
        # Find the last v1 steering arg
        last_v1_arg = 'help="Gaussian kernel sigma (width)")'
        if last_v1_arg not in content:
            last_v1_arg = 'help="Clamped projective decode steering scale'
            # Find the full line
            idx = content.find(last_v1_arg)
            if idx >= 0:
                # Find the end of this line
                end_idx = content.find("\n", idx)
                last_v1_arg = content[content.rfind("parser.add_argument", 0, idx):end_idx + 1]

        if last_v1_arg in content:
            v2_cli = """
        # === DAS v2 + v3 CLI args ===
        parser.add_argument("--steering-per-layer-path", type=str, default=None,
            help="Path to per-layer refusal directions (.pt, shape [n_layers, (k,) hidden_size])")
        parser.add_argument("--steering-attn-scale", type=float, default=0.0,
            help="DAS v2: post-attention steering scale (0 = disabled)")
        parser.add_argument("--steering-mlp-scale", type=float, default=0.0,
            help="DAS v2: post-MoE steering scale (0 = disabled)")
        parser.add_argument("--steering-n-directions", type=int, default=1,
            help="DAS v3: number of SVD directions per layer (1=v2 compat)")
        parser.add_argument("--steering-decode-layers", type=str, default=None,
            help="DAS v3: JSON list of decode steering layers, e.g. '[35,40,47,55,60]'")
        parser.add_argument("--steering-kernel", type=str, default="gaussian",
            choices=["gaussian", "trapezoidal"],
            help="DAS v2: layer weight kernel type")
        parser.add_argument("--steering-trap-start", type=int, default=30,
            help="DAS v2: trapezoidal kernel start layer")
        parser.add_argument("--steering-trap-end", type=int, default=65,
            help="DAS v2: trapezoidal kernel end layer")
        parser.add_argument("--steering-trap-ramp", type=int, default=5,
            help="DAS v2: trapezoidal kernel ramp width")
"""
            content = content.replace(last_v1_arg, last_v1_arg + v2_cli, 1)
            print("Added v2+v3 CLI args after existing v1 args")
        else:
            print("WARNING: Could not find last v1 CLI arg to extend")
    elif "steering-n-directions" not in content:
        # v1+v2 CLI exist, add only v3 CLI args
        last_v2_arg = 'help="DAS v2: trapezoidal kernel ramp width")'
        if last_v2_arg in content:
            v3_cli = """
        # === DAS v3 CLI args ===
        parser.add_argument("--steering-n-directions", type=int, default=1,
            help="DAS v3: number of SVD directions per layer (1=v2 compat)")
        parser.add_argument("--steering-decode-layers", type=str, default=None,
            help="DAS v3: JSON list of decode steering layers, e.g. '[35,40,47,55,60]'")
"""
            content = content.replace(last_v2_arg, last_v2_arg + v3_cli, 1)
            print("Added v3 CLI args after existing v2 args")
        else:
            print("WARNING: Could not find last v2 CLI arg to extend")

with open(SA_FILE, "w") as f:
    f.write(content)

print("server_args.py patch complete.")
