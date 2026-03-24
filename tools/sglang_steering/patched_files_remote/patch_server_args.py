"""
Patch server_args.py to add DAS v1 + v2 + v3 + v5 steering CLI arguments.

Adds dataclass fields and argparse CLI args for:
- v1: steering_vector_path, steering_scale, steering_layers, steering_mode,
      steering_kernel_width, steering_decode_scale
- v2: steering_per_layer_path, steering_attn_scale, steering_mlp_scale,
      steering_kernel, steering_trap_start, steering_trap_end, steering_trap_ramp
- v3: steering_n_directions, steering_decode_layers
- v5: steering_attn_scale_full, steering_attn_scale_linear,
      steering_mlp_scale_full, steering_mlp_scale_linear, steering_k_directions

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
if ("steering_vector_path" in content and "steering_attn_scale" in content
        and "steering_n_directions" in content and "steering_intervention_mode" in content
        and "steering_k_directions" in content and "steering_sig_mode" in content
        and "steering_prefill_mode" in content
        and "steering_diagnostics" in content
        and "steering_attn_per_layer_path" in content
        and "abliteration_vector_path" in content
        and "abliteration_rank" in content):
    print("server_args.py: Already fully patched (v1+v2+v3+WRMD+v5+v6+v8+v9+v10+ablit-v2). Skipping.")
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
    steering_trap_ramp: int = 5
    # === WRMD Additive steering ===
    steering_intervention_mode: str = "projective"
    steering_layer_coeffs_path: Optional[str] = None
    # === DAS v5: hybrid attention-type steering ===
    steering_attn_scale_full: float = 0.0
    steering_attn_scale_linear: float = 0.0
    steering_mlp_scale_full: float = 0.0
    steering_mlp_scale_linear: float = 0.0
    steering_k_directions: int = 4
    # === DAS v6: sigmoid mode + SV weights ===
    steering_sig_mode: str = "sigmoid"  # 'sigmoid', 'linear', or 'none'
    steering_sig_steepness: float = 4.0
    steering_sv_weights_path: Optional[str] = None
    # === DAS v8: prefill steering mode ===
    steering_prefill_mode: str = "sublayer"  # 'sublayer' (v2) or 'fullresidual' (v8)
    # === DAS v9: diagnostics ===
    steering_diagnostics: bool = False
    # === DAS v10: sub-layer direction files ===
    steering_attn_per_layer_path: Optional[str] = None
    steering_mlp_per_layer_path: Optional[str] = None"""

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
    steering_trap_ramp: int = 5
    # === WRMD Additive steering ===
    steering_intervention_mode: str = "projective"
    steering_layer_coeffs_path: Optional[str] = None
    # === DAS v5: hybrid attention-type steering ===
    steering_attn_scale_full: float = 0.0
    steering_attn_scale_linear: float = 0.0
    steering_mlp_scale_full: float = 0.0
    steering_mlp_scale_linear: float = 0.0
    steering_k_directions: int = 4
    # === DAS v6: sigmoid mode + SV weights ===
    steering_sig_mode: str = "sigmoid"  # 'sigmoid', 'linear', or 'none'
    steering_sig_steepness: float = 4.0
    steering_sv_weights_path: Optional[str] = None
    # === DAS v8: prefill steering mode ===
    steering_prefill_mode: str = "sublayer"  # 'sublayer' (v2) or 'fullresidual' (v8)
    # === DAS v9: diagnostics ===
    steering_diagnostics: bool = False
    # === DAS v10: sub-layer direction files ===
    steering_attn_per_layer_path: Optional[str] = None
    steering_mlp_per_layer_path: Optional[str] = None"""
        content = content.replace(target_field, target_field + v2_fields, 1)
        print("Added v2+v3 steering dataclass fields (v1 already present)")
    else:
        print("WARNING: Could not find v1 steering fields to extend")
elif "steering_n_directions" not in content:
    # v1+v2 exist, add v3 + WRMD fields
    target_field = "    steering_trap_ramp: int = 5"
    if target_field in content:
        v3_fields = """
    # === DAS v3 additions ===
    steering_n_directions: int = 1
    steering_decode_layers: Optional[str] = None  # JSON list, e.g. '[35,40,47,55,60]'
    # === WRMD Additive steering ===
    steering_intervention_mode: str = "projective"
    steering_layer_coeffs_path: Optional[str] = None
    # === DAS v5: hybrid attention-type steering ===
    steering_attn_scale_full: float = 0.0
    steering_attn_scale_linear: float = 0.0
    steering_mlp_scale_full: float = 0.0
    steering_mlp_scale_linear: float = 0.0
    steering_k_directions: int = 4
    # === DAS v6: sigmoid mode + SV weights ===
    steering_sig_mode: str = "sigmoid"  # 'sigmoid', 'linear', or 'none'
    steering_sig_steepness: float = 4.0
    steering_sv_weights_path: Optional[str] = None
    # === DAS v8: prefill steering mode ===
    steering_prefill_mode: str = "sublayer"  # 'sublayer' (v2) or 'fullresidual' (v8)
    # === DAS v9: diagnostics ===
    steering_diagnostics: bool = False
    # === DAS v10: sub-layer direction files ===
    steering_attn_per_layer_path: Optional[str] = None
    steering_mlp_per_layer_path: Optional[str] = None"""
        content = content.replace(target_field, target_field + v3_fields, 1)
        print("Added v3+WRMD steering dataclass fields (v1+v2 already present)")
    else:
        print("WARNING: Could not find v2 steering fields to extend")
elif "steering_intervention_mode" not in content:
    # v1+v2+v3 exist, add WRMD-only fields
    target_field_wrmd = "    steering_decode_layers: Optional[str] = None  # JSON list, e.g. '[35,40,47,55,60]'"
    if target_field_wrmd not in content:
        target_field_wrmd = "    steering_trap_ramp: int = 5"
    if target_field_wrmd in content:
        wrmd_fields = """
    # === WRMD Additive steering ===
    steering_intervention_mode: str = "projective"
    steering_layer_coeffs_path: Optional[str] = None
    # === DAS v5: hybrid attention-type steering ===
    steering_attn_scale_full: float = 0.0
    steering_attn_scale_linear: float = 0.0
    steering_mlp_scale_full: float = 0.0
    steering_mlp_scale_linear: float = 0.0
    steering_k_directions: int = 4
    # === DAS v6: sigmoid mode + SV weights ===
    steering_sig_mode: str = "sigmoid"  # 'sigmoid', 'linear', or 'none'
    steering_sig_steepness: float = 4.0
    steering_sv_weights_path: Optional[str] = None
    # === DAS v8: prefill steering mode ===
    steering_prefill_mode: str = "sublayer"  # 'sublayer' (v2) or 'fullresidual' (v8)
    # === DAS v9: diagnostics ===
    steering_diagnostics: bool = False
    # === DAS v10: sub-layer direction files ===
    steering_attn_per_layer_path: Optional[str] = None
    steering_mlp_per_layer_path: Optional[str] = None"""
        content = content.replace(target_field_wrmd, target_field_wrmd + wrmd_fields, 1)
        print("Added WRMD+v5 dataclass fields (v1+v2+v3 already present)")
    else:
        print("WARNING: Could not find insertion point for WRMD fields")
elif "steering_k_directions" not in content:
    # v1+v2+v3+WRMD exist, add v5-only fields
    target_field_v5 = "    steering_layer_coeffs_path: Optional[str] = None"
    if target_field_v5 in content:
        v5_fields = """
    # === DAS v5: hybrid attention-type steering ===
    steering_attn_scale_full: float = 0.0
    steering_attn_scale_linear: float = 0.0
    steering_mlp_scale_full: float = 0.0
    steering_mlp_scale_linear: float = 0.0
    steering_k_directions: int = 4
    # === DAS v6: sigmoid mode + SV weights ===
    steering_sig_mode: str = "sigmoid"  # 'sigmoid', 'linear', or 'none'
    steering_sig_steepness: float = 4.0
    steering_sv_weights_path: Optional[str] = None
    # === DAS v8: prefill steering mode ===
    steering_prefill_mode: str = "sublayer"  # 'sublayer' (v2) or 'fullresidual' (v8)
    # === DAS v9: diagnostics ===
    steering_diagnostics: bool = False
    # === DAS v10: sub-layer direction files ===
    steering_attn_per_layer_path: Optional[str] = None
    steering_mlp_per_layer_path: Optional[str] = None"""
        content = content.replace(target_field_v5, target_field_v5 + v5_fields, 1)
        print("Added v5 dataclass fields (v1+v2+v3+WRMD already present)")
    else:
        print("WARNING: Could not find insertion point for v5 fields")

# =============================================
# Part 1b: Add abliteration dataclass field
# =============================================
if "abliteration_vector_path" not in content:
    # Find the last steering field to insert after
    ablit_target = "    steering_mlp_per_layer_path: Optional[str] = None"
    if ablit_target not in content:
        ablit_target = "    steering_diagnostics: bool = False"
    if ablit_target in content:
        ablit_field = """
    # === Inline Abliteration ===
    abliteration_vector_path: Optional[str] = None
    abliteration_rank: int = 1"""
        content = content.replace(ablit_target, ablit_target + ablit_field, 1)
        print("Added abliteration dataclass fields")
    else:
        print("WARNING: Could not find insertion point for abliteration field")

# =============================================
# Part 2: Add CLI args inside add_cli_args()
# =============================================
# Find the add_cli_args method and a suitable insertion point inside it
if ("steering-vector-path" in content and "steering-attn-scale" in content
        and "steering-n-directions" in content and "steering-intervention-mode" in content
        and "steering-k-directions" in content and "steering-sig-mode" in content
        and "steering-prefill-mode" in content
        and "steering-diagnostics" in content
        and "steering-attn-per-layer-path" in content
        and "abliteration-vector-path" in content
        and "abliteration-rank" in content):
    print("CLI args already fully patched (v1+v2+v3+WRMD+v5+v6+v8+v9+v10+ablit-v2). Skipping.")
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
        # === WRMD Additive steering CLI args ===
        parser.add_argument("--steering-intervention-mode", type=str, default="projective",
            choices=["projective", "additive"],
            help="Steering type: 'projective' (remove projection) or 'additive' (add direction)")
        parser.add_argument("--steering-layer-coeffs-path", type=str, default=None,
            help="Path to per-layer scaling coefficients (.pt, shape [n_layers])")
        # === DAS v5: hybrid attention-type steering CLI args ===
        parser.add_argument("--steering-attn-scale-full", type=float, default=0.0,
            help="DAS v5: post-attention steering scale for full-attention layers")
        parser.add_argument("--steering-attn-scale-linear", type=float, default=0.0,
            help="DAS v5: post-attention steering scale for linear-attention layers")
        parser.add_argument("--steering-mlp-scale-full", type=float, default=0.0,
            help="DAS v5: post-MLP steering scale for full-attention layers")
        parser.add_argument("--steering-mlp-scale-linear", type=float, default=0.0,
            help="DAS v5: post-MLP steering scale for linear-attention layers")
        parser.add_argument("--steering-k-directions", type=int, default=4,
            help="DAS v5: number of orthogonal WRMD directions per layer")
        # === DAS v8: prefill steering mode ===
        parser.add_argument("--steering-prefill-mode", type=str, default="sublayer",
            help="Prefill steering: 'sublayer' (v2 post-attn+post-MLP) or 'fullresidual' (v8 post-layer on h+residual)")
        parser.add_argument("--steering-diagnostics", action="store_true", default=False,
            help="Enable per-layer projection diagnostics (dumps to /tmp/steering_diagnostics.json)")
        # === DAS v10: sub-layer direction files ===
        parser.add_argument("--steering-attn-per-layer-path", type=str, default=None,
            help="DAS v10: Path to attn-specific per-layer directions (.pt, shape [n_layers, k, hidden_size])")
        parser.add_argument("--steering-mlp-per-layer-path", type=str, default=None,
            help="DAS v10: Path to MLP-specific per-layer directions (.pt, shape [n_layers, k, hidden_size])")

"""
        content = content.replace(target_cli, cli_args + target_cli, 1)
        print("Added all DAS v1 + v2 + v3 + WRMD + v5 + v8 CLI args to add_cli_args()")
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
        # === WRMD Additive steering CLI args ===
        parser.add_argument("--steering-intervention-mode", type=str, default="projective",
            choices=["projective", "additive"],
            help="Steering type: 'projective' (remove projection) or 'additive' (add direction)")
        parser.add_argument("--steering-layer-coeffs-path", type=str, default=None,
            help="Path to per-layer scaling coefficients (.pt, shape [n_layers])")
        # === DAS v5: hybrid attention-type steering CLI args ===
        parser.add_argument("--steering-attn-scale-full", type=float, default=0.0,
            help="DAS v5: post-attention steering scale for full-attention layers")
        parser.add_argument("--steering-attn-scale-linear", type=float, default=0.0,
            help="DAS v5: post-attention steering scale for linear-attention layers")
        parser.add_argument("--steering-mlp-scale-full", type=float, default=0.0,
            help="DAS v5: post-MLP steering scale for full-attention layers")
        parser.add_argument("--steering-mlp-scale-linear", type=float, default=0.0,
            help="DAS v5: post-MLP steering scale for linear-attention layers")
        parser.add_argument("--steering-k-directions", type=int, default=4,
            help="DAS v5: number of orthogonal WRMD directions per layer")
        # === DAS v6: sigmoid mode + SV weights ===
        parser.add_argument("--steering-sig-mode", type=str, default="sigmoid",
            choices=["sigmoid", "linear", "none"],
            help="DAS v6: adaptive scaling mode ('sigmoid'=v5, 'linear'=proportional ramp, 'none'=fixed)")
        parser.add_argument("--steering-sig-steepness", type=float, default=4.0,
            help="DAS v6: sigmoid steepness (only used in sigmoid mode, default=4.0)")
        parser.add_argument("--steering-sv-weights-path", type=str, default=None,
            help="Path to per-direction SV weights (.pt, shape [n_layers, k]) for proportional weighting")
        # === DAS v8: prefill steering mode ===
        parser.add_argument("--steering-prefill-mode", type=str, default="sublayer",
            help="Prefill steering: 'sublayer' (v2 post-attn+post-MLP) or 'fullresidual' (v8 post-layer on h+residual)")
        parser.add_argument("--steering-diagnostics", action="store_true", default=False,
            help="Enable per-layer projection diagnostics (dumps to /tmp/steering_diagnostics.json)")
        # === DAS v10: sub-layer direction files ===
        parser.add_argument("--steering-attn-per-layer-path", type=str, default=None,
            help="DAS v10: Path to attn-specific per-layer directions (.pt, shape [n_layers, k, hidden_size])")
        parser.add_argument("--steering-mlp-per-layer-path", type=str, default=None,
            help="DAS v10: Path to MLP-specific per-layer directions (.pt, shape [n_layers, k, hidden_size])")
"""
            content = content.replace(last_v1_arg, last_v1_arg + v2_cli, 1)
            print("Added v2+v3+WRMD+v5 CLI args after existing v1 args")
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
        # === WRMD Additive steering CLI args ===
        parser.add_argument("--steering-intervention-mode", type=str, default="projective",
            choices=["projective", "additive"],
            help="Steering type: 'projective' (remove projection) or 'additive' (add direction)")
        parser.add_argument("--steering-layer-coeffs-path", type=str, default=None,
            help="Path to per-layer scaling coefficients (.pt, shape [n_layers])")
        # === DAS v5: hybrid attention-type steering CLI args ===
        parser.add_argument("--steering-attn-scale-full", type=float, default=0.0,
            help="DAS v5: post-attention steering scale for full-attention layers")
        parser.add_argument("--steering-attn-scale-linear", type=float, default=0.0,
            help="DAS v5: post-attention steering scale for linear-attention layers")
        parser.add_argument("--steering-mlp-scale-full", type=float, default=0.0,
            help="DAS v5: post-MLP steering scale for full-attention layers")
        parser.add_argument("--steering-mlp-scale-linear", type=float, default=0.0,
            help="DAS v5: post-MLP steering scale for linear-attention layers")
        parser.add_argument("--steering-k-directions", type=int, default=4,
            help="DAS v5: number of orthogonal WRMD directions per layer")
        # === DAS v6: sigmoid mode + SV weights ===
        parser.add_argument("--steering-sig-mode", type=str, default="sigmoid",
            choices=["sigmoid", "linear", "none"],
            help="DAS v6: adaptive scaling mode ('sigmoid'=v5, 'linear'=proportional ramp, 'none'=fixed)")
        parser.add_argument("--steering-sig-steepness", type=float, default=4.0,
            help="DAS v6: sigmoid steepness (only used in sigmoid mode, default=4.0)")
        parser.add_argument("--steering-sv-weights-path", type=str, default=None,
            help="Path to per-direction SV weights (.pt, shape [n_layers, k]) for proportional weighting")
        # === DAS v8: prefill steering mode ===
        parser.add_argument("--steering-prefill-mode", type=str, default="sublayer",
            help="Prefill steering: 'sublayer' (v2 post-attn+post-MLP) or 'fullresidual' (v8 post-layer on h+residual)")
        parser.add_argument("--steering-diagnostics", action="store_true", default=False,
            help="Enable per-layer projection diagnostics (dumps to /tmp/steering_diagnostics.json)")
        # === DAS v10: sub-layer direction files ===
        parser.add_argument("--steering-attn-per-layer-path", type=str, default=None,
            help="DAS v10: Path to attn-specific per-layer directions (.pt, shape [n_layers, k, hidden_size])")
        parser.add_argument("--steering-mlp-per-layer-path", type=str, default=None,
            help="DAS v10: Path to MLP-specific per-layer directions (.pt, shape [n_layers, k, hidden_size])")
"""
            content = content.replace(last_v2_arg, last_v2_arg + v3_cli, 1)
            print("Added v3+WRMD+v5 CLI args after existing v2 args")
        else:
            print("WARNING: Could not find last v2 CLI arg to extend")
    elif "steering-intervention-mode" not in content:
        # v1+v2+v3 CLI exist, add WRMD-only CLI args
        # Find the LAST steering-related closing paren ")" to insert after
        # Search for the trap-ramp arg close first, then decode-layers
        for marker in [
            'help="DAS v2: trapezoidal kernel ramp width")',
            "help=\"DAS v2: trapezoidal kernel ramp width\",\n        )",
            "help=\"DAS v3: JSON list of decode steering layers",
        ]:
            idx = content.find(marker)
            if idx >= 0:
                # Find the closing ')' of this parser.add_argument call
                close_idx = content.find(")", idx + len(marker) - 1)
                if close_idx < 0:
                    close_idx = content.find(")", idx)
                # Find the end of the line with the closing paren
                end_idx = content.find("\n", close_idx)
                break
        else:
            idx = -1

        if idx >= 0:
            wrmd_cli = """
        # === WRMD Additive steering CLI args ===
        parser.add_argument("--steering-intervention-mode", type=str, default="projective",
            choices=["projective", "additive"],
            help="Steering type: 'projective' (remove projection) or 'additive' (add direction)")
        parser.add_argument("--steering-layer-coeffs-path", type=str, default=None,
            help="Path to per-layer scaling coefficients (.pt, shape [n_layers])")
        # === DAS v5: hybrid attention-type steering CLI args ===
        parser.add_argument("--steering-attn-scale-full", type=float, default=0.0,
            help="DAS v5: post-attention steering scale for full-attention layers")
        parser.add_argument("--steering-attn-scale-linear", type=float, default=0.0,
            help="DAS v5: post-attention steering scale for linear-attention layers")
        parser.add_argument("--steering-mlp-scale-full", type=float, default=0.0,
            help="DAS v5: post-MLP steering scale for full-attention layers")
        parser.add_argument("--steering-mlp-scale-linear", type=float, default=0.0,
            help="DAS v5: post-MLP steering scale for linear-attention layers")
        parser.add_argument("--steering-k-directions", type=int, default=4,
            help="DAS v5: number of orthogonal WRMD directions per layer")
        # === DAS v6: sigmoid mode + SV weights ===
        parser.add_argument("--steering-sig-mode", type=str, default="sigmoid",
            choices=["sigmoid", "linear", "none"],
            help="DAS v6: adaptive scaling mode ('sigmoid'=v5, 'linear'=proportional ramp, 'none'=fixed)")
        parser.add_argument("--steering-sig-steepness", type=float, default=4.0,
            help="DAS v6: sigmoid steepness (only used in sigmoid mode, default=4.0)")
        parser.add_argument("--steering-sv-weights-path", type=str, default=None,
            help="Path to per-direction SV weights (.pt, shape [n_layers, k]) for proportional weighting")
        # === DAS v8: prefill steering mode ===
        parser.add_argument("--steering-prefill-mode", type=str, default="sublayer",
            help="Prefill steering: 'sublayer' (v2 post-attn+post-MLP) or 'fullresidual' (v8 post-layer on h+residual)")
        parser.add_argument("--steering-diagnostics", action="store_true", default=False,
            help="Enable per-layer projection diagnostics (dumps to /tmp/steering_diagnostics.json)")
        # === DAS v10: sub-layer direction files ===
        parser.add_argument("--steering-attn-per-layer-path", type=str, default=None,
            help="DAS v10: Path to attn-specific per-layer directions (.pt, shape [n_layers, k, hidden_size])")
        parser.add_argument("--steering-mlp-per-layer-path", type=str, default=None,
            help="DAS v10: Path to MLP-specific per-layer directions (.pt, shape [n_layers, k, hidden_size])")
"""
            content = content[:end_idx + 1] + wrmd_cli + content[end_idx + 1:]
            print("Added WRMD+v5 CLI args after existing v1+v2+v3 args")
        else:
            print("WARNING: Could not find insertion point for WRMD CLI args")
    elif "steering-k-directions" not in content:
        # v1+v2+v3+WRMD CLI exist, add v5-only CLI args
        last_wrmd_arg = 'help="Path to per-layer scaling coefficients (.pt, shape [n_layers])")'
        if last_wrmd_arg in content:
            v5_cli = """
        # === DAS v5: hybrid attention-type steering CLI args ===
        parser.add_argument("--steering-attn-scale-full", type=float, default=0.0,
            help="DAS v5: post-attention steering scale for full-attention layers")
        parser.add_argument("--steering-attn-scale-linear", type=float, default=0.0,
            help="DAS v5: post-attention steering scale for linear-attention layers")
        parser.add_argument("--steering-mlp-scale-full", type=float, default=0.0,
            help="DAS v5: post-MLP steering scale for full-attention layers")
        parser.add_argument("--steering-mlp-scale-linear", type=float, default=0.0,
            help="DAS v5: post-MLP steering scale for linear-attention layers")
        parser.add_argument("--steering-k-directions", type=int, default=4,
            help="DAS v5: number of orthogonal WRMD directions per layer")
        # === DAS v6: sigmoid mode + SV weights ===
        parser.add_argument("--steering-sig-mode", type=str, default="sigmoid",
            choices=["sigmoid", "linear", "none"],
            help="DAS v6: adaptive scaling mode ('sigmoid'=v5, 'linear'=proportional ramp, 'none'=fixed)")
        parser.add_argument("--steering-sig-steepness", type=float, default=4.0,
            help="DAS v6: sigmoid steepness (only used in sigmoid mode, default=4.0)")
        parser.add_argument("--steering-sv-weights-path", type=str, default=None,
            help="Path to per-direction SV weights (.pt, shape [n_layers, k]) for proportional weighting")
        # === DAS v8: prefill steering mode ===
        parser.add_argument("--steering-prefill-mode", type=str, default="sublayer",
            help="Prefill steering: 'sublayer' (v2 post-attn+post-MLP) or 'fullresidual' (v8 post-layer on h+residual)")
        parser.add_argument("--steering-diagnostics", action="store_true", default=False,
            help="Enable per-layer projection diagnostics (dumps to /tmp/steering_diagnostics.json)")
        # === DAS v10: sub-layer direction files ===
        parser.add_argument("--steering-attn-per-layer-path", type=str, default=None,
            help="DAS v10: Path to attn-specific per-layer directions (.pt, shape [n_layers, k, hidden_size])")
        parser.add_argument("--steering-mlp-per-layer-path", type=str, default=None,
            help="DAS v10: Path to MLP-specific per-layer directions (.pt, shape [n_layers, k, hidden_size])")
"""
            content = content.replace(last_wrmd_arg, last_wrmd_arg + v5_cli, 1)
            print("Added v5 CLI args after existing v1+v2+v3+WRMD args")
        else:
            print("WARNING: Could not find last WRMD CLI arg to extend")

# =============================================
# Part 2b: Add abliteration CLI arg
# =============================================
if "abliteration-vector-path" not in content:
    ablit_cli_target = '        return parser'
    if ablit_cli_target in content:
        ablit_cli = """
        # === Inline Abliteration ===
        parser.add_argument("--abliteration-vector-path", type=str, default=None,
            help="Path to abliteration direction vector (.pt, shape [hidden_size] or [n_layers, k, hidden_size]). "
                 "Enables per-request weight-equivalent abliteration without modifying weights.")
        parser.add_argument("--abliteration-rank", type=int, default=1,
            help="Number of SVD directions to project out per layer (1=standard, 3=recommended for Qwen)")
"""
        content = content.replace(ablit_cli_target, ablit_cli + ablit_cli_target, 1)
        print("Added abliteration-vector-path CLI arg")
    else:
        print("WARNING: Could not find insertion point for abliteration CLI arg")

with open(SA_FILE, "w") as f:
    f.write(content)

print("server_args.py patch complete.")
