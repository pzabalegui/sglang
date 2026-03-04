"""
DAS v4: Adaptive decode steering via PyTorch forward hooks.

Requires --disable-cuda-graph (hooks don't fire during CUDA graph replay).
Registers a post-forward hook on each Glm4MoeDecoderLayer that:

1. Detects decode mode (residual is not None + forward_mode.is_decode)
2. Computes projection of (h+residual) onto refusal direction
3. Applies momentum-adaptive scale: higher when model keeps projecting onto refusal
4. Subtracts the clamped, scaled projection from hidden_states

This replaces the inline decode steering in glm4_moe.py forward() loop.
When hooks are active, inline decode steering should be disabled (decode_scale=0.0)
to avoid double-steering.

Usage:
  1. Start server with --disable-cuda-graph --steering-decode-scale 0.0
  2. After server starts, run this script to register hooks:
     python3 patch_decode_hooks.py  (runs on the server)

  OR: import and call register_decode_hooks() from within the model initialization.
"""
import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger("sglang")

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Base decode scale (projection coefficient)
    "base_scale": 2.0,
    # Maximum adaptive scale multiplier (e.g., 2.0 = up to 2x base)
    "max_scale_mult": 2.5,
    # EMA decay for momentum tracking (0.9 = 10-token memory)
    "ema_decay": 0.85,
    # Sigmoid steepness for adaptive scaling
    "sigmoid_steepness": 4.0,
    # Sigmoid center (momentum value where scale = 50% of max)
    "sigmoid_center": 0.3,
    # Layer range for decode steering [start, end] (inclusive)
    "layer_start": 30,
    "layer_end": 65,
    # Number of SVD directions to use per layer (1-3)
    "n_directions": 1,
    # Whether to use trapezoidal weighting within layer range
    "use_layer_weights": True,
    "trap_ramp": 5,
    # Minimum projection to trigger steering (noise filter)
    "proj_threshold": 0.01,
}


def _compute_layer_weight(layer_idx, config):
    """Trapezoidal layer weight within [start, end]."""
    s, e = config["layer_start"], config["layer_end"]
    ramp = config["trap_ramp"]
    if layer_idx < s or layer_idx > e:
        return 0.0
    if not config["use_layer_weights"]:
        return 1.0
    if s + ramp <= layer_idx <= e - ramp:
        return 1.0
    elif layer_idx < s + ramp:
        return (layer_idx - s) / max(ramp, 1)
    else:
        return (e - layer_idx) / max(ramp, 1)


class DecodeSteeringHook:
    """Per-layer decode steering hook with momentum-adaptive scale.

    For each decode token, tracks an EMA of the refusal projection magnitude.
    When the model consistently projects onto refusal (high momentum),
    the steering scale increases. When generating non-refusal content,
    the scale decreases.

    Scale formula:
        adaptive_scale = base_scale * layer_weight * sigmoid(momentum)
        where sigmoid maps [0, inf) -> [0, max_scale_mult]
    """

    def __init__(self, layer_idx, directions, config, device):
        """
        Args:
            layer_idx: which transformer layer this hook is on
            directions: tensor [k, hidden_size] — refusal directions for this layer
            config: dict of hyperparameters
            device: GPU device
        """
        self.layer_idx = layer_idx
        self.config = config
        self.layer_weight = _compute_layer_weight(layer_idx, config)
        self.k = directions.shape[0]

        # Directions on GPU, normalized
        self.directions = directions.to(device=device, dtype=torch.bfloat16)
        norms = self.directions.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.directions = self.directions / norms

        # Per-sequence momentum (reset each prefill)
        # Using a simple scalar that we'll broadcast
        self.momentum = torch.zeros(1, device=device, dtype=torch.float32)
        self.step_count = 0

        # Stats for logging
        self.total_projections = 0
        self.total_steered = 0

    def reset_momentum(self):
        """Call at the start of each new sequence (prefill)."""
        self.momentum.zero_()
        self.step_count = 0

    def __call__(self, module, args, output):
        """Post-forward hook on Glm4MoeDecoderLayer.

        Layer returns (hidden_states, residual).
        We modify hidden_states in-place when in decode mode.
        """
        # Unpack output — Glm4MoeDecoderLayer returns (hidden_states, residual)
        if not isinstance(output, tuple) or len(output) != 2:
            return output

        hidden_states, residual = output

        # Only steer during decode (residual is not None)
        if residual is None:
            return output

        # Skip if layer weight is zero
        if self.layer_weight < 1e-6:
            return output

        # Compute h + residual for projection basis
        h_plus_r = hidden_states + residual  # [bs, hidden_size]

        cfg = self.config
        self.step_count += 1

        for ki in range(self.k):
            d = self.directions[ki]  # [hidden_size]

            # Project onto refusal direction: (h+r) · d
            proj = (h_plus_r * d).sum(dim=-1, keepdim=True)  # [bs, 1]

            # Clamp: only steer when aligned with refusal (positive projection)
            proj_clamped = proj.clamp(min=0)  # [bs, 1]

            # Mean projection magnitude (across batch)
            proj_mag = proj_clamped.mean().item()
            self.total_projections += 1

            # Skip if projection is negligible (noise filter)
            if proj_mag < cfg["proj_threshold"]:
                continue

            # Update momentum (EMA of projection magnitude)
            self.momentum.mul_(cfg["ema_decay"]).add_(
                proj_mag * (1 - cfg["ema_decay"])
            )

            # Adaptive scale via sigmoid
            # sigmoid((momentum - center) * steepness) maps momentum to [0, 1]
            m = self.momentum.item()
            sig_input = (m - cfg["sigmoid_center"]) * cfg["sigmoid_steepness"]
            sig = 1.0 / (1.0 + math.exp(-sig_input))

            adaptive_scale = (
                cfg["base_scale"]
                * self.layer_weight
                * sig
                * cfg["max_scale_mult"]
            )

            # Apply: h' = h - scale * max(0, proj) * d
            correction = adaptive_scale * proj_clamped * d  # [bs, hidden_size]
            hidden_states = hidden_states - correction
            self.total_steered += 1

            # Recompute h+r for next direction
            if ki < self.k - 1:
                h_plus_r = hidden_states + residual

        return (hidden_states, residual)


class DecodeSteeringManager:
    """Manages all decode steering hooks across layers."""

    def __init__(self):
        self.hooks = []  # registered hook handles
        self.layer_hooks = {}  # layer_idx -> DecodeSteeringHook
        self.config = dict(DEFAULT_CONFIG)

    def register(self, model, config=None):
        """Register decode steering hooks on all layers of a Glm4MoeModel.

        Args:
            model: Glm4MoeModel (model.model if from Glm4MoeForCausalLM)
            config: optional config dict (merged with DEFAULT_CONFIG)
        """
        if config:
            self.config.update(config)

        # Get directions from the model
        if not hasattr(model, '_steering_dirs'):
            logger.warning("[v4 hooks] No _steering_dirs on model, skipping hook registration")
            return

        dirs = model._steering_dirs  # [n_layers, k, hidden_size]
        device = dirs.device
        n_layers = dirs.shape[0]
        k = min(self.config["n_directions"], dirs.shape[1])

        logger.info(
            f"[v4 hooks] Registering decode steering hooks on layers "
            f"{self.config['layer_start']}-{self.config['layer_end']}, "
            f"k={k}, base_scale={self.config['base_scale']}"
        )

        registered = 0
        for i, layer in enumerate(model.layers):
            weight = _compute_layer_weight(i, self.config)
            if weight < 1e-6:
                continue

            hook_obj = DecodeSteeringHook(
                layer_idx=i,
                directions=dirs[i, :k, :],
                config=self.config,
                device=device,
            )
            handle = layer.register_forward_hook(hook_obj)
            self.hooks.append(handle)
            self.layer_hooks[i] = hook_obj
            registered += 1

        logger.info(f"[v4 hooks] Registered {registered} decode steering hooks")

    def reset_all_momentum(self):
        """Reset momentum for all hooks (call at start of each new sequence)."""
        for hook in self.layer_hooks.values():
            hook.reset_momentum()

    def remove_all(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.layer_hooks.clear()

    def get_stats(self):
        """Get aggregated steering statistics."""
        total_proj = sum(h.total_projections for h in self.layer_hooks.values())
        total_steered = sum(h.total_steered for h in self.layer_hooks.values())
        return {
            "total_projections": total_proj,
            "total_steered": total_steered,
            "steer_rate": total_steered / max(total_proj, 1),
        }


def register_decode_hooks(model_runner, config=None):
    """Convenience function to register hooks from a ModelRunner.

    Args:
        model_runner: SGLang ModelRunner instance
        config: optional config dict
    Returns:
        DecodeSteeringManager instance
    """
    model = model_runner.model
    # Navigate to Glm4MoeModel (inner model)
    inner = getattr(model, 'model', model)

    mgr = DecodeSteeringManager()
    mgr.register(inner, config)
    return mgr


# ─── Standalone: patch running server ────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Register DAS v4 decode hooks on running server")
    parser.add_argument("--base-scale", type=float, default=2.0)
    parser.add_argument("--max-scale-mult", type=float, default=2.5)
    parser.add_argument("--ema-decay", type=float, default=0.85)
    parser.add_argument("--layer-start", type=int, default=30)
    parser.add_argument("--layer-end", type=int, default=65)
    parser.add_argument("--n-directions", type=int, default=1)
    args = parser.parse_args()

    config = {
        "base_scale": args.base_scale,
        "max_scale_mult": args.max_scale_mult,
        "ema_decay": args.ema_decay,
        "layer_start": args.layer_start,
        "layer_end": args.layer_end,
        "n_directions": args.n_directions,
    }

    print(f"DAS v4 decode hooks config: {json.dumps(config, indent=2)}")
    print("NOTE: This script must be imported from within the SGLang process.")
    print("Use: from patch_decode_hooks import register_decode_hooks")
    print("Or add to glm4_moe.py _init_steering() method.")
