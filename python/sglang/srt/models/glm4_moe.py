# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference-only GLM-4.5, GLM-4.6 and GLM-4.7 model compatible with HuggingFace weights"""

# ============================================================
# ACTIVATION CAPTURE for refusal direction extraction
# ============================================================
import json as _json
import logging
import math as _math
import os as _os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    parallel_state,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_allocation_symmetric,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    filter_moe_weight_param_global_expert,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

_CAPTURE_STORE = {}
_CAPTURE_COUNTER = [0]
_CAPTURE_CONFIG_CACHE = [None, 0]  # [config, last_check_time]


def _get_capture_config():
    """Read capture config with 1-second cache."""
    import time

    now = time.time()
    if now - _CAPTURE_CONFIG_CACHE[1] < 1.0 and _CAPTURE_CONFIG_CACHE[0] is not None:
        return _CAPTURE_CONFIG_CACHE[0]
    _CAPTURE_CONFIG_CACHE[1] = now

    config_path = "/tmp/capture_config.json"
    if not _os.path.exists(config_path):
        _CAPTURE_CONFIG_CACHE[0] = None
        return None
    try:
        with open(config_path, "r") as f:
            cfg = _json.load(f)
        if not cfg.get("enabled", False):
            _CAPTURE_CONFIG_CACHE[0] = None
            return None
        _CAPTURE_CONFIG_CACHE[0] = cfg
        return cfg
    except Exception:
        _CAPTURE_CONFIG_CACHE[0] = None
        return None


def _maybe_capture(hidden_states, layer_idx, forward_batch, n_layers=92, residual=None):
    """Capture hidden states during prefill for refusal direction extraction.

    Captures full representation (hidden_states + residual) for last token only,
    to match what HuggingFace output_hidden_states provides.
    """
    import torch

    cfg = _get_capture_config()
    if cfg is None:
        return

    # Only capture during prefill (not decode)
    if hidden_states.shape[0] <= 2:
        return

    # Only capture on rank 0 (all ranks have identical hidden_states after allreduce)
    try:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass

    target_layers = cfg.get("layers", list(range(n_layers)))
    if layer_idx not in target_layers:
        return

    # Save last token's hidden state (where refusal signal is strongest)
    # Compute full representation (h + residual) only for last token (memory efficient)
    last_h = hidden_states[-1, :]
    if residual is not None:
        last_h = last_h + residual[-1, :]
    _CAPTURE_STORE[layer_idx] = last_h.detach().cpu().to(torch.float32)

    # When we reach the last target layer, save everything to disk
    max_target = max(target_layers)
    if layer_idx == max_target:
        save_dir = cfg.get("save_dir", "/tmp/captures")
        _os.makedirs(save_dir, exist_ok=True)
        sample_id = _CAPTURE_COUNTER[0]
        save_path = _os.path.join(save_dir, f"sample_{sample_id}.pt")
        torch.save(dict(_CAPTURE_STORE), save_path)
        _CAPTURE_STORE.clear()
        _CAPTURE_COUNTER[0] += 1


from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    add_prefix,
    cpu_has_amx_support,
    get_bool_env_var,
    get_device_sm,
    is_cpu,
    is_cuda,
    is_hip,
    is_non_idle_and_non_empty,
    log_info_on_rank0,
    make_layers,
)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_device_sm = get_device_sm()

logger = logging.getLogger(__name__)


# ============================================================
# DAS v4: Adaptive decode steering via PyTorch forward hooks
# Requires --disable-cuda-graph (hooks don't fire during CUDA graph replay).
# When active, inline decode steering is disabled to avoid double-steering.
# ============================================================

_V4_DEFAULT_CONFIG = {
    "base_scale": 2.0,
    "max_scale_mult": 2.5,
    "ema_decay": 0.85,
    "sigmoid_steepness": 4.0,
    "sigmoid_center": 0.3,
    "proj_threshold": 0.01,
}


class _DecodeSteeringHook:
    """Per-layer decode steering hook with per-request momentum-adaptive scale.

    Registered via register_forward_hook on each Glm4MoeDecoderLayer.
    Only fires during decode mode. Uses per-request EMA momentum tracking
    to adaptively scale steering via sigmoid mapping.
    Per-request isolation: each request in the batch has independent momentum.
    """

    def __init__(self, layer_idx, directions, layer_weight, config, device):
        self.layer_idx = layer_idx
        self.config = config
        self.layer_weight = layer_weight
        self.k = directions.shape[0]
        self.directions = directions.to(device=device, dtype=torch.bfloat16)
        norms = self.directions.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.directions = self.directions / norms
        # Per-request momentum: lazily allocated [capacity, 1]
        self.momentum = None
        self._mom_capacity = 0
        self.step_count = 0
        self.total_projections = 0
        self.total_steered = 0

    def reset_momentum(self):
        if self.momentum is not None:
            self.momentum.zero_()
        self.step_count = 0

    def _ensure_momentum(self, bs, device):
        """Ensure momentum buffer can hold at least bs requests."""
        if self.momentum is None or bs > self._mom_capacity:
            new_cap = max(bs, self._mom_capacity * 2, 16)
            self.momentum = torch.zeros(new_cap, 1, device=device, dtype=torch.float32)
            self._mom_capacity = new_cap

    def __call__(self, module, input, output):
        """Post-forward hook on Glm4MoeDecoderLayer."""
        if not isinstance(output, tuple) or len(output) != 2:
            return output

        hidden_states, residual = output

        forward_batch = input[2]
        if not forward_batch.forward_mode.is_decode():
            self.reset_momentum()
            return output

        # Skip only if ALL requests have steering off
        if getattr(forward_batch, "steering_disabled", False):
            return output

        if residual is None or self.layer_weight < 1e-6:
            return output

        # Lazy device transfer
        if self.directions.device != hidden_states.device:
            self.directions = self.directions.to(device=hidden_states.device)

        bs = hidden_states.shape[0]
        cfg = self.config
        self._ensure_momentum(bs, hidden_states.device)
        mom = self.momentum[:bs]  # [bs, 1]

        # Build per-request mask [bs, 1] from forward_batch
        _mask_vals = getattr(forward_batch, "steering_mask_values", None)
        if _mask_vals is not None:
            mask = torch.tensor(
                _mask_vals[:bs], device=hidden_states.device, dtype=hidden_states.dtype
            ).unsqueeze(1)
        else:
            mask = None

        # Fold per-request decode scale overrides into mask
        _scale_vals = getattr(forward_batch, "steering_decode_scale_values", None)
        if _scale_vals is not None:
            _base = cfg.get("base_scale", 1.0)
            if _base > 0:
                _ratios = [
                    (
                        float(_scale_vals[i]) / _base
                        if i < len(_scale_vals) and _scale_vals[i] is not None
                        else 1.0
                    )
                    for i in range(bs)
                ]
                if any(abs(r - 1.0) > 1e-6 for r in _ratios):
                    _rt = torch.tensor(
                        _ratios, device=hidden_states.device, dtype=hidden_states.dtype
                    ).unsqueeze(1)
                    mask = _rt if mask is None else mask * _rt

        h_plus_r = hidden_states + residual
        self.step_count += 1

        for ki in range(self.k):
            d = self.directions[ki]
            proj = (h_plus_r * d).sum(dim=-1, keepdim=True)  # [bs, 1]
            proj_clamped = proj.clamp(min=0)  # [bs, 1]
            self.total_projections += bs

            # Per-request EMA momentum update (no batch averaging)
            mom.mul_(cfg["ema_decay"]).add_(proj_clamped, alpha=(1 - cfg["ema_decay"]))

            # Per-request sigmoid adaptive scale (pure tensor ops)
            sig = torch.sigmoid(
                (mom - cfg["sigmoid_center"]) * cfg["sigmoid_steepness"]
            )
            adaptive_scale = (
                cfg["base_scale"] * self.layer_weight * sig * cfg["max_scale_mult"]
            )  # [bs, 1]

            correction = adaptive_scale * proj_clamped * d  # [bs, hidden]
            if mask is not None:
                correction = correction * mask
            hidden_states = hidden_states - correction
            self.total_steered += bs

            if ki < self.k - 1:
                h_plus_r = hidden_states + residual

        return (hidden_states, residual)


class Glm4MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        forward_batch=None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        if (self.tp_size == 1) and x.shape[0] == 0:
            return x

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter
        )
        return x


class Glm4MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        partial_rotary_factor: float = 0.5,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-05,
        attention_bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        use_qk_norm: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.use_qk_norm = use_qk_norm
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_tensor_model_parallel_rank()

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            partial_rotary_factor=partial_rotary_factor,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.alt_stream = alt_stream

    def op_prepare(self, state):
        state.attn_intermediate_state = self.forward_prepare(
            positions=state.positions,
            hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
            forward_batch=state.forward_batch,
        )

    def op_core(self, state):
        state.hidden_states_after_attn = self.forward_core(
            state.pop("attn_intermediate_state")
        )

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states, forward_batch, None
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.use_qk_norm:
            q, k = apply_qk_norm(
                q=q,
                k=k,
                q_norm=self.q_norm,
                k_norm=self.k_norm,
                head_dim=self.head_dim,
                alt_stream=self.alt_stream,
            )
        q, k = self.rotary_emb(positions, q, k)
        inner_state = q, k, v, forward_batch
        return None, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states
        attn_output = self.attn(*inner_state)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        s = self.forward_prepare(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        return self.forward_core(s)


class Glm4MoeGate(nn.Module):
    def __init__(
        self,
        config,
        prefix: str = "",
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty((config.n_routed_experts), dtype=torch.float32)
        )

    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight, None)
        return logits


class Glm4MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        nn.Module.__init__(self)
        self.top_k = config.num_experts_per_tok
        self.tp_size = get_tensor_model_parallel_world_size()
        self.moe_ep_size = get_moe_expert_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.num_fused_shared_experts = (
            0
            if get_global_server_args().disable_shared_experts_fusion
            else config.n_shared_experts
        )

        self.config = config
        self.layer_id = layer_id
        self.alt_stream = alt_stream

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = Glm4MoeGate(config=config, prefix=add_prefix("gate", prefix))

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts + self.num_fused_shared_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            top_k=self.top_k + self.num_fused_shared_experts,
            layer_id=self.layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            routing_method_type=RoutingMethodType.DeepSeekV3,
            prefix=add_prefix("experts", prefix),
        )

        self.topk = TopK(
            top_k=self.top_k + self.num_fused_shared_experts,
            layer_id=self.layer_id,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            num_fused_shared_experts=self.num_fused_shared_experts,
            apply_routed_scaling_factor_on_output=getattr(
                self.experts, "should_fuse_routed_scaling_factor_in_topk", False
            ),
            fused_shared_experts_scaling_factor=1,
        )

        # shared expert
        if config.n_shared_experts is not None and self.num_fused_shared_experts == 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = Glm4MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if get_moe_a2a_backend().is_deepep()
                    or get_moe_a2a_backend().is_mooncake()
                    or get_moe_a2a_backend().is_flashinfer()
                    or should_use_flashinfer_cutlass_moe_fp4_allgather()
                    else {}
                ),
            )

        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            # TODO: we will support tp < ep in the future
            self.ep_size = get_moe_expert_parallel_world_size()
            self.num_experts = (
                config.n_routed_experts
                + get_global_server_args().ep_num_redundant_experts
            )
            self.renormalize = config.norm_topk_prob
            self.topk_group = config.topk_group
            self.num_expert_group = config.n_group
            self.correction_bias = (
                self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None
            )

        self._enable_a2a_moe = (
            get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake()
        )

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
            and filter_moe_weight_param_global_expert(
                name, x, self.experts.num_local_experts
            )
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:

        if not get_moe_a2a_backend().is_deepep():
            if (
                self.alt_stream is not None
                and self.num_fused_shared_experts == 0
                and hidden_states.shape[0] > 0
                and get_is_capture_mode()
            ):
                return self.forward_normal_dual_stream(
                    hidden_states, should_allreduce_fusion, use_reduce_scatter
                )
            else:
                return self.forward_normal(
                    hidden_states, should_allreduce_fusion, use_reduce_scatter
                )
        else:
            return self.forward_deepep(hidden_states, forward_batch)

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        shared_output = self._forward_shared_experts(hidden_states)

        with torch.cuda.stream(self.alt_stream):
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)

            final_hidden_states = self.experts(hidden_states, topk_output)
            if not _is_cuda and not _use_aiter:
                # fused in biased_grouped_topk so we can skip here
                final_hidden_states *= self.routed_scaling_factor

        current_stream.wait_stream(self.alt_stream)

        with use_symmetric_memory(
            parallel_state.get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            final_hidden_states_out = torch.empty_like(final_hidden_states)
        torch.add(final_hidden_states, shared_output, out=final_hidden_states_out)
        final_hidden_states = final_hidden_states_out
        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if hidden_states.shape[0] > 0:
            shared_output = self._forward_shared_experts(hidden_states)
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)
        else:
            shared_output = None
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        final_hidden_states = self.experts(hidden_states, topk_output)
        if not _is_cuda and not _use_aiter:
            final_hidden_states *= self.routed_scaling_factor
        if shared_output is not None:
            with use_symmetric_memory(
                parallel_state.get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                final_hidden_states_out = torch.empty_like(final_hidden_states)
            torch.add(final_hidden_states, shared_output, out=final_hidden_states_out)
            final_hidden_states = final_hidden_states_out
        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        shared_output = None
        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states)
            shared_output = self._forward_shared_experts(hidden_states)
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

        if shared_output is not None:
            x = shared_output
            if self.experts.should_fuse_routed_scaling_factor_in_topk:
                x.add_(final_hidden_states)
            else:
                x.add_(final_hidden_states, alpha=self.routed_scaling_factor)
            final_hidden_states = x
        else:
            if not self.experts.should_fuse_routed_scaling_factor_in_topk:
                final_hidden_states *= self.routed_scaling_factor

        return final_hidden_states

    def _forward_shared_experts(self, hidden_states: torch.Tensor):
        if (hidden_states.shape[0] > 0) and (self.num_fused_shared_experts == 0):
            return self.shared_experts(hidden_states)
        else:
            return None

    def op_gate(self, state):
        if is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, state.hidden_states_mlp_input
        ):
            # router_logits: (num_tokens, n_experts)
            state.router_logits = self.gate(state.hidden_states_mlp_input)
        else:
            state.router_logits = None

    def op_select_experts(self, state):
        router_logits = state.pop("router_logits")
        hidden_states = state.hidden_states_mlp_input

        if router_logits is not None:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.topk_output = self.topk(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    num_token_non_padded=state.forward_batch.num_token_non_padded,
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,
                    ),
                )
        else:
            state.topk_output = self.topk.empty_topk_output(hidden_states.device)

    def op_dispatch_a(self, state):
        if self.ep_size > 1:
            self.experts.dispatcher.dispatch_a(
                hidden_states=state.hidden_states_mlp_input,
                topk_output=state.pop("topk_output"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_dispatch_b(self, state):
        if self.ep_size > 1:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.dispatch_output = self.experts.dispatcher.dispatch_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),
                )

    def op_experts(self, state):
        state.combine_input = self.experts.run_moe_core(
            dispatch_output=state.dispatch_output,
        )

    def op_combine_a(self, state):
        if self.ep_size > 1:
            self.experts.dispatcher.combine_a(
                combine_input=state.pop("combine_input"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )
            state.pop("dispatch_output")

    def op_combine_b(self, state):
        if self.ep_size > 1:
            state.hidden_states_after_combine = self.experts.dispatcher.combine_b(
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_output(self, state):
        final_hidden_states = state.pop("hidden_states_after_combine")

        if (shared_output := state.pop("shared_output")) is not None:
            x = shared_output
            x.add_(final_hidden_states, alpha=self.routed_scaling_factor)
            final_hidden_states = x
        else:
            final_hidden_states *= self.routed_scaling_factor

        state.hidden_states_mlp_output = final_hidden_states


class Glm4MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.config = config
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        partial_rotary_factor = getattr(
            getattr(config, "rope_parameters", None), "partial_rotary_factor", None
        ) or getattr(config, "partial_rotary_factor", 0.5)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rms_norm_eps = config.rms_norm_eps
        attention_bias = config.attention_bias
        self.layer_id = layer_id

        use_qk_norm = config.use_qk_norm if hasattr(config, "use_qk_norm") else False

        self.self_attn = Glm4MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=partial_rotary_factor,
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            use_qk_norm=use_qk_norm,
            alt_stream=alt_stream,
        )

        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)
        is_next_layer_sparse = self._is_layer_sparse(layer_id + 1, is_nextn=False)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=1 if is_nextn else config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = Glm4MoeSparseMoeBlock(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=self.layer_id,
                alt_stream=alt_stream,
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = Glm4MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(
                is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
            ),
        )

    def _is_layer_sparse(self, layer_id: int, is_nextn: bool) -> bool:
        return is_nextn or (
            self.config.n_routed_experts is not None
            and layer_id >= self.config.first_k_dense_replace
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        steering_ctx: Optional[Dict] = None,
    ) -> torch.Tensor:

        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # DAS v2: post-attention steering (after o_proj, before prepare_mlp)
        # Projects refusal direction out of attention output.
        # Note: hidden_states here is the raw attention output (before allreduce in some configs).
        # We only do this during prefill when TP allreduce has happened (reduce_results=False
        # on o_proj means the TP reduction happens in prepare_mlp, so we steer before that).
        # CORRECTION: o_proj has reduce_results=False, so hidden_states is partial TP shard.
        # We steer AFTER prepare_mlp which does the allreduce. See below.
        # Actually the attn steering happens right after prepare_mlp gives us the full tensor.

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        # DAS v2/v3: post-attention steering (after allreduce, before MLP)
        # At this point hidden_states is the fully-reduced attention output after layernorm.
        # This is the correct point to project out refusal from attention contribution.
        # v3: loops over k directions per layer (k=1 for v2 backward compat).
        if steering_ctx is not None and steering_ctx.get("attn_active"):
            _layer_id = self.layer_id
            if _layer_id in steering_ctx["attn_set"]:
                _dirs_k = steering_ctx["dirs"][_layer_id]  # [k, hidden_size]
                _s = steering_ctx["attn_scales"][_layer_id]
                _k = steering_ctx["k"]
                for _ki in range(_k):
                    _dir = _dirs_k[_ki]
                    _proj = (hidden_states * _dir).sum(dim=-1, keepdim=True)
                    _proj.clamp_(min=0)  # only subtract when aligned with refusal
                    hidden_states = hidden_states - _s * _proj * _dir

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # Disable allreduce fusion when steering is active - steering requires
        # fully all-reduced hidden_states to compute correct projections
        if should_allreduce_fusion and forward_batch.steering_config is not None:
            if forward_batch.steering_config.enabled:
                should_allreduce_fusion = False
        # Also disable for DAS v2 MLP steering
        if should_allreduce_fusion and steering_ctx is not None:
            if steering_ctx.get("mlp_active") and self.layer_id in steering_ctx.get(
                "mlp_set", set()
            ):
                should_allreduce_fusion = False

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        hidden_states = self.mlp(
            hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
        )

        # DAS v2/v3: post-MoE steering (after MLP/MoE output, before residual add)
        # This targets the MLP contribution to refusal specifically.
        # v3: loops over k directions per layer.
        if steering_ctx is not None and steering_ctx.get("mlp_active"):
            _layer_id = self.layer_id
            if _layer_id in steering_ctx["mlp_set"]:
                _dirs_k = steering_ctx["dirs"][_layer_id]  # [k, hidden_size]
                _s = steering_ctx["mlp_scales"][_layer_id]
                _k = steering_ctx["k"]
                for _ki in range(_k):
                    _dir = _dirs_k[_ki]
                    _proj = (hidden_states * _dir).sum(dim=-1, keepdim=True)
                    _proj.clamp_(min=0)  # only subtract when aligned with refusal
                    hidden_states = hidden_states - _s * _proj * _dir

        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual

    def op_comm_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tbo_subbatch_index: Optional[int] = None,
    ):
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)
        )
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_comm_prepare_mlp(self, state):
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self.layer_communicator.prepare_mlp(
                state.pop("hidden_states_after_attn"),
                state.pop("residual_after_input_ln"),
                state.forward_batch,
            )
        )

    def op_mlp(self, state):
        hidden_states = state.pop("hidden_states_mlp_input")
        if not (
            enable_moe_dense_fully_dp()
            and (not self.is_layer_sparse)
            and hidden_states.shape[0] == 0
        ):
            state.hidden_states_mlp_output = self.mlp(
                hidden_states, state.forward_batch
            )
        else:
            state.hidden_states_mlp_output = hidden_states

    def op_comm_postprocess_layer(self, state):
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            state.pop("hidden_states_mlp_output"),
            state.pop("residual_after_comm_pre_mlp"),
            state.forward_batch,
        )

        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=state.forward_batch,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )

        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "tbo_subbatch_index",
            }
        )
        return output


class Glm4MoeModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace
        self.embed_dim = config.hidden_size
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream = torch.cuda.Stream() if _is_cuda else None
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Glm4MoeDecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        self.layers_to_capture = []

    def get_input_embeddings(self) -> torch.Tensor:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        normal_start_layer = self.start_layer
        normal_end_layer = self.end_layer
        if forward_batch.can_run_tbo:
            if (
                self.first_k_dense_replace > normal_start_layer
                and self.first_k_dense_replace < normal_end_layer
            ):
                normal_end_layer = self.first_k_dense_replace
            elif self.first_k_dense_replace < normal_start_layer:
                normal_end_layer = normal_start_layer = 0

        # --- DAS v1/v2 steering setup ---
        # Prefill: always eager → steering is safe (out-of-place ops).
        # Decode: only clamped projective with pre-allocated buffers (CUDA-graph safe).
        _is_prefill = not forward_batch.forward_mode.is_decode()

        # Per-request steering toggle
        _steering_off = getattr(forward_batch, "steering_disabled", False)

        # DAS v2/v3: multi-point steering context (attn + MLP intervention)
        _steering_v2 = (
            getattr(self, "_steering_v2", False) and _is_prefill and not _steering_off
        )
        _steering_k = getattr(self, "_steering_k", 1)

        # Post-layer steering (v1 compatible): projects refusal from hidden_states
        # When v2 is active, post-layer steering is skipped to avoid double-steering
        # (v2 already steers at attn + MLP points within the layer).
        _has_steering = (
            hasattr(self, "_steering_dir")
            and self._steering_dir is not None
            and _is_prefill
            and not _steering_v2
            and not _steering_off
        )
        _steered_layers = getattr(self, "_steered_layer_set", None)
        _steering_ctx = None
        if _steering_v2:
            _attn_set = getattr(self, "_attn_steered_layer_set", frozenset())
            _mlp_set = getattr(self, "_mlp_steered_layer_set", frozenset())
            _steering_ctx = {
                "dirs": self._steering_dirs,  # [n_layers, k, hidden_size]
                "attn_scales": self._steering_attn_scales,  # [n_layers]
                "mlp_scales": self._steering_mlp_scales,  # [n_layers]
                "attn_set": _attn_set,
                "mlp_set": _mlp_set,
                "attn_active": len(_attn_set) > 0,
                "mlp_active": len(_mlp_set) > 0,
                "k": _steering_k,
            }

        # Clamped projective decode steering: pre-allocated buffers -> CUDA-graph safe.
        # v3: multi-layer decode with per-layer scales
        # v4: momentum-adaptive scaling in eager mode (--disable-cuda-graph)
        _has_decode_steering = (
            forward_batch.forward_mode.is_decode()
            and getattr(self, "_steer_dec_scale", None) is not None
            and not _steering_off
        )
        _decode_peak_layer = getattr(self, "_decode_steer_peak_layer", -1)
        _decode_steered_set = getattr(self, "_decode_steered_set", frozenset())
        _has_multi_layer_decode = getattr(self, "_has_multi_layer_decode", False)
        _v4_adaptive = getattr(self, "_v4_adaptive", False)

        # v4: Reset momentum at prefill (new sequence start)
        if _is_prefill and _v4_adaptive and hasattr(self, "_steer_momentum"):
            self._steer_momentum.zero_()

        # Per-request mask + decode scale override (eager mode: set mask from forward_batch)
        _saved_steering_mask = None
        if (
            _has_decode_steering
            and hasattr(self, "_steering_mask")
            and self._steering_mask is not None
        ):
            _bs_fb = forward_batch.batch_size
            _mask_vals_fb = getattr(forward_batch, "steering_mask_values", None)
            _scale_vals_fb = getattr(
                forward_batch, "steering_decode_scale_values", None
            )
            _saved_steering_mask = self._steering_mask[:_bs_fb].clone()
            if _mask_vals_fb is not None:
                _bs_actual = min(len(_mask_vals_fb), _bs_fb)
                self._steering_mask[:_bs_actual, 0] = torch.tensor(
                    _mask_vals_fb[:_bs_actual],
                    dtype=self._steering_mask.dtype,
                    device=self._steering_mask.device,
                )
            else:
                self._steering_mask[:_bs_fb].fill_(1.0)
            # Fold per-request decode scale overrides into mask
            if _scale_vals_fb is not None and self._steer_dec_scale is not None:
                _global_default = self._steer_dec_scale.item()
                if _global_default > 0:
                    for _si in range(min(len(_scale_vals_fb), _bs_fb)):
                        if _scale_vals_fb[_si] is not None:
                            self._steering_mask[_si, 0] *= (
                                float(_scale_vals_fb[_si]) / _global_default
                            )

        aux_hidden_states = []
        for i in range(normal_start_layer, normal_end_layer):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                if i in self.layers_to_capture:
                    aux_hidden_states.append(hidden_states + residual)
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                    steering_ctx=_steering_ctx,
                )
                # POST-LAYER projective steering (prefill-only, v1 compatible).
                # v3: loops over k directions per layer.
                if (
                    _has_steering
                    and _steered_layers is not None
                    and i in _steered_layers
                ):
                    _scale = self._steering_scales[i]
                    _dirs_layer = (
                        self._steering_dirs[i]
                        if hasattr(self, "_steering_dirs")
                        else self._steering_dir.unsqueeze(0)
                    )
                    for _ki in range(_steering_k):
                        _dir = _dirs_layer[_ki]
                        _proj = (hidden_states * _dir).sum(dim=-1, keepdim=True)
                        _proj.clamp_(min=0)  # only subtract when aligned with refusal
                        hidden_states = hidden_states - _scale * _proj * _dir
                # Clamped projective decode steering (pre-allocated, CUDA-graph safe).
                # v3: multi-layer decode with per-layer scales + multi-vector k-loop.
                if _has_decode_steering and residual is not None:
                    _should_decode_steer = (
                        _has_multi_layer_decode and i in _decode_steered_set
                    ) or (not _has_multi_layer_decode and i == _decode_peak_layer)
                    if _should_decode_steer:
                        _bs = hidden_states.shape[0]
                        _t1 = self._steer_dec_tmp1[:_bs]
                        _t2 = self._steer_dec_tmp2[:_bs]
                        _p = self._steer_dec_proj[:_bs]
                        # Per-layer decode scale (v3: already includes decode_scale*weight)
                        # or global scale (v1/v2: single _steer_dec_scale value)
                        # Note: _steer_dec_scales already has decode_scale baked in;
                        # CUDA graph toggle zeros _steer_dec_scales directly.
                        _dec_scale_i = (
                            self._steer_dec_scales[i]
                            if _has_multi_layer_decode
                            else self._steer_dec_scale
                        )
                        torch.add(hidden_states, residual, out=_t1)
                        for _ki in range(_steering_k):
                            _dir_ki = self._steering_dirs[i][_ki]
                            torch.mul(_t1, _dir_ki, out=_t2)
                            _p.copy_(_t2.sum(dim=-1, keepdim=True))
                            _p.clamp_(min=0)
                            torch.mul(_p, _dir_ki, out=_t2)
                            if _v4_adaptive:
                                # DAS v4: per-request momentum-adaptive scaling (CUDA-graph safe)
                                # Each row in the batch has its own momentum/sigmoid.
                                _mom = self._steer_momentum[:_bs]
                                # 1. Per-request EMA: momentum[i] = decay*momentum[i] + (1-decay)*proj[i]
                                _mom.mul_(self._v4_ema_decay).add_(
                                    _p, alpha=self._v4_ema_complement
                                )
                                # 2. Sigmoid input: (momentum - center) * steepness
                                _stmp = self._v4_sig_tmp[:_bs]
                                torch.sub(_mom, self._v4_sig_center, out=_stmp)
                                _stmp.mul_(self._v4_sig_steep)
                                # 3. Sigmoid → per-request adaptive scale [bs, 1]
                                _sres = self._v4_sig_result[:_bs]
                                torch.sigmoid(_stmp, out=_sres)
                                # 4. Apply: correction * base_scale * sigmoid * max_mult * mask
                                _t2.mul_(_dec_scale_i)
                                _t2.mul_(_sres)
                                _t2.mul_(self._v4_max_mult)
                                _t2.mul_(self._steering_mask[:_bs])
                            else:
                                _t2.mul_(_dec_scale_i)
                                _t2.mul_(self._steering_mask[:_bs])
                            hidden_states.sub_(_t2)
                            # Recompute h+residual for next direction (hidden_states changed)
                            if _ki < _steering_k - 1:
                                torch.add(hidden_states, residual, out=_t1)
                _maybe_capture(
                    hidden_states,
                    i,
                    forward_batch,
                    n_layers=len(self.layers),
                    residual=residual,
                )

        # Restore per-request mask (eager mode)
        if _saved_steering_mask is not None:
            self._steering_mask[: _saved_steering_mask.shape[0]].copy_(
                _saved_steering_mask
            )

        if normal_end_layer != self.end_layer:
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers[normal_end_layer : self.end_layer],
                enable_tbo=True,
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
                input_data_scatter_mode=self.layers[
                    normal_end_layer - 1
                ].layer_scatter_modes.layer_output_mode,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if not forward_batch.forward_mode.is_idle():
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states


class Glm4MoeForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.pp_group = get_pp_group()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.num_fused_shared_experts = 0
        self.determine_num_fused_shared_experts()
        self.model = Glm4MoeModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

        # Initialize GPU-native steering buffers (CUDA-graph compatible)
        self._init_steering()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def _init_steering(self) -> None:
        """Initialize GPU-native steering buffers for CUDA-graph compatible DAS.

        DAS v3 supports:
        - Multi-vector per-layer refusal directions [n_layers, k, hidden_size] (SVD)
        - Backward compatible with [n_layers, hidden_size] (k=1) and [hidden_size] (single)
        - Separate attn/MLP scales (--steering-attn-scale, --steering-mlp-scale)
        - Trapezoidal or Gaussian kernel for layer weighting
        - Three intervention points: post-attn (o_proj), post-MoE, post-layer (residual)
        - Multi-layer decode steering with kernel-weighted per-layer scales
        - Clamped projective decode steering with pre-allocated buffers
        """
        args = get_global_server_args()
        if not getattr(args, "steering_vector_path", None):
            return

        n_layers = self.config.num_hidden_layers
        hid = self.config.hidden_size
        n_dirs = int(getattr(args, "steering_n_directions", 1))

        # --- Load steering vectors ---
        # DAS v3: per-layer vectors [n_layers, k, hidden_size] via --steering-per-layer-path
        # DAS v2: per-layer vectors [n_layers, hidden_size] (auto-upgraded to k=1)
        # DAS v1 fallback: single vector [hidden_size] via --steering-vector-path
        per_layer_path = getattr(args, "steering_per_layer_path", None)
        if per_layer_path:
            per_layer_vecs = torch.load(
                per_layer_path, map_location="cpu", weights_only=True
            )
            # Backward compatibility: [n_layers, hidden_size] -> [n_layers, 1, hidden_size]
            if per_layer_vecs.dim() == 2:
                if per_layer_vecs.shape[0] != n_layers:
                    raise ValueError(
                        f"Per-layer vectors must have {n_layers} layers, got {per_layer_vecs.shape[0]}"
                    )
                per_layer_vecs = per_layer_vecs.unsqueeze(1)  # [n_layers, 1, hid]
                logger.info(
                    f"[steering] Upgraded 2D vectors to 3D: {per_layer_vecs.shape}"
                )
            elif per_layer_vecs.dim() == 3:
                if per_layer_vecs.shape[0] != n_layers:
                    raise ValueError(
                        f"Per-layer vectors must have {n_layers} layers, got {per_layer_vecs.shape[0]}"
                    )
            else:
                raise ValueError(
                    f"Per-layer vectors must be 2D or 3D, got {per_layer_vecs.dim()}D: {per_layer_vecs.shape}"
                )
            # Clamp k to what's available in the file
            k = min(n_dirs, per_layer_vecs.shape[1])
            per_layer_vecs = per_layer_vecs[:, :k, :]  # [n_layers, k, hid]
            # Normalize each direction
            norms = per_layer_vecs.norm(dim=2, keepdim=True).clamp(min=1e-8)
            per_layer_vecs = per_layer_vecs / norms
            logger.info(
                f"[steering] Loaded per-layer directions: {per_layer_vecs.shape} (k={k})"
            )
            self.model._steering_per_layer = True
        else:
            # v1 fallback: broadcast single vector to all layers, k=1
            vec = torch.load(
                args.steering_vector_path, map_location="cpu", weights_only=True
            )
            if vec.dim() != 1:
                raise ValueError(f"Steering vector must be 1-D, got shape {vec.shape}")
            vec = vec.float()
            vec = vec / vec.norm()
            per_layer_vecs = (
                vec.unsqueeze(0).unsqueeze(0).expand(n_layers, 1, -1).contiguous()
            )
            k = 1
            logger.info(
                f"[steering] Using single vector broadcast to all layers: {vec.shape}"
            )
            self.model._steering_per_layer = False

        # Store k for use in forward loops
        self.model._steering_k = k

        # Register per-layer direction matrix [n_layers, k, hidden_size] on Glm4MoeModel
        self.model.register_buffer("_steering_dirs", per_layer_vecs.bfloat16())

        # Also keep a single global direction for v1 decode steering compatibility
        steering_layers_str = getattr(args, "steering_layers", None)
        center_layers = (
            _json.loads(steering_layers_str) if steering_layers_str else [47]
        )
        peak_layer = int(center_layers[0]) if center_layers else 47
        self.model.register_buffer(
            "_steering_dir", per_layer_vecs[peak_layer, 0].clone().bfloat16()
        )

        # --- Compute layer weights (kernel) ---
        mode = getattr(args, "steering_mode", "gaussian")
        kernel = getattr(args, "steering_kernel", mode)  # 'gaussian' or 'trapezoidal'
        base_scale = float(getattr(args, "steering_scale", 5.0))
        sigma = float(getattr(args, "steering_kernel_width", 2.0))

        # Attn and MLP scales: if not specified, both default to base_scale
        attn_scale = float(getattr(args, "steering_attn_scale", 0.0))
        mlp_scale = float(getattr(args, "steering_mlp_scale", 0.0))

        # v2 mode detection: if attn_scale or mlp_scale is set, use multi-point steering
        self.model._steering_v2 = attn_scale > 0.0 or mlp_scale > 0.0

        # Compute layer weight envelope (0-1 range, then scaled)
        layer_weights = torch.zeros(n_layers, dtype=torch.float32)

        if kernel == "trapezoidal":
            # Heretic-style trapezoidal: flat top around peak, linear ramp down
            # Default range: L30-L65 for 92-layer model
            trap_start = int(getattr(args, "steering_trap_start", 30))
            trap_end = int(getattr(args, "steering_trap_end", 65))
            trap_ramp = int(getattr(args, "steering_trap_ramp", 5))
            for i in range(n_layers):
                if trap_start + trap_ramp <= i <= trap_end - trap_ramp:
                    layer_weights[i] = 1.0
                elif trap_start <= i < trap_start + trap_ramp:
                    layer_weights[i] = (i - trap_start) / max(trap_ramp, 1)
                elif trap_end - trap_ramp < i <= trap_end:
                    layer_weights[i] = (trap_end - i) / max(trap_ramp, 1)
            logger.info(
                f"[steering] Trapezoidal kernel: [{trap_start}, {trap_end}] ramp={trap_ramp}"
            )
        else:
            # Gaussian kernel (v1 compatible)
            for c in center_layers:
                for i in range(n_layers):
                    w = _math.exp(-0.5 * ((i - c) / sigma) ** 2)
                    if w > layer_weights[i].item():
                        layer_weights[i] = w
            logger.info(
                f"[steering] Gaussian kernel: centers={center_layers}, sigma={sigma}"
            )

        # Compute per-layer scales for each intervention point
        # v1 mode: base_scale applies to post-layer (hidden_states after MLP+residual)
        # v2 mode: attn_scale applies post-attn, mlp_scale applies post-MoE
        scales = layer_weights * base_scale  # post-layer scales (v1 compatible)
        attn_scales = layer_weights * attn_scale  # post-attn scales (v2)
        mlp_scales = layer_weights * mlp_scale  # post-MoE scales (v2)

        self.model.register_buffer("_steering_scales", scales.bfloat16())
        self.model.register_buffer("_steering_attn_scales", attn_scales.bfloat16())
        self.model.register_buffer("_steering_mlp_scales", mlp_scales.bfloat16())

        # Pre-compute active layer sets for each intervention point
        # Only layers with non-zero scale generate GPU ops
        _active_set = frozenset(
            int(i) for i in range(n_layers) if scales[i].item() > 1e-6
        )
        _attn_active_set = frozenset(
            int(i) for i in range(n_layers) if attn_scales[i].item() > 1e-6
        )
        _mlp_active_set = frozenset(
            int(i) for i in range(n_layers) if mlp_scales[i].item() > 1e-6
        )
        self.model._steered_layer_set = _active_set
        self.model._attn_steered_layer_set = _attn_active_set
        self.model._mlp_steered_layer_set = _mlp_active_set
        n_active = len(_active_set)
        n_attn_active = len(_attn_active_set)
        n_mlp_active = len(_mlp_active_set)

        # --- Decode-time clamped projective steering ---
        # v3: multi-layer decode with kernel-weighted per-layer scales
        decode_scale = float(getattr(args, "steering_decode_scale", 0.0))
        self.model._decode_steer_peak_layer = peak_layer
        if decode_scale > 0.0:
            max_bs = getattr(args, "cuda_graph_max_bs", 80)
            self.model.register_buffer(
                "_steer_dec_tmp1", torch.zeros(max_bs, hid, dtype=torch.bfloat16)
            )
            self.model.register_buffer(
                "_steer_dec_tmp2", torch.zeros(max_bs, hid, dtype=torch.bfloat16)
            )
            self.model.register_buffer(
                "_steer_dec_proj", torch.zeros(max_bs, 1, dtype=torch.bfloat16)
            )
            self.model.register_buffer(
                "_steer_dec_scale", torch.tensor(decode_scale, dtype=torch.bfloat16)
            )

            # v3: compute per-layer decode scales (kernel-weighted)
            # Parse decode layers: explicit list or auto-derive from kernel
            decode_layers_str = getattr(args, "steering_decode_layers", None)
            if decode_layers_str:
                decode_layers = _json.loads(decode_layers_str)
            else:
                # Auto: use layers with kernel weight > 0.1 (trapezoidal body or Gaussian core)
                decode_layers = [peak_layer]

            # Per-layer decode scales: decode_scale * layer_weights[i] for selected layers
            decode_scales = torch.zeros(n_layers, dtype=torch.float32)
            for dl in decode_layers:
                if 0 <= dl < n_layers:
                    w = layer_weights[dl].item()
                    decode_scales[dl] = decode_scale * max(
                        w, 0.1
                    )  # minimum 0.1 for selected layers

            self.model.register_buffer("_steer_dec_scales", decode_scales.bfloat16())
            _decode_steered_set = frozenset(
                int(i) for i in range(n_layers) if decode_scales[i].item() > 1e-6
            )
            self.model._decode_steered_set = _decode_steered_set
            self.model._has_multi_layer_decode = len(_decode_steered_set) > 1

            logger.info(
                f"[steering] Clamped-projective decode steering: "
                f"layers={sorted(_decode_steered_set)}, decode_scale={decode_scale}, "
                f"max_bs={max_bs}, k={k}"
            )
        else:
            self.model._steer_dec_scale = None
            self.model._steer_dec_scales = None
            self.model._decode_steered_set = frozenset()
            self.model._has_multi_layer_decode = False

        logger.info(
            f"[steering] DAS {'v3' if k > 1 else ('v2' if self.model._steering_v2 else 'v1')} initialized: "
            f"kernel={kernel}, center_layers={center_layers}, k={k}, "
            f"base_scale={base_scale}, attn_scale={attn_scale}, mlp_scale={mlp_scale}, "
            f"post-layer active={n_active}/{n_layers}, "
            f"post-attn active={n_attn_active}/{n_layers}, "
            f"post-MoE active={n_mlp_active}/{n_layers}, "
            f"decode_layers={sorted(self.model._decode_steered_set) if self.model._decode_steered_set else 'none'}, "
            f"per_layer={'yes' if self.model._steering_per_layer else 'no'}, "
            f"decode_scale={decode_scale}"
        )

        # --- DAS v4: Momentum-adaptive decode steering ---
        # CUDA-graph safe: all ops are in-place on pre-allocated buffers.
        # Momentum persists across decode steps (graph replays reuse same tensors).
        # Reset happens at prefill (eager mode) or via cuda_graph_runner on toggle.
        self.model._v4_adaptive = decode_scale > 0.0
        _v4_max_bs = getattr(args, "cuda_graph_max_bs", 80)
        if self.model._v4_adaptive:
            # Per-request momentum: each slot in the decode batch has its own EMA
            self.model.register_buffer(
                "_steer_momentum", torch.zeros(_v4_max_bs, 1, dtype=torch.float32)
            )
            # Per-request sigmoid intermediates
            self.model.register_buffer(
                "_v4_sig_tmp", torch.zeros(_v4_max_bs, 1, dtype=torch.float32)
            )
            self.model.register_buffer(
                "_v4_sig_result", torch.zeros(_v4_max_bs, 1, dtype=torch.float32)
            )
            # Per-request steering mask: 1.0=ON, 0.0=OFF (CUDA-graph safe)
            self.model.register_buffer(
                "_steering_mask", torch.ones(_v4_max_bs, 1, dtype=torch.bfloat16)
            )
            # EMA config as registered buffers (move to GPU with model)
            self.model.register_buffer(
                "_v4_ema_decay", torch.tensor(0.85, dtype=torch.float32)
            )
            self.model._v4_ema_complement = 0.15  # Python float, compile-time constant
            self.model.register_buffer(
                "_v4_max_mult", torch.tensor(2.5, dtype=torch.bfloat16)
            )
            self.model.register_buffer(
                "_v4_sig_center", torch.tensor(0.3, dtype=torch.float32)
            )
            self.model.register_buffer(
                "_v4_sig_steep", torch.tensor(4.0, dtype=torch.float32)
            )
            logger.info(
                f"[steering] DAS v4 momentum-adaptive decode enabled "
                f"(ema_decay=0.85, max_mult=2.5, per-request isolation, CUDA-graph safe)"
            )

    def determine_num_fused_shared_experts(self):
        if get_global_server_args().disable_shared_experts_fusion:
            return

        disable_reason = None
        if not getattr(self.config, "n_shared_experts", None):
            disable_reason = "No shared experts are defined in the config."
        elif not _is_cuda:
            disable_reason = "Shared experts fusion currently requires CUDA devices."
        elif _is_cuda and (_device_sm is not None) and (_device_sm < 80):
            disable_reason = "Shared experts fusion requires SM80 or newer GPUs."
        elif get_moe_expert_parallel_world_size() > 1:
            disable_reason = "Shared experts fusion is not supported together with expert parallelism yet."
        elif get_moe_a2a_backend().is_deepep():
            disable_reason = "Shared experts fusion is not supported when Deepep MoE backend is enabled."

        if disable_reason is not None:
            get_global_server_args().disable_shared_experts_fusion = True
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        self.num_fused_shared_experts = self.config.n_shared_experts
        assert (
            self.num_fused_shared_experts == 1
        ), "Only 1 fused shared expert is supported for Glm4MoeForCausalLM"
        log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                # compatible with old design
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names = [
                "shared_head.norm",
                "eh_proj",
                "enorm",
                "hnorm",
            ]

        params_dict = dict(self.named_parameters())
        weight_names = []
        for name, loaded_weight in weights:
            weight_names.append(name)

            if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
                # Map shared expert weights to the last expert slot
                # Shared expert becomes expert ID = n_routed_experts
                name = name.replace(
                    "mlp.shared_experts",
                    f"mlp.experts.{self.config.n_routed_experts}",
                )

            if not is_nextn:
                if hasattr(self.config, "num_nextn_predict_layers"):
                    num_nextn_layers = self.config.num_nextn_predict_layers
                    if num_nextn_layers > 0 and name.startswith("model.layers"):
                        name_list = name.split(".")
                        if (
                            len(name_list) >= 3
                            and int(name_list[2]) >= self.config.num_hidden_layers
                        ):
                            continue
            else:
                if not name.startswith(nextn_layer_prefix):
                    continue

                # Use shared head and embed weights from target model
                if "shared_head.head" in name or "embed_tokens" in name:
                    continue

                is_decoder = True
                # For nextn specific weights
                for weight_name in nextn_spec_weight_names:
                    if weight_name in name:
                        name = name.replace(nextn_layer_prefix, "model")
                        is_decoder = False
                        break
                # For decoder layer weights
                if is_decoder:
                    name = name.replace(nextn_layer_prefix, "model.decoder")

            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Track if this is an expert weight to enable early skipping
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Mark as expert weight regardless of whether we can process it
                    is_expert_weight = True

                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        # Expert weight not on this rank, will be skipped below
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")

        # Migrate steering buffers to GPU now that all model weights are loaded.
        # Must be here (not in forward()) so the CPU→GPU copy happens BEFORE
        # CUDA graph capture — otherwise the copy op gets embedded in the graph
        # and the source CPU tensor may be GC'd before graph replay.
        if (
            hasattr(self.model, "_steering_dir")
            and self.model._steering_dir is not None
        ):
            if self.model._steering_dir.device.type == "cpu":
                try:
                    _dev = next(self.parameters()).device
                    # Core steering buffers (v1 + v2)
                    for _buf_name in (
                        "_steering_dir",
                        "_steering_dirs",
                        "_steering_scales",
                        "_steering_attn_scales",
                        "_steering_mlp_scales",
                        "_steer_dec_tmp1",
                        "_steer_dec_tmp2",
                        "_steer_dec_proj",
                        "_steer_dec_scale",
                    ):
                        _buf = getattr(self.model, _buf_name, None)
                        if (
                            _buf is not None
                            and hasattr(_buf, "device")
                            and _buf.device.type == "cpu"
                        ):
                            setattr(
                                self.model,
                                _buf_name,
                                _buf.to(device=_dev, dtype=torch.bfloat16),
                            )
                    logger.info(
                        f"[steering] load_weights: migrated all steering buffers to {_dev}"
                    )
                except StopIteration:
                    pass  # no parameters yet, will migrate later

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.n_group,
        )

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            self.capture_aux_hidden_states = True
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.capture_aux_hidden_states = True
            # we plus 1 here because in sglang, for the ith layer, it takes the output
            # of the (i-1)th layer as aux hidden state
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    def determine_num_fused_shared_experts(self):
        super().determine_num_fused_shared_experts("GlmMoeDsaForCausalLM")


EntryClass = [Glm4MoeForCausalLM, GlmMoeDsaForCausalLM]
