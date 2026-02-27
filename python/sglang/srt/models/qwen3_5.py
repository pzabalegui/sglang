# Copyright 2025 Qwen Team
# Copyright 2025 SGLang Team
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
"""Inference-only Qwen3.5 model and Qwen3.5 MoE model compatible with HuggingFace weights."""
import logging
from functools import lru_cache
from typing import Iterable, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

# Model Executor

# Configs
from sglang.srt.configs.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3_5TextConfig,
)

# Distributed
from sglang.srt.distributed import get_pp_group
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation

# Layers - Attention
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.srt.layers.attention.mamba.mamba import mamba_v2_sharded_weight_loader
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)

# Layers - Others
from sglang.srt.layers.layernorm import GemmaRMSNorm

# Layers - Linear
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP, Qwen2MoeSparseMoeBlock

# Models
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration

# Utils
from sglang.srt.utils import add_prefix, is_cuda, is_npu, make_layers, set_weight_attrs
from sglang.srt.utils.hf_transformers_utils import get_processor

from sglang.srt.server_args import get_global_server_args
import math as _math
import json as _json
import os as _os

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_npu = is_npu()

cached_get_processor = lru_cache(get_processor)


# ============================================================
# ACTIVATION CAPTURE for refusal direction extraction
# ============================================================

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


def _maybe_capture(hidden_states, layer_idx, forward_batch, n_layers=64, residual=None):
    """Capture hidden states during prefill for refusal direction extraction."""
    cfg = _get_capture_config()
    if cfg is None:
        return

    if hidden_states.shape[0] <= 2:
        return

    try:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass

    target_layers = cfg.get("layers", list(range(n_layers)))
    if layer_idx not in target_layers:
        return

    last_h = hidden_states[-1, :]
    if residual is not None:
        last_h = last_h + residual[-1, :]
    _CAPTURE_STORE[layer_idx] = last_h.detach().cpu().to(torch.float32)

    max_target = max(target_layers)
    if layer_idx == max_target:
        save_dir = cfg.get("save_dir", "/tmp/captures")
        _os.makedirs(save_dir, exist_ok=True)
        sample_id = _CAPTURE_COUNTER[0]
        save_path = _os.path.join(save_dir, f"sample_{sample_id}.pt")
        torch.save(dict(_CAPTURE_STORE), save_path)
        _CAPTURE_STORE.clear()
        _CAPTURE_COUNTER[0] += 1


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.alt_stream = alt_stream

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_id = layer_id
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        # Conv1d layer
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            quant_config=None,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("conv1d", prefix),
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # Split projection layers (following vLLM's implementation)
        # Instead of fused in_proj_qkvz and in_proj_ba, use separate layers
        self.in_proj_qkv = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.key_dim, self.key_dim, self.value_dim],
            bias=False,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("in_proj_qkv", prefix),
        )
        self.in_proj_z = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.value_dim,
            bias=False,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("in_proj_z", prefix),
        )
        self.in_proj_b = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_v_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("in_proj_b", prefix),
        )
        self.in_proj_a = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_v_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("in_proj_a", prefix),
        )

        # Conv1d weight loader setup
        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": mamba_v2_sharded_weight_loader(
                    [
                        query_key_settings,
                        query_key_settings,
                        value_settings,
                    ],
                    self.attn_tp_size,
                    self.attn_tp_rank,
                )
            },
        )

        # State parameters
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.attn_tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(self.num_v_heads // self.attn_tp_size),
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        # RadixLinearAttention layer
        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.num_k_heads // self.attn_tp_size,
            num_k_heads=self.num_k_heads // self.attn_tp_size,
            num_v_heads=self.num_v_heads // self.attn_tp_size,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=conv_weights,
            bias=self.conv1d.bias,
            activation=self.activation,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

        # Normalization layer
        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=torch.get_device_module().current_device(),
            dtype=config.torch_dtype,
        )

        # Output projection
        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=False,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("out_proj", prefix),
        )

    def fix_query_key_value_ordering(
        self,
        mixed_qkv,
        z,
        b,
        a,
    ):
        raise NotImplementedError(
            "Qwen3.5 Series dont need to fix query key value ordering"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        seq_len, _ = hidden_states.shape

        mixed_qkv, _ = self.in_proj_qkv(hidden_states)
        z, _ = self.in_proj_z(hidden_states)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b, _ = self.in_proj_b(hidden_states)
        a, _ = self.in_proj_a(hidden_states)

        b = b.contiguous()
        a = a.contiguous()

        core_attn_out = self.attn.forward(
            forward_batch=forward_batch,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output, _ = self.out_proj(core_attn_out)
        return output


class Qwen3_5LinearDecoderLayer(nn.Module):
    """Qwen3.5 Decoder Layer with Linear Attention (GatedDeltaNet)."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.linear_attn = Qwen3_5GatedDeltaNet(
            config, layer_id, quant_config, alt_stream, prefix
        )

        # NOTE: Determine the MLP type based on the model type
        # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),
            )
            is_layer_sparse = True
            is_previous_layer_sparse = True
            is_next_layer_sparse = True
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),
            )
            is_layer_sparse = False
            is_previous_layer_sparse = False
            is_next_layer_sparse = False
        else:
            raise ValueError(f"Invalid model type: {config.model_type}")

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ):
        forward_batch = kwargs.get("forward_batch", None)
        steering_ctx = kwargs.get("steering_ctx", None)

        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.linear_attn(
                hidden_states,
                forward_batch,
            )

        # DAS v2: post-attention steering (prefill only)
        if steering_ctx is not None and steering_ctx.get("attn_active"):
            _layer_id = self.layer_id
            if _layer_id in steering_ctx["attn_set"]:
                _dirs_k = steering_ctx["dirs"][_layer_id]
                _s = steering_ctx["attn_scales"][_layer_id]
                _k = steering_ctx["k"]
                for _ki in range(_k):
                    _dir = _dirs_k[_ki]
                    _proj = (hidden_states * _dir).sum(dim=-1, keepdim=True)
                    _proj.clamp_(min=0)
                    hidden_states = hidden_states - _s * _proj * _dir

        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)

        # DAS v2: post-MLP steering (prefill only)
        if steering_ctx is not None and steering_ctx.get("mlp_active"):
            _layer_id = self.layer_id
            if _layer_id in steering_ctx["mlp_set"]:
                _dirs_k = steering_ctx["dirs"][_layer_id]
                _s = steering_ctx["mlp_scales"][_layer_id]
                _k = steering_ctx["k"]
                for _ki in range(_k):
                    _dir = _dirs_k[_ki]
                    _proj = (hidden_states * _dir).sum(dim=-1, keepdim=True)
                    _proj.clamp_(min=0)
                    hidden_states = hidden_states - _s * _proj * _dir

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


class Qwen3_5AttentionDecoderLayer(nn.Module):
    """Qwen3.5 Decoder Layer with Full Attention."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % self.attn_tp_size == 0
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.attn_tp_size:
            assert self.total_num_kv_heads % self.attn_tp_size == 0
        else:
            assert self.attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        if hasattr(config, "rope_parameters"):
            self.rope_scaling = getattr(config, "rope_parameters", None)
        else:
            self.rope_scaling = getattr(config, "rope_scaling", None)

        self.rope_theta = self.rope_scaling.get("rope_theta", 10000)
        self.partial_rotary_factor = self.rope_scaling.get("partial_rotary_factor", 1.0)
        self.layer_id = layer_id

        self.attn_output_gate = getattr(config, "attn_output_gate", True)
        if self.attn_output_gate:
            logger.warning_once("using attn output gate!")

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            rope_scaling=self.rope_scaling,
            base=self.rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("o_proj", prefix),
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=f"{prefix}.attn",
        )

        # Dense MLP for non-MoE variant
        if config.model_type == "qwen3_5_text":
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),
            )
            is_layer_sparse = False
            is_previous_layer_sparse = False
            is_next_layer_sparse = False
        elif config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),
            )
            is_layer_sparse = True
            is_previous_layer_sparse = True
            is_next_layer_sparse = True
        else:
            raise ValueError(f"Invalid model type: {config.model_type}")

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
        )

        self.alt_stream = alt_stream

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Q/K normalization with optional alt_stream overlap."""
        if self.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            with torch.cuda.stream(self.alt_stream):
                k_by_head = k.reshape(-1, self.head_dim)
                k_by_head = self.k_norm(k_by_head)
            current_stream.wait_stream(self.alt_stream)
        else:
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            k_by_head = k.reshape(-1, self.head_dim)
            k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        return q, k

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Full attention forward pass."""
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        steering_ctx = kwargs.get("steering_ctx", None)

        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.self_attention(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        # DAS v2: post-attention steering (prefill only)
        if steering_ctx is not None and steering_ctx.get("attn_active"):
            _layer_id = self.layer_id
            if _layer_id in steering_ctx["attn_set"]:
                _dirs_k = steering_ctx["dirs"][_layer_id]
                _s = steering_ctx["attn_scales"][_layer_id]
                _k = steering_ctx["k"]
                for _ki in range(_k):
                    _dir = _dirs_k[_ki]
                    _proj = (hidden_states * _dir).sum(dim=-1, keepdim=True)
                    _proj.clamp_(min=0)
                    hidden_states = hidden_states - _s * _proj * _dir

        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)

        # DAS v2: post-MLP steering (prefill only)
        if steering_ctx is not None and steering_ctx.get("mlp_active"):
            _layer_id = self.layer_id
            if _layer_id in steering_ctx["mlp_set"]:
                _dirs_k = steering_ctx["dirs"][_layer_id]
                _s = steering_ctx["mlp_scales"][_layer_id]
                _k = steering_ctx["k"]
                for _ki in range(_k):
                    _dir = _dirs_k[_ki]
                    _proj = (hidden_states * _dir).sum(dim=-1, keepdim=True)
                    _proj.clamp_(min=0)
                    hidden_states = hidden_states - _s * _proj * _dir

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3_5AttentionDecoderLayer,
    "linear_attention": Qwen3_5LinearDecoderLayer,
}


class Qwen3_5ForCausalLM(nn.Module):
    """Qwen3.5 Model with support for dense variant."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.pp_group = get_pp_group()

        alt_stream = torch.cuda.Stream() if _is_cuda else None

        # Embedding layer
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                enable_tp=not is_dp_attention_enabled(),
            )

        # Decoder layers
        def get_layer(idx: int, prefix: str):
            layer_type = config.layers_block_type[idx]
            layer_class = ALL_DECODER_LAYER_TYPES[layer_type]
            if layer_type == "attention":
                prefix = add_prefix("self_attn", prefix)
            else:
                prefix = add_prefix("linear_attn", prefix)
            return layer_class(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            )

        self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )

        # Final normalization
        if self.pp_group.is_last_rank:
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_deepstack_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        # Initialize hidden states
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

        # --- DAS v1/v2/v4 steering setup ---
        _is_prefill = not forward_batch.forward_mode.is_decode()
        _steering_off = getattr(forward_batch, "steering_disabled", False)

        # DAS v2: multi-point steering context (attn + MLP intervention)
        _steering_v2 = (
            getattr(self, "_steering_v2", False) and _is_prefill and not _steering_off
        )
        _steering_k = getattr(self, "_steering_k", 1)

        # Post-layer steering (v1 compatible)
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
                "dirs": self._steering_dirs,
                "attn_scales": self._steering_attn_scales,
                "mlp_scales": self._steering_mlp_scales,
                "attn_set": _attn_set,
                "mlp_set": _mlp_set,
                "attn_active": len(_attn_set) > 0,
                "mlp_active": len(_mlp_set) > 0,
                "k": _steering_k,
            }

        # Clamped projective decode steering (CUDA-graph safe)
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

        # Per-request decode scale override
        _saved_dec_scale_for_override = None
        _saved_dec_scales_for_override = None
        _ds_override = getattr(forward_batch, "steering_decode_scale_override", None)
        if _has_decode_steering and _ds_override is not None:
            if _has_multi_layer_decode:
                _old_global = self._steer_dec_scale.item()
                if _old_global > 0:
                    _ratio = float(_ds_override) / _old_global
                    _saved_dec_scales_for_override = self._steer_dec_scales.clone()
                    self._steer_dec_scales.mul_(_ratio)
                _saved_dec_scale_for_override = _old_global
                self._steer_dec_scale.fill_(float(_ds_override))
            else:
                _saved_dec_scale_for_override = self._steer_dec_scale.item()
                self._steer_dec_scale.fill_(float(_ds_override))

        # Pass through decoder layers
        for layer_idx in range(len(self.layers)):
            layer = self.layers[layer_idx]
            with get_global_expert_distribution_recorder().with_current_layer(
                layer_idx
            ):
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    forward_batch=forward_batch,
                    steering_ctx=_steering_ctx,
                )

            # POST-LAYER projective steering (prefill-only, v1 compatible)
            if (
                _has_steering
                and _steered_layers is not None
                and layer_idx in _steered_layers
            ):
                _scale = self._steering_scales[layer_idx]
                _dirs_layer = (
                    self._steering_dirs[layer_idx]
                    if hasattr(self, "_steering_dirs")
                    else self._steering_dir.unsqueeze(0)
                )
                for _ki in range(_steering_k):
                    _dir = _dirs_layer[_ki]
                    _proj = (hidden_states * _dir).sum(dim=-1, keepdim=True)
                    _proj.clamp_(min=0)
                    hidden_states = hidden_states - _scale * _proj * _dir

            # Clamped projective decode steering (CUDA-graph safe)
            if _has_decode_steering and residual is not None:
                _should_decode_steer = (
                    _has_multi_layer_decode and layer_idx in _decode_steered_set
                ) or (
                    not _has_multi_layer_decode and layer_idx == _decode_peak_layer
                )
                if _should_decode_steer:
                    _bs = hidden_states.shape[0]
                    _t1 = self._steer_dec_tmp1[:_bs]
                    _t2 = self._steer_dec_tmp2[:_bs]
                    _p = self._steer_dec_proj[:_bs]
                    _dec_scale_i = (
                        self._steer_dec_scales[layer_idx]
                        if _has_multi_layer_decode
                        else self._steer_dec_scale
                    )
                    torch.add(hidden_states, residual, out=_t1)
                    for _ki in range(_steering_k):
                        _dir_ki = self._steering_dirs[layer_idx][_ki]
                        torch.mul(_t1, _dir_ki, out=_t2)
                        _p.copy_(_t2.sum(dim=-1, keepdim=True))
                        _p.clamp_(min=0)
                        if _v4_adaptive:
                            # DAS v4: per-request momentum-adaptive (CUDA-graph safe)
                            _mom = self._steer_momentum[:_bs]
                            _mom.mul_(self._v4_ema_decay).add_(_p, alpha=0.15)
                            _stmp = self._v4_sig_tmp[:_bs]
                            torch.sub(_mom, self._v4_sig_center, out=_stmp)
                            _stmp.mul_(self._v4_sig_steep)
                            _sres = self._v4_sig_result[:_bs]
                            torch.sigmoid(_stmp, out=_sres)
                            torch.mul(_p, _dir_ki, out=_t2)
                            _t2.mul_(_dec_scale_i)
                            _t2.mul_(_sres)
                            _t2.mul_(self._v4_max_mult)
                            _t2.mul_(self._steering_mask[:_bs])
                        else:
                            torch.mul(_p, _dir_ki, out=_t2)
                            _t2.mul_(_dec_scale_i)
                        hidden_states.sub_(_t2)
                        if _ki < _steering_k - 1:
                            torch.add(hidden_states, residual, out=_t1)

            _maybe_capture(
                hidden_states,
                layer_idx,
                forward_batch,
                n_layers=len(self.layers),
                residual=residual,
            )

            # Process deepstack embeddings if provided
            if (
                input_deepstack_embeds is not None
                and input_deepstack_embeds.numel() > 0
                and layer_idx < 3
            ):
                sep = self.hidden_size * layer_idx
                hidden_states.add_(
                    input_deepstack_embeds[:, sep : sep + self.hidden_size]
                )

        # Restore decode scale after per-request override
        if _saved_dec_scale_for_override is not None:
            self._steer_dec_scale.fill_(_saved_dec_scale_for_override)
        if _saved_dec_scales_for_override is not None:
            self._steer_dec_scales.copy_(_saved_dec_scales_for_override)

        # Return intermediate tensors for pipeline parallelism
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        # Apply final normalization
        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "visual" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                # if is_pp_missing_parameter(name, self):
                #     continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning(f"Parameter {name} not found in params_dict")
                    continue
                param = params_dict[name]

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        num_experts = self.config.num_experts

        def load_fused_expert_weights(
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ):
            param = params_dict[name]
            weight_loader = param.weight_loader
            # let ep moe layer to gracefully handle expert_ids that do not belong to local moe rank
            for expert_id in range(num_experts):
                curr_expert_weight = loaded_weight[expert_id]
                weight_loader(
                    param,
                    curr_expert_weight,
                    name,
                    shard_id,
                    expert_id,
                )
            return True

        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "visual" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

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
                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
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
                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                        else:
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                    else:
                        # Skip loading extra parameters for GPTQ/modelopt models.
                        if (
                            name_mapped.endswith(ignore_suffixes)
                            and name_mapped not in params_dict
                        ):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # # other available replicas.
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")
            loaded_params.add(name)

        return loaded_params


class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(
        self,
        config: Qwen3_5Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen3_5ForCausalLM,
    ):
        super().__init__(config, quant_config, prefix, language_model_cls)

        rope_config = getattr(self.config, "rope_parameters", None) or getattr(
            self.config, "rope_scaling", {}
        )
        self.is_mrope_enabled = "mrope_section" in rope_config

        self.deepstack_visual_indexes = self.visual.deepstack_visual_indexes

        # Initialize GPU-native steering buffers (CUDA-graph compatible)
        self._init_steering()

    def _init_steering(self) -> None:
        """Initialize GPU-native steering buffers for CUDA-graph compatible DAS.

        Registers all steering tensors on self.model (Qwen3_5ForCausalLM),
        which owns the layer loop and the steering forward logic.

        Supports DAS v1 (single vector), v2 (per-layer attn+MLP), v3 (SVD multi-vector),
        and v4 (momentum-adaptive decode with per-request isolation).
        """
        args = get_global_server_args()
        if not getattr(args, "steering_vector_path", None):
            return

        text_config = getattr(self.config, "text_config", self.config)
        n_layers = text_config.num_hidden_layers
        hid = text_config.hidden_size
        n_dirs = int(getattr(args, "steering_n_directions", 1))

        # Detect target device: model.embed_tokens should already be on CUDA
        # during SGLang initialization (inside `with target_device:` context)
        _target_device = next(self.model.parameters()).device
        logger.info(f"[steering] Target device for buffers: {_target_device}")

        # --- Load steering vectors ---
        per_layer_path = getattr(args, "steering_per_layer_path", None)
        if per_layer_path:
            per_layer_vecs = torch.load(
                per_layer_path, map_location=_target_device, weights_only=True
            )
            if per_layer_vecs.dim() == 2:
                if per_layer_vecs.shape[0] != n_layers:
                    raise ValueError(
                        f"Per-layer vectors must have {n_layers} layers, "
                        f"got {per_layer_vecs.shape[0]}"
                    )
                per_layer_vecs = per_layer_vecs.unsqueeze(1)
                logger.info(
                    f"[steering] Upgraded 2D vectors to 3D: {per_layer_vecs.shape}"
                )
            elif per_layer_vecs.dim() == 3:
                if per_layer_vecs.shape[0] != n_layers:
                    raise ValueError(
                        f"Per-layer vectors must have {n_layers} layers, "
                        f"got {per_layer_vecs.shape[0]}"
                    )
            else:
                raise ValueError(
                    f"Per-layer vectors must be 2D or 3D, "
                    f"got {per_layer_vecs.dim()}D: {per_layer_vecs.shape}"
                )
            k = min(n_dirs, per_layer_vecs.shape[1])
            per_layer_vecs = per_layer_vecs[:, :k, :]
            norms = per_layer_vecs.norm(dim=2, keepdim=True).clamp(min=1e-8)
            per_layer_vecs = per_layer_vecs / norms
            logger.info(
                f"[steering] Loaded per-layer directions: "
                f"{per_layer_vecs.shape} (k={k})"
            )
            self.model._steering_per_layer = True
        else:
            vec = torch.load(
                args.steering_vector_path, map_location=_target_device, weights_only=True
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

        self.model._steering_k = k
        self.model.register_buffer("_steering_dirs", per_layer_vecs.bfloat16())

        # Global direction for v1 decode steering compatibility
        steering_layers_str = getattr(args, "steering_layers", None)
        center_layers = (
            _json.loads(steering_layers_str) if steering_layers_str else [32]
        )
        peak_layer = int(center_layers[0]) if center_layers else 32
        self.model.register_buffer(
            "_steering_dir", per_layer_vecs[peak_layer, 0].clone().bfloat16()
        )

        # --- Compute layer weights (kernel) ---
        mode = getattr(args, "steering_mode", "gaussian")
        kernel = getattr(args, "steering_kernel", mode)
        base_scale = float(getattr(args, "steering_scale", 5.0))
        sigma = float(getattr(args, "steering_kernel_width", 2.0))
        attn_scale = float(getattr(args, "steering_attn_scale", 0.0))
        mlp_scale = float(getattr(args, "steering_mlp_scale", 0.0))

        self.model._steering_v2 = attn_scale > 0.0 or mlp_scale > 0.0

        layer_weights = torch.zeros(n_layers, dtype=torch.float32)

        if kernel == "trapezoidal":
            trap_start = int(getattr(args, "steering_trap_start", 21))
            trap_end = int(getattr(args, "steering_trap_end", 45))
            trap_ramp = int(getattr(args, "steering_trap_ramp", 4))
            for i in range(n_layers):
                if trap_start + trap_ramp <= i <= trap_end - trap_ramp:
                    layer_weights[i] = 1.0
                elif trap_start <= i < trap_start + trap_ramp:
                    layer_weights[i] = (i - trap_start) / max(trap_ramp, 1)
                elif trap_end - trap_ramp < i <= trap_end:
                    layer_weights[i] = (trap_end - i) / max(trap_ramp, 1)
            logger.info(
                f"[steering] Trapezoidal kernel: [{trap_start}, {trap_end}] "
                f"ramp={trap_ramp}"
            )
        else:
            for c in center_layers:
                for i in range(n_layers):
                    w = _math.exp(-0.5 * ((i - c) / sigma) ** 2)
                    if w > layer_weights[i].item():
                        layer_weights[i] = w
            logger.info(
                f"[steering] Gaussian kernel: centers={center_layers}, sigma={sigma}"
            )

        scales = layer_weights * base_scale
        attn_scales = layer_weights * attn_scale
        mlp_scales = layer_weights * mlp_scale

        self.model.register_buffer("_steering_scales", scales.bfloat16())
        self.model.register_buffer("_steering_attn_scales", attn_scales.bfloat16())
        self.model.register_buffer("_steering_mlp_scales", mlp_scales.bfloat16())

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
        decode_scale = float(getattr(args, "steering_decode_scale", 0.0))
        self.model._decode_steer_peak_layer = peak_layer
        if decode_scale > 0.0:
            max_bs = getattr(args, "cuda_graph_max_bs", 128)
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
                "_steer_dec_scale",
                torch.tensor(decode_scale, dtype=torch.bfloat16),
            )

            decode_layers_str = getattr(args, "steering_decode_layers", None)
            if decode_layers_str:
                decode_layers = _json.loads(decode_layers_str)
            else:
                decode_layers = [peak_layer]

            decode_scales = torch.zeros(n_layers, dtype=torch.float32)
            for dl in decode_layers:
                if 0 <= dl < n_layers:
                    w = layer_weights[dl].item()
                    decode_scales[dl] = decode_scale * max(w, 0.1)

            self.model.register_buffer(
                "_steer_dec_scales", decode_scales.bfloat16()
            )
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
            f"[steering] DAS "
            f"{'v3' if k > 1 else ('v2' if self.model._steering_v2 else 'v1')} "
            f"initialized: kernel={kernel}, k={k}, "
            f"base_scale={base_scale}, attn_scale={attn_scale}, "
            f"mlp_scale={mlp_scale}, "
            f"post-layer active={n_active}/{n_layers}, "
            f"post-attn active={n_attn_active}/{n_layers}, "
            f"post-MLP active={n_mlp_active}/{n_layers}, "
            f"decode_scale={decode_scale}"
        )

        # --- DAS v4: Momentum-adaptive decode steering ---
        self.model._v4_adaptive = decode_scale > 0.0
        if self.model._v4_adaptive:
            max_bs = getattr(args, "cuda_graph_max_bs", 128)
            # Per-request buffers [max_bs, 1]
            self.model.register_buffer(
                "_steer_momentum", torch.zeros(max_bs, 1, dtype=torch.float32)
            )
            self.model.register_buffer(
                "_v4_sig_tmp", torch.zeros(max_bs, 1, dtype=torch.float32)
            )
            self.model.register_buffer(
                "_v4_sig_result", torch.zeros(max_bs, 1, dtype=torch.float32)
            )
            # Per-request steering mask (1.0=ON, 0.0=OFF)
            self.model.register_buffer(
                "_steering_mask", torch.ones(max_bs, 1, dtype=torch.bfloat16)
            )
            # EMA config as registered buffers
            self.model.register_buffer(
                "_v4_ema_decay", torch.tensor(0.85, dtype=torch.float32)
            )
            self.model._v4_ema_complement = 0.15
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
                f"(ema_decay=0.85, max_mult=2.5, per-request isolation, "
                f"CUDA-graph safe)"
            )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "visual" in name or "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                # if is_pp_missing_parameter(name, self):
                #     continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "visual" in name:
                    # adapt to VisionAttention
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                    name = name.replace(r"model.visual.", r"visual.")

                # print(name, loaded_weight.shape)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning(f"Parameter {name} not found in params_dict")
                    continue
                param = params_dict[name]

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3_5MoeForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Qwen3.5 MoE Vision-Language Model."""

    def __init__(
        self,
        config: Qwen3_5MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen3_5MoeForCausalLM,
    ) -> None:
        super().__init__(config, quant_config, prefix, language_model_cls)
        rope_config = getattr(self.config, "rope_parameters", None) or getattr(
            self.config, "rope_scaling", {}
        )
        self.is_mrope_enabled = "mrope_section" in rope_config

        self.deepstack_visual_indexes = self.visual.deepstack_visual_indexes

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        num_experts = self.config.num_experts

        def load_fused_expert_weights(
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ):
            param = params_dict[name]
            weight_loader = param.weight_loader
            # let ep moe layer to gracefully handle expert_ids that do not belong to local moe rank
            for expert_id in range(num_experts):
                curr_expert_weight = loaded_weight[expert_id]
                weight_loader(
                    param,
                    curr_expert_weight,
                    name,
                    shard_id,
                    expert_id,
                )
            return True

        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                if "visual" in name:
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
                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
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
                    if "visual" in name or self.config.encoder_only:
                        continue
                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                        else:
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                    else:
                        # Skip loading extra parameters for GPTQ/modelopt models.
                        if (
                            name_mapped.endswith(ignore_suffixes)
                            and name_mapped not in params_dict
                        ):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # # other available replicas.
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        continue

                    if "visual" in name:
                        # adapt to VisionAttention
                        name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                        name = name.replace(r"model.visual.", r"visual.")

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")
            loaded_params.add(name)

        return loaded_params

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        text_config = getattr(config, "text_config", config)
        return ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=text_config.num_experts,
            num_groups=None,
        )


EntryClass = [Qwen3_5MoeForConditionalGeneration, Qwen3_5ForConditionalGeneration]
