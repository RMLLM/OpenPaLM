# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
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

"""Transformer."""
import math
import torch
import torch.nn.functional as F
from torch import nn
from packaging.version import Version

from megatron import get_args, logging
from megatron import mpu
from .module import MegatronModule
from megatron.enums import AttnMaskType, LayerType, AttnType, PositionEmbeddingType
from megatron.model.fused_layer_norm import MixedFusedLayerNorm as LayerNorm
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.fused_gelu import gelu_impl
from megatron.model.utils import attention_mask_func, openai_gelu, erf_gelu, get_linear_layer

import deepspeed

from .glu_activations import GLU_ACTIVATIONS
from .positional_embeddings import RotaryEmbedding, apply_rotary_pos_emb_torch, apply_rotary_pos_emb

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

logger = logging.get_logger(__name__)

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    import flash_attn as _flash_attn
    from flash_attn.flash_attn_interface import flash_attn_func
except ImportError:
    flash_attn_func = None
    
try:
    assert Version(getattr(_flash_attn, "__version__", "1")) >= Version("2")    
except:
    print("Only supports flash version 2 or later. Flash version 1 is not supported.")


""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""

class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self, init_method, output_layer_init_method, parallel_output=False):
        super(ParallelMLP, self).__init__()
        args = get_args()

        # Project to ffn_hidden_size
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            args.hidden_size,
            # GLU is a special activation that divides the dimension by a factor 2.
            2 * args.ffn_hidden_size if args.glu_activation else args.ffn_hidden_size,
            bias=args.add_bias_linear,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True)

        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.gelu_fusion = args.gelu_fusion

        self.activation_func = F.gelu
        if args.glu_activation:
            self.activation_func = GLU_ACTIVATIONS[args.glu_activation]
        elif args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu

        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            args.ffn_hidden_size,
            args.hidden_size,
            bias=args.add_bias_linear,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output)

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if bias_parallel is not None:
            if self.bias_gelu_fusion:
                intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel)
        else:
            if self.gelu_fusion:
                intermediate_parallel = gelu_impl(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias
    

# Copied from https://github.com/bigcode-project/Megatron-LM/blob/multi-query-attention/megatron/model/transformer.py#L503-L583
class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        assert flash_attn_func is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """
        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        assert all((i.is_cuda for i in (q,k,v)))

        seqlen_q, seqlen_k = q.shape[1], k.shape[1]

        if self.training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q
            is_causal = self.causal
            dropout_p = self.dropout_p
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = self.causal and (seqlen_q == seqlen_k)
            dropout_p = 0

        output = flash_attn_func(q, k, v, dropout_p,softmax_scale=self.softmax_scale, causal=is_causal)

        return output


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding,
                 parallel_output=False):
        super(ParallelAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16
        self.position_embedding_type = args.position_embedding_type

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.attention_head_type = args.attention_head_type
        self.use_flash_attn = args.use_flash_attn

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(projection_size,
                                                    world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(
            args.num_attention_heads, world_size)

        # Strided linear layer.
        if attention_type == AttnType.self_attn and self.attention_head_type == 'multihead':
            self.query_key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                3 * projection_size,
                bias=args.add_bias_linear,
                gather_output=False,
                init_method=init_method,
                skip_bias_add=False)
        elif attention_type == AttnType.self_attn and self.attention_head_type == 'multiquery':
            # TODO: Find a way to merge the query and key-value computations?
            self.query = mpu.ColumnParallelLinear(
                args.hidden_size,
                projection_size,
                bias=args.add_bias_linear,
                gather_output=False,
                init_method=init_method,
                skip_bias_add=False)
            # In MultiQuery attention, keys and values are shared across heads
            # Use args.kv_channels instead of projection_size
            # No `.fork()` so the rng tracker is shared across tensor-parallel processes.
            # with mpu.get_cuda_rng_tracker():
            self.key_value = get_linear_layer(
                args.hidden_size,
                2 * args.kv_channels,
                bias=args.add_bias_linear,
                init_method=init_method)
        elif attention_type == AttnType.cross_attn and self.attention_head_type == 'multihead':
            # NOTE: Below code block is not used.
            self.query = mpu.ColumnParallelLinear(
                args.hidden_size,
                projection_size,
                gather_output=False,
                init_method=init_method)

            self.key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                2 * projection_size,
                gather_output=False,
                init_method=init_method)
        elif attention_type == AttnType.cross_attn and self.attention_head_type == 'multiquery':
            raise NotImplementedError("Multiquery attention not implemented for cross-attention.")
        else:
            raise ValueError(f"Invalid attention arguments: {attention_type}, {self.attention_head_type}")
            
        if self.use_flash_attn:
            if flash_attn_func is None:
                raise ImportError('FlashAttention is not installed, please install with '
                                  'pip install flash-attn')
            assert attention_type == AttnType.self_attn, ('FlashAttention code path only supports '
                                                          'self-attention for now')
            assert self.attn_mask_type == AttnMaskType.causal, ('FlashAttention code path only '
                                                                'supports causal mask for now')
            # assert args.position_embedding_type != PositionEmbeddingType.alibi, \
            #     ('FlashAttention does not support alibi positional embeddings yet')
            if rearrange is None:
                raise ImportError('einops is not installed, please install with pip install einops')

            self.core_attention_flash = FlashSelfAttention(
                causal=True, attention_dropout=args.attention_dropout
            )

        else:
            # The following features are applicable only when flash attention is disabled.
            coeff = None
            self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
            if self.apply_query_key_layer_scaling:
                coeff = self.layer_number
                self.norm_factor *= coeff

            self.scale_mask_softmax = FusedScaleMaskSoftmax(
                self.fp16, self.bf16,
                self.attn_mask_type,
                args.masked_softmax_fusion,
                attention_mask_func,
                self.attention_softmax_in_fp32,
                coeff)

            # Dropout. Note that for a single iteration, this layer will generate
            # different outputs on different number of parallel partitions but
            # on average it should not be partition dependent.
            self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

        # Output.
        self.dense = mpu.RowParallelLinear(
            projection_size,
            args.hidden_size,
            bias=args.add_bias_linear,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

        if self.position_embedding_type == PositionEmbeddingType.rotary:
            self.rotary_emb = RotaryEmbedding(
                    dim=self.hidden_size_per_attention_head,
                    max_position_embeddings=args.max_position_embeddings,
                    base=args.rope_theta,
            )

    # def forward(self, hidden_states, attention_mask, layer_past=None,
    #             get_key_value=False, encoder_output=None, alibi=None):
    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False, encoder_output=None):
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn and self.attention_head_type == 'multihead':
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer,
             key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)
        elif self.attention_type == AttnType.self_attn and self.attention_head_type == 'multiquery':
            kv_input=hidden_states
            # Attention heads [sq, b, h] --> [sq, b, (2 * hn)]
            mixed_kv_layer = self.key_value(kv_input)
            mixed_kv_layer = mpu.copy_to_tensor_model_parallel_region(mixed_kv_layer)

            # [sq, b, (2 * hn)] --> [sq, b, 1, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                (1,
                 2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sq, b, 1, 2 * hn] --> 2 [sq, b, 1, hn]
            (key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, np * hn]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, np * hn] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)

            # [sq, b, np, hn] -> [b, np * sq, hn]
        else:
            # NOTE: Below code block is not used.
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        if get_key_value:
            present = (key_layer, value_layer)

        # Rotary embeddings
        if self.position_embedding_type == PositionEmbeddingType.rotary:
            apply_rotary_fn = apply_rotary_pos_emb_torch if self.bf16 else apply_rotary_pos_emb

            seq_len = key_layer.shape[0]
            offset = 0
            if layer_past is not None and layer_past.numel() > 0:
                offset = layer_past[0].shape[0]
                seq_len += offset
            cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
            query_layer, key_layer = apply_rotary_fn(query_layer, key_layer, cos, sin, offset=offset)

        if self.use_flash_attn:
            q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous()
                       for x in (query_layer, key_layer, value_layer)]
            with mpu.get_cuda_rng_tracker().fork():
                context_layer = self.core_attention_flash(q, k, v)
            context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()
        else:
            if self.attention_head_type == "multihead":
                # ===================================
                # Raw attention scores. [b, np, s, s]
                # ===================================

                # [b, np, sq, sk]
                output_size = (query_layer.size(1),
                            query_layer.size(2),
                            query_layer.size(0),
                            key_layer.size(0))

                # [sq, b, np, hn] -> [sq, b * np, hn]
                query_layer = query_layer.view(output_size[2],
                                            output_size[0] * output_size[1], -1)
                # [sk, b, np, hn] -> [sk, b * np, hn]
                key_layer = key_layer.view(output_size[3],
                                        output_size[0] * output_size[1], -1)

                # if alibi is None:
                #     # preallocting result tensor: [b * np, sq, sk]
                #     matmul_result = torch.empty(
                #         output_size[0]*output_size[1],
                #         output_size[2],
                #         output_size[3],
                #         dtype=query_layer.dtype,
                #         device=torch.cuda.current_device())

                #     # Raw attention scores. [b * np, sq, sk]
                #     matmul_result = torch.baddbmm(
                #         matmul_result,
                #         query_layer.transpose(0, 1),   # [b * np, sq, hn]
                #         key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                #         beta=0.0, alpha=(1.0/self.norm_factor))
                # else:
                #     if not hasattr(self, "logged_alibi"):
                #         logger.debug("Using Alibi.")
                #         self.logged_alibi = True

                #     if self.apply_query_key_layer_scaling:
                #         beta = 1.0 / self.layer_number
                #     else:
                #         beta = 1.0

                #     # preallocting result tensor: [b * np, sq, sk]
                #     matmul_result = alibi[:output_size[0]*output_size[1], :, :output_size[3]]

                #     # Raw attention scores. [b * np, sq, sk]
                #     matmul_result = torch.baddbmm(
                #         matmul_result,
                #         query_layer.transpose(0, 1),  # [b * np, sq, hn]
                #         key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                #         beta=beta, alpha=(1.0 / self.norm_factor))
            
                # preallocting result tensor: [b * np, sq, sk]
                matmul_result = torch.empty(
                    output_size[0]*output_size[1],
                    output_size[2],
                    output_size[3],
                    dtype=query_layer.dtype,
                    device=torch.cuda.current_device())

                # Raw attention scores. [b * np, sq, sk]
                matmul_result = torch.baddbmm(
                    matmul_result,
                    query_layer.transpose(0, 1),   # [b * np, sq, hn]
                    key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                    beta=0.0, alpha=(1.0/self.norm_factor))

                # change view to [b, np, sq, sk]
                attention_scores = matmul_result.view(*output_size)
                # ==================================================
                # Update attention mask for inference. [b, np, sq, sk]
                # ==================================================

                if get_key_value:
                    with torch.no_grad():
                        # TODO @thomasw21 Handle case where `attention_mask` is None
                        if layer_past is not None:
                            attention_mask = attention_mask[
                                ...,
                                attention_scores.size(3) - 1,
                                :attention_scores.size(3)].unsqueeze(2)
                        else:
                            attention_mask = attention_mask[
                                ...,
                                :attention_scores.size(3),
                                :attention_scores.size(3)]

                # ===========================
                # Attention probs and dropout
                # ===========================

                # attention scores and attention mask [b, np, sq, sk]
                attention_probs = self.scale_mask_softmax(attention_scores,
                                                        attention_mask)

                # This is actually dropping out entire tokens to attend to, which might
                # seem a bit unusual, but is taken from the original Transformer paper.
                with mpu.get_cuda_rng_tracker().fork():
                    attention_probs = self.attention_dropout(attention_probs)

                # =========================
                # Context layer. [sq, b, hp]
                # =========================

                # value_layer -> context layer.
                # [sk, b, np, hn] --> [b, np, sq, hn]

                # context layer shape: [b, np, sq, hn]
                output_size = (value_layer.size(1),
                            value_layer.size(2),
                            query_layer.size(0),
                            value_layer.size(3))

                # change view [sk, b * np, hn]
                value_layer = value_layer.view(value_layer.size(0),
                                            output_size[0] * output_size[1], -1)

                # change view [b * np, sq, sk]
                attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                                    output_size[2], -1)

                # matmul: [b * np, sq, hn]
                context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

                # change view [b, np, sq, hn]
                context_layer = context_layer.view(*output_size)

                # [b, np, sq, hn] --> [sq, b, np, hn]
                context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

                # [sq, b, np, hn] --> [sq, b, hp]
                new_context_layer_shape = context_layer.size()[:-2] + \
                    (self.hidden_size_per_partition,)
                context_layer = context_layer.view(*new_context_layer_shape)

            # Copied from https://github.com/bigcode-project/Megatron-LM/blob/multi-query-attention/megatron/model/transformer.py#L382-L500
            elif self.attention_head_type == "multiquery":
                # Only one head for key and values
                assert key_layer.size(2) == 1 and value_layer.size(2) == 1
                
                sq = query_layer.size(0)
                bs = query_layer.size(1)
                np = query_layer.size(2)
                sk = key_layer.size(0)
                
                # [sq, b, np, hn] -> [b, np * sq, hn]
                query_layer = query_layer.permute([1, 2, 0, 3]).reshape(bs, np * sq, -1)
                # [sk, b, 1, hn] -> [b, hn, sk]
                key_layer = key_layer.squeeze(2).permute(1, 2, 0)
                
                # preallocting result tensor: [b, np * sq, sk]
                matmul_result = torch.empty(
                    bs,
                    np * sq,
                    sk,
                    dtype=query_layer.dtype,
                    device=torch.cuda.current_device())
                
                # Raw attention scores. [b, np * sq, sk]
                matmul_result = torch.baddbmm(
                    matmul_result,
                    query_layer,   # [b, np * sq, hn]
                    key_layer,  # [b, hn, sk]
                    beta=0.0, alpha=(1.0/self.norm_factor))
                
                # change view to [b, np, sq, sk]
                attention_scores = matmul_result.view(bs, np, sq, sk)
                # ==================================================
                # Update attention mask for inference. [b, np, sq, sk]
                # ==================================================

                if get_key_value:
                    with torch.no_grad():
                        # TODO @thomasw21 Handle case where `attention_mask` is None
                        if layer_past is not None:
                            attention_mask = attention_mask[
                                ...,
                                attention_scores.size(3) - 1,
                                :attention_scores.size(3)].unsqueeze(2)
                        else:
                            attention_mask = attention_mask[
                                ...,
                                :attention_scores.size(3),
                                :attention_scores.size(3)]

                # ===========================
                # Attention probs and dropout
                # ===========================

                # attention scores and attention mask [b, np, sq, sk]
                attention_probs = self.scale_mask_softmax(attention_scores,
                                                        attention_mask)

                # This is actually dropping out entire tokens to attend to, which might
                # seem a bit unusual, but is taken from the original Transformer paper.
                with mpu.get_cuda_rng_tracker().fork():
                    attention_probs = self.attention_dropout(attention_probs)

                # =========================
                # Context layer. [sq, b, hp]
                # =========================

                # value_layer -> context layer.
                # [sk, b, np, hn] --> [b, np, sq, hn]

                # [sk, b, 1, hn] -> [b, sk, hn]
                value_layer = value_layer.squeeze(2).transpose(0, 1)

                # change view [b, np * sq, sk]
                attention_probs = attention_probs.view(bs, np * sq, -1)

                # matmul: [b, np * sq, hn]
                context_layer = torch.bmm(attention_probs, value_layer)

                # change view [b, np, sq, hn]
                context_layer = context_layer.view(bs, np, sq, -1)

                # [b, np, sq, hn] --> [sq, b, np, hn]
                context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

                # [sq, b, np, hn] --> [sq, b, hp]
                new_context_layer_shape = context_layer.size()[:-2] + \
                    (self.hidden_size_per_partition,)
                context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output, bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, False)


def dropout_add(x, tensor, prob, training):
    # type: (Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x, p=prob, training=training) + tensor
    return out


def get_dropout_add(training):
    def _dropout_add(x, tensor, prob):
        return dropout_add(x, tensor, prob, training)
    return _dropout_add


@torch.jit.script
def dropout_add_fused_train(x, tensor, prob: float):
    # type: (Tensor, Tensor, float) -> Tensor
    return dropout_add(x, tensor, prob, True)


@torch.jit.script
def dropout_add_fused_inference(x, tensor, prob: float):
    # type: (Tensor, Tensor, float) -> Tensor
    return dropout_add(x, tensor, prob, False)


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method, output_layer_init_method,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        
        self.use_bias = args.add_bias_linear
        # parallel-layer arg
        self.use_parallel_residual = args.use_parallel_residual
        if self.use_parallel_residual:
            self.reduce = mpu.mappings.reduce_from_tensor_model_parallel_region

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        # Self attention.
        self.self_attention = ParallelAttention(
            init_method,
            output_layer_init_method,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type,
            parallel_output=self.use_parallel_residual)
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion
        self.dropout_fusion = args.dropout_fusion

        # Layernorm on the attention output
        # if parallel-layer is not applied
        if not self.use_parallel_residual:
            self.post_attention_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon)

        # NOTE: Below code block is not used.
        if self.layer_type == LayerType.decoder:
            self.inter_attention = ParallelAttention(
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon)

        # MLP
        self.mlp = ParallelMLP(init_method,
                               output_layer_init_method,
                               parallel_output=self.use_parallel_residual)

        # Alibi
        # if args.position_embedding_type == PositionEmbeddingType.alibi:
        #     self.alibi = self._build_alibi_tensor(args.seq_length, args.num_attention_heads, args.micro_batch_size).to(torch.cuda.current_device())
        #     if args.params_dtype == torch.float16:
        #         self.alibi = self.alibi.to(torch.float16)
        #     elif args.params_dtype == torch.bfloat16:
        #         self.alibi = self.alibi.to(torch.bfloat16)
        # else:
        #     self.alibi = None

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                layer_past=None, get_key_value=False):
        # jit scripting for a nn.module (with dropout) is not
        # trigerring the fusion kernel. For now, we use two
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.use_bias:
            if self.bias_dropout_fusion:
                if self.training:
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                bias_dropout_add_func = get_bias_dropout_add(self.training)
        else:
            if self.dropout_fusion:
                if self.training:
                    dropout_add_func = dropout_add_fused_train
                else:
                    dropout_add_func = dropout_add_fused_inference
            else:
                dropout_add_func = get_dropout_add(self.training)

        # hidden_states: [b, s, h]
        # apply PaLM (gpt-j) style parallel-layers
        # ref: https://github.com/EleutherAI/gpt-neox/blob/c883e8c6a2ff6a60b07f0f8006ce0208f41317f3/megatron/model/transformer.py#L804-L847
        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln(x)) + mlp(ln(x))
            # this means we can avoid doing the allreduce in the attn / mlp outputs
            # to save communication time (we can do a single allreduce after we add mlp / attn outputs).
            # due to a bug, the two layernorms are not tied in GPT-NeoX-20B. This is non-desirable, but
            # we preserve the functionality for backwards compatibility
            residual = hidden_states
            # tie input- and post-layernorm
            x = self.input_layernorm(hidden_states)
            x1, x2 = x, x

            # Self attention.
            attention_output, attention_bias = \
                self.self_attention(x1,
                                    attention_mask,
                                    layer_past=layer_past,
                                    get_key_value=get_key_value)

            if get_key_value:
                attention_output, _ = attention_output

            # MLP.
            mlp_output, mlp_bias = self.mlp(x2)

            # output = (x + attn(ln(x)) + mlp(ln(x)))
            output = self.reduce(attention_output + mlp_output)

            with torch.enable_grad():
                if attention_bias is not None and mlp_bias is not None:
                    bias = (attention_bias + mlp_bias).expand_as(output)
                    output = bias_dropout_add_func(
                        output,
                        bias,
                        residual,
                        self.hidden_dropout
                    )
                elif attention_bias is None and mlp_bias is None:
                    output = dropout_add_func(
                        output,
                        residual,
                        self.hidden_dropout
                    )
                else:
                    raise NotImplementedError("attention_bias and mlp_bias must both be None or not None")

        # parallel-layers not applied
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))

            # Layer norm at the beginning of the transformer layer.
            layernorm_output = self.input_layernorm(hidden_states)
            # Self attention.
            # attention_output, attention_bias = \
            #     self.self_attention(layernorm_output,
            #                         attention_mask,
            #                         layer_past=layer_past,
            #                         get_key_value=get_key_value,
            #                         alibi=self.alibi)
            attention_output, attention_bias = \
                self.self_attention(layernorm_output,
                                    attention_mask,
                                    layer_past=layer_past,
                                    get_key_value=get_key_value)

            if get_key_value:
                attention_output, presents = attention_output

            # Residual connection.
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = hidden_states

            # re-enable torch grad to enable fused optimization.
            with torch.enable_grad():
                if attention_bias is not None:
                    layernorm_input = bias_dropout_add_func(
                        attention_output,
                        attention_bias.expand_as(residual),
                        residual,
                        self.hidden_dropout)
                else:
                    layernorm_input = dropout_add_func(
                        attention_output,
                        residual,
                        self.hidden_dropout)

            # Layer norm post the self attention.
            layernorm_output = self.post_attention_layernorm(layernorm_input)

            # NOTE: Below code block is not used.
            if self.layer_type == LayerType.decoder:
                attention_output, attention_bias = \
                    self.inter_attention(layernorm_output,
                                        enc_dec_attn_mask,
                                        encoder_output=encoder_output)
                # residual connection
                if self.apply_residual_connection_post_layernorm:
                    residual = layernorm_output
                else:
                    residual = layernorm_input

                # re-enable torch grad to enable fused optimization.
                with torch.enable_grad():
                    if attention_bias is not None:
                        layernorm_input = bias_dropout_add_func(
                            attention_output,
                            attention_bias.expand_as(residual),
                            residual,
                            self.hidden_dropout)
                    else:
                        layernorm_input = dropout_add_func(
                            attention_output,
                            residual,
                            self.hidden_dropout)

                # Layer norm post the decoder attention
                layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

            # MLP.
            mlp_output, mlp_bias = self.mlp(layernorm_output)

            # Second residual connection.
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = layernorm_input

            # re-enable torch grad to enable fused optimization.
            with torch.enable_grad():
                if mlp_bias is not None:
                    output = bias_dropout_add_func(
                        mlp_output,
                        mlp_bias.expand_as(residual),
                        residual,
                        self.hidden_dropout)
                else:
                    output = dropout_add_func(
                        mlp_output,
                        residual,
                        self.hidden_dropout
                    )

            if get_key_value:
                output = [output, presents]

        return output

    # @staticmethod
    # def _build_alibi_tensor(max_seq_len, num_attention_heads, batch_size):
    #     # Based on https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    #     """Returns tensor shaped (batch_size * num_attention_heads, 1, max_seq_len)"""

    #     def get_slopes(n):
    #         def get_slopes_power_of_2(n):
    #             start = (2 ** (-2 ** -(math.log2(n) - 3)))
    #             ratio = start
    #             return [start * ratio ** i for i in range(n)]

    #         if math.log2(n).is_integer():
    #             return get_slopes_power_of_2(n)
    #         else:
    #             closest_power_of_2 = 2 ** math.floor(math.log2(n))
    #             return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
    #                                                                :n - closest_power_of_2]

    #     slopes = torch.Tensor(get_slopes(num_attention_heads))
    #     alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(
    #         num_attention_heads, -1, -1)
        
    #     #Select the part of the tensor that corresponds to our tensor parallel index.
    #     tp_world_size = mpu.get_tensor_model_parallel_world_size()
    #     tp_index = mpu.get_tensor_model_parallel_rank()
    #     alibi = alibi.reshape((tp_world_size, -1, *alibi.shape[1:]))[tp_index]
        
    #     alibi = alibi.repeat(batch_size, 1, 1)
    #     return alibi

class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline.

    Forward has two usages that affect attention mask communication:

    1) forward((input, attn_mask) , **kwargs) -> (output, mask)
       When the attention mask is provided as the second positional
       argument, typical pipeline behavior is used and both the output
       *and* mask are returned in a tuple. This tuple is then forwarded
       to the next stage in the pipeline.

       This version is useful if masks are dynamic.
    
    2) forward(input, **kwargs) -> output
       When the mask is static over all samples, it is advantageous to
       cache the mask and avoid communicating it.
    """
    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if torch.is_tensor(inputs) or len(inputs) == 1:
            hidden_states, attention_mask = inputs, None
            return super().forward(hidden_states, attention_mask, **kwargs)
        elif len(inputs) == 2:
            # Attention mask is an activation.
            hidden_states, attention_mask = inputs[0], inputs[1]
            return super().forward(*inputs, **kwargs), attention_mask
        else:
            raise RuntimeError('Received more inputs than understood.')


class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, init_method, output_layer_init_method,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 pre_process=True, post_process=True):
        super(ParallelTransformer, self).__init__()
        args = get_args()

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None

        # Store activation checkpoiting flag.
        self.checkpoint_activations = args.checkpoint_activations
        self.checkpoint_num_layers = args.checkpoint_num_layers

        # Number of layers.
        assert args.num_layers % mpu.get_pipeline_model_parallel_world_size() == 0, \
            'num_layers must be divisible by pipeline_model_parallel_size'
        self.num_layers = args.num_layers // mpu.get_pipeline_model_parallel_world_size()

        # Transformer layers.
        def build_layer(layer_number):
            return ParallelTransformerLayer(
                init_method,
                output_layer_init_method,
                layer_number,
                layer_type=layer_type,
                self_attn_mask_type=self_attn_mask_type)
        if args.virtual_pipeline_model_parallel_size is not None:
            assert args.num_layers % args.virtual_pipeline_model_parallel_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // args.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
                args.num_layers // args.virtual_pipeline_model_parallel_size) + \
                (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers

        self.layers = torch.nn.ModuleList(
            [build_layer(i + 1 + offset) for i in range(self.num_layers)])

        if self.post_process:
            # Final layer norm before output.
            self.final_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask,
                              encoder_output, enc_dec_attn_mask):
        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                attention_mask = inputs[1]
                encoder_output = inputs[2]
                enc_dec_attn_mask = inputs[3]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, attention_mask, encoder_output, enc_dec_attn_mask)
                return x_
            return custom_forward

        # Make sure memory is freed.
        mpu.reset_checkpointed_activations_memory_buffer()
        l = 0
        while l < self.num_layers:
            hidden_states = mpu.checkpoint(
                custom(l, l + self.checkpoint_num_layers),
                hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
            l += self.checkpoint_num_layers

        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False, encoder_output=None, enc_dec_attn_mask=None):

        # Checks.
        if layer_past is not None:
            assert get_key_value, \
                'for not None values in layer_past, ' \
                'expected get_key_value to be set'
        if get_key_value:
            assert not self.checkpoint_activations, \
                'get_key_value does not work with ' \
                'activation checkpointing'

        if self.pre_process:
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            # If the input flag for fp32 residual connection is set, convert for float.
            if self.fp32_residual_connection:
                hidden_states = hidden_states.transpose(0, 1).contiguous().float()
            # Otherwise, leave it as is.
            else:
                hidden_states = hidden_states.transpose(0, 1).contiguous()
        else:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        if encoder_output is not None:
             encoder_output = encoder_output.transpose(0, 1).contiguous()

        if self.checkpoint_activations:
            hidden_states = self._checkpointed_forward(hidden_states,
                                                       attention_mask,
                                                       encoder_output,
                                                       enc_dec_attn_mask)
        else:
            if get_key_value:
                presents = []
            for index in range(self.num_layers):
                layer = self._get_layer(index)
                past = None
                if layer_past is not None:
                    past = layer_past[index]
                hidden_states = layer(hidden_states,
                                      attention_mask,
                                      encoder_output=encoder_output,
                                      enc_dec_attn_mask=enc_dec_attn_mask,
                                      layer_past=past,
                                      get_key_value=get_key_value)
                if get_key_value:
                    hidden_states, present = hidden_states
                    presents.append(present)

        # Final layer norm.
        if self.post_process:
            # Reverting data format change [s b h] --> [b s h].
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            output = self.final_layernorm(hidden_states)
        else:
            output = hidden_states
        if get_key_value:
            output = [output, presents]

        return output
