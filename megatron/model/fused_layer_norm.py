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

"""This code is copied fron NVIDIA apex:
      https://github.com/NVIDIA/apex
   with some changes. """

import numbers


from megatron import get_args
from megatron import mpu
from packaging import version
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import importlib
import torch
import torch.nn.functional as F

global fused_mix_prec_layer_norm_cuda
fused_mix_prec_layer_norm_cuda = None


class FusedLayerNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.has_bias = True if bias is not None else False

        input_ = input.contiguous()
        weight_ = weight.contiguous()

        if ctx.has_bias:
            bias_ = bias.contiguous()
        else:
            bias_ = torch.zeros_like(weight).contiguous()

        output, mean, invvar = fused_mix_prec_layer_norm_cuda.forward_affine(
            input_, ctx.normalized_shape, weight_, bias_, ctx.eps)
        ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_, bias_, mean, invvar = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_input, grad_weight, grad_bias \
        = fused_mix_prec_layer_norm_cuda.backward_affine(
            grad_output.contiguous(), mean, invvar,
            input_, ctx.normalized_shape,
            weight_, bias_, ctx.eps)

        if not ctx.has_bias:
            grad_bias = None
        return grad_input, grad_weight, grad_bias, None, None


class MixedFusedLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(MixedFusedLayerNorm, self).__init__()

        global fused_mix_prec_layer_norm_cuda
        fused_mix_prec_layer_norm_cuda = importlib.import_module(
        "fused_mix_prec_layer_norm_cuda")

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        args = get_args()

        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.bias = Parameter(torch.Tensor(*normalized_shape)) if args.add_bias_linear else None

        self.reset_parameters()

        self.layernorm_tp_auto_sync = args.sync_tp_duplicated_parameters

        self.use_meg_ds_fused_layer_norm = (
            args.bf16 # Current Meg-DS cuda kernel has better throughput than torch.nn.LayerNorm
            or version.parse(torch.__version__) >= version.parse("1.11.0") # https://github.com/pytorch/pytorch/pull/66920
        )

    def reset_parameters(self):
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input):
        if self.layernorm_tp_auto_sync:
            torch.distributed.all_reduce(self.weight, op=torch.distributed.ReduceOp.AVG, group=mpu.get_tensor_model_parallel_group())
            if self.bias is not None:
                torch.distributed.all_reduce(self.bias, op=torch.distributed.ReduceOp.AVG, group=mpu.get_tensor_model_parallel_group())

        if self.use_meg_ds_fused_layer_norm:
            return FusedLayerNormAffineFunction.apply(input, self.weight, self.bias, self.normalized_shape, self.eps)
        else:
            # fp32로 학습하는 경우
            # fp16으로 학습하는 경우 (mixed_precision_training)
            # weight나 bias의 자료형이 input의 자료형보다 큰 경우는 존재하지않음.
            input_dtype = input.dtype

            if input_dtype == self.weight.dtype:
                # fp32로 학습하는 경우
                # fp16으로 학습하는 데, args.fp32_residual_connection=False인 경우
                cast_weight = self.weight
                cast_bias = self.bias

                # fp32으로 학습하는 경우에는 그냥 연산, fp32로 결과를 출력한다.
                # fp16으로 학습하는 경우에는 mixed_precision_training에 따라 layernorm 연산을 fp32로 전환해서하고, 결과를 fp16으로 출력한다..
                return F.layer_norm(input, self.normalized_shape, cast_weight, cast_bias, self.eps)
            else:
                # fp16으로 학습하는 데, args.fp32_residual_connection=True인 경우로 input이 fp32로 들어옴.
                # input이 fp32이고 weight와 bias가 fp16이므로 F.layer_norm 실행 시 type mismatch error 발생한다.
                # type mismatch error를 방지하기위해 input의 dtype인 fp32로 weight와 bias를 casting한다.
                cast_weight = self.weight.to(input_dtype)
                cast_bias = self.bias.to(input_dtype) if self.bias is not None else self.bias

                # 아래의 함수에서 fp32, fp32로 들어간 상태이므로 원래 우리가 필요한 fp16으로 나오지않는다.
                # 따라서 강제로 fp16으로 다시 casting한다.
                return F.layer_norm(input, self.normalized_shape, cast_weight, cast_bias, self.eps).to(self.weight.dtype)