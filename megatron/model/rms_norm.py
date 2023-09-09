# This was copied from https://github.com/EleutherAI/gpt-neox/blob/f6c0b7762216f52daf7572de3f3c50802d729ea8/megatron/model/norms.py#L34
# and modified by seopbo.

import torch
import numbers

from megatron import get_args
from megatron import mpu


class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-8):
        """
        Root Mean Square Layer Normalization

        :param normalized_shape: model size
        :param eps:  epsilon value, default 1e-8
        """
        super(RMSNorm, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)

        args = get_args()
        self.eps = eps
        self.d = normalized_shape[-1]
        self.norm_tp_auto_sync = args.sync_tp_duplicated_parameters

        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.register_parameter("weight", self.weight)

        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
        self.register_parameter("bias", self.bias)

    def forward(self, x):
        if self.norm_tp_auto_sync:
            torch.distributed.all_reduce(self.weight, op=torch.distributed.ReduceOp.AVG, group=mpu.get_tensor_model_parallel_group())
            if self.use_bias is not None:
                torch.distributed.all_reduce(self.bias, op=torch.distributed.ReduceOp.AVG, group=mpu.get_tensor_model_parallel_group())

        # fp32로 학습하는 경우
        # fp16으로 학습하는 경우 (mixed_precision_training)
        # weight나 bias의 자료형이 input의 자료형보다 큰 경우는 존재하지않음.
        input_dtype = x.dtype

        if input_dtype == self.weight.dtype:
            # fp32로 학습하는 경우
            # fp16으로 학습하는 데, args.fp32_residual_connection=False인 경우
            cast_weight = self.weight
            cast_bias = self.bias

            # fp32으로 학습하는 경우에는 그냥 연산, fp32로 결과를 출력한다.
            # fp16으로 학습하는 경우에는 mixed_precision_training에 따라 layernorm 연산을 fp32로 전환해서하고, 결과를 fp16으로 출력한다..
            return self.rmsnorm(x, self.d, self.eps, cast_weight, cast_bias)
        else:
            # fp16으로 학습하는 데, args.fp32_residual_connection=True인 경우로 input이 fp32로 들어옴.
            # input이 fp32이고 weight와 bias가 fp16이므로 F.layer_norm 실행 시 type mismatch error 발생한다.
            # type mismatch error를 방지하기위해 input의 dtype인 fp32로 weight와 bias를 casting한다.
            cast_weight = self.weight.to(input_dtype)
            cast_bias = self.bias.to(input_dtype)

            # 아래의 함수에서 fp32, fp32로 들어간 상태이므로 원래 우리가 필요한 fp16으로 나오지않는다.
            # 따라서 강제로 fp16으로 다시 casting한다.
            return self.rmsnorm(x, self.d, self.eps, cast_weight, cast_bias).to(self.weight.dtype)

    @staticmethod
    def rmsnorm(x, d, eps, weight, bias):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        d_x = d

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + eps)

        return weight * x_normed + bias
