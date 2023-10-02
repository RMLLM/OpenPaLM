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

import torch
import torch.nn.functional as F

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

###### BIAS SILU FUSION/ NO AUTOGRAD ################
# actual silu is:
# x * F.sigmoid(x)

@torch.jit.script
def bias_silu(bias, y):
    x = bias + y
    return  x * F.sigmoid(x)

# gradient of actual silu is:
# F.sigmoid(x) + silu(x) * (1 - F.sigmoid(x))
@torch.jit.script
def bias_silu_back(g, bias, y):
    x = bias + y
    silu_out = x * F.sigmoid(x)
    sigmoid_out = F.sigmoid(x)
    return sigmoid_out + silu_out * (1 - sigmoid_out)
    

class SiLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_silu(bias, input)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_silu_back(grad_output, bias, input)
        return tmp, tmp

bias_silu_impl = SiLUFunction.apply
