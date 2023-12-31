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
# import torch.nn.functional as F

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

###### BIAS SILU FUSION/ NO AUTOGRAD ################
# actual silu is:
# x * F.sigmoid(x)

@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)

# gradient of actual silu is:
# F.sigmoid(x) + silu(x) * (1 - F.sigmoid(x))
@torch.jit.script
def silu_back(g, x):
    silu_out = x * torch.sigmoid(x)
    sigmoid_out = torch.sigmoid(x)
    ff = sigmoid_out + silu_out * (1 - sigmoid_out)
    return g * ff

class SiLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return silu(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        tmp = silu_back(grad_output, input)
        return tmp
    
silu_impl = SiLUFunction.apply
