# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

def _apply_op(inpt, dx_val, channel, dim, order, padding, mode="dx"):
    "Core logic for numerical derivatives"
    var = inpt[:, channel : channel + 1, ...]
    spatial_dims = var.dim() - 2
    
    # get filter
    if mode == "dx":
        if order == 1:
            ddx1D = torch.Tensor([-0.5, 0.0, 0.5]).to(inpt.device)
        elif order == 3:
            ddx1D = torch.Tensor([-1.0/60.0, 3.0/20.0, -3.0/4.0, 0.0, 3.0/4.0, -3.0/20.0, 1.0/60.0]).to(inpt.device)
    else: # ddx (2nd order)
        if order == 1:
            ddx1D = torch.Tensor([1.0, -2.0, 1.0]).to(inpt.device)
        elif order == 3:
            ddx1D = torch.Tensor([1.0/90.0, -3.0/20.0, 3.0/2.0, -49.0/18.0, 3.0/2.0, -3.0/20.0, 1.0/90.0]).to(inpt.device)
    
    # Reshape filter
    shape = [1, 1] + [1] * spatial_dims
    shape[2 + dim] = -1
    kernel = torch.reshape(ddx1D, shape)

    # Pad all spatial dims
    pad_len = (ddx1D.shape[0] - 1) // 2
    padding_list = [pad_len] * (2 * spatial_dims)
    
    pad_mode = "replicate" if padding == "replication" else "constant"
    var = F.pad(var, padding_list, mode=pad_mode, value=0 if pad_mode == "constant" else None)

    # Convolve
    if spatial_dims == 2:
        output = F.conv2d(var, kernel, padding="valid")
    elif spatial_dims == 3:
        output = F.conv3d(var, kernel, padding="valid")
    else:
        raise NotImplementedError(f"Numerical derivative not implemented for {spatial_dims} spatial dims")

    # Crop dimensions that were NOT differentiated but were padded
    # After conv(valid) with size K on DIM, the size is (L + 2*P) - K + 1 = L (if K=2P+1)
    # The other dimensions are L + 2P - 1 + 1 = L + 2P. We need to crop P from each side.
    slices = [slice(None)] * (spatial_dims + 2)
    for i in range(spatial_dims):
        if i != dim:
            slices[2 + i] = slice(pad_len, -pad_len)
    
    output = output[slices]
    
    scale = (1.0 / dx_val) if mode == "dx" else (1.0 / dx_val**2)
    return scale * output

def dx(inpt, dx, channel, dim, order=1, padding="zeros"):
    return _apply_op(inpt, dx, channel, dim, order, padding, mode="dx")

def ddx(inpt, dx, channel, dim, order=1, padding="zeros"):
    return _apply_op(inpt, dx, channel, dim, order, padding, mode="ddx")
