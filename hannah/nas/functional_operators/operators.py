#
# Copyright (c) 2024 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
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
#
import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.fx as fx
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import FakeQuantize

from hannah.nas.core.parametrized import is_parametrized
from hannah.nas.expressions.utils import prod
from hannah.nas.functional_operators.lazy import lazy
from hannah.nas.functional_operators.op import Choice, Op, Tensor
from hannah.nas.functional_operators.shapes import (
    adaptive_average_pooling_shape,
    conv_shape,
    identity_shape,
    linear_shape,
    padding_expression,
    pool_shape,
)
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter
from hannah.nas.parameters.parametrize import parametrize


@torch.fx.wrap
def conv1d(input, weight, stride, padding, dilation, groups, *, id):
    return F.conv1d(
        input=input,
        weight=weight,
        stride=lazy(stride),
        padding=lazy(padding),
        dilation=lazy(dilation),
        groups=lazy(groups),
    )


@torch.fx.wrap
def conv2d(input, weight, bias, stride=1, padding=1, dilation=1, groups=1, *, id):
    return F.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=lazy(stride),
        padding=lazy(padding),
        dilation=lazy(dilation),
        groups=lazy(groups),
    )


@torch.fx.wrap
def linear(input, weight, bias, *, id):
    return F.linear(input=input, weight=weight.T, bias=bias)


@torch.fx.wrap
def batch_norm(input, running_mu, running_std, *, id, training, track_running_stats):
    if not training or track_running_stats:
        running_mu = running_mu.to(input.device)
        running_std = running_std.to(input.device)
    else:
        running_mu = None
        running_std = None
    res = F.batch_norm(input, running_mu, running_std, training=training)
    return res


@torch.fx.wrap
def relu(input, *, id):
    return F.relu(input)


@torch.fx.wrap
def add(input, other, *, id):
    return torch.add(input, other)


@torch.fx.wrap
def adaptive_avg_pooling2d(input, output_size=(1, 1), *, id):
    return F.adaptive_avg_pool2d(input, output_size=output_size)


@torch.fx.wrap
def adaptive_avg_pooling1d(input, output_size=(1, 1), *, id):
    return F.adaptive_avg_pool1d(input, output_size=output_size)


@torch.fx.wrap
def max_pool(input, kernel_size, stride, padding, dilation, *, id):
    return F.max_pool2d(input, kernel_size, stride, padding, dilation)


@torch.fx.wrap
def avg_pool(input, kernel_size, stride, padding, *, id):
    return F.avg_pool2d(input, kernel_size, stride, padding)


@torch.fx.wrap
def max_avg_pool(input, kernel_size, stride, padding):
    avg_out = F.avg_pool2d(input, kernel_size, stride, padding)
    max_out = F.max_pool2d(input, kernel_size, stride, padding, dilation=1)
    return 0.5 * avg_out + 0.5 * max_out


@torch.fx.wrap
def interleave(input, step_size):
    # Assuming NCHW layout!! Maybe change later to use named axis of Tensor?
    return torch.concat(
        [input[:, shift_pos::step_size, :, :] for shift_pos in range(step_size)], dim=1
    )


@torch.fx.wrap
def dropout(input, p, *, id):
    return F.dropout(input, p)


@torch.fx.wrap
def self_attention2d(q, k, v, num_heads, d_model, *, id):
    """
    Arguments:
        q: Tensor, shape ``[B, h*d, H, W]``
        k: Tensor, shape ``[B, h*d, H, W]``
        v: Tensor, shape ``[B, h*d, H, W]``
    """
    scale = d_model ** -0.5
    b, _, h, w = q.shape
    q = q.view(b, num_heads, d_model, h * w)
    k = k.view(b, num_heads, d_model, h * w)
    v = v.view(b, num_heads, d_model, h * w)
    # [B, h, d, H*W]

    q *= scale
    attn = q.transpose(-2, -1) @ k  # [B, h, H*W, H*W]
    attn = F.softmax(attn, dim=-1)
    score = v @ attn  # [B, h, d, H*W]
    out = score.reshape(b, -1, h, w)  # [B, h*d, H, W]

    return out


@torch.fx.wrap
def relu_linear_attention(q, k, v, num_heads, d_model, *, id):
    """
    Adapted from EfficientViT.
    Arguments:
        q: Tensor, shape ``[B, h*d, H, W]``
        k: Tensor, shape ``[B, h*d, H, W]``
        v: Tensor, shape ``[B, h*d, H, W]``
    """
    b, _, h, w = q.shape
    q = q.view(b, num_heads, d_model, h * w)
    k = k.view(b, num_heads, d_model, h * w)
    v = v.view(b, num_heads, d_model, h * w)
    # [B, h, d, H*W]

    # lightweight linear attention
    q = F.relu(q, inplace=False)
    k = F.relu(k, inplace=False)

    # linear matmul
    v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
    # [B, h, d+1, H*W]
    kv = torch.matmul(v, k.transpose(-1, -2))
    # [B, h, d+1, d]
    out = torch.matmul(kv, q)
    # [B, h, d+1, H*W]
    out = out[:, :, :-1] / (out[:, :, -1:] + 1.0e-15)
    # [B, h, d, H*W]

    out = out.reshape(b, -1, h, w)
    # [B, h*d, H, W]

    return out


@torch.fx.wrap
def reshape(input, new_shape):
    return input.view(new_shape)


@torch.fx.wrap
def permute(input, dims):
    return input.permute(dims)


@parametrize
class Conv1d(Op):
    def __init__(self, kernel_size=1, stride=1, dilation=1, groups=1) -> None:
        super().__init__(name="Conv1d")
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding_expression(self.kernel_size, self.stride, self.dilation)

    def __call__(self, *operands) -> Any:
        new_conv = super().__call__(*operands)
        input_shape = operands[0].shape()
        weight_shape = operands[1].shape()
        operands[1].id = f"{new_conv.id}.{operands[1].id}"

        new_conv.in_channels = input_shape[1]
        new_conv.out_channels = weight_shape[0]
        new_conv.kernel_size = weight_shape[2]
        new_conv.padding = padding_expression(
            new_conv.kernel_size, new_conv.stride, new_conv.dilation
        )
        return new_conv

    def _forward_implementation(self, *operands):
        x = operands[0]
        weight = operands[1]
        return conv1d(
            x,
            weight,
            stride=lazy(self.stride),
            padding=lazy(self.padding),
            dilation=lazy(self.dilation),
            groups=lazy(self.groups),
            id=self.id,
        )

    def shape_fun(self):
        return conv_shape(
            *self.operands,
            dims=1,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


@parametrize
class Conv2d(Op):
    def __init__(self, stride=1, dilation=1, groups=1, padding=None) -> None:
        super().__init__(
            name="Conv2d", stride=stride, dilation=dilation, groups=groups,
        )
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding

    def __call__(self, *operands) -> Any:
        new_conv = super().__call__(*operands)
        input_shape = operands[0].shape()
        weight_shape = operands[1].shape()
        operands[1].id = f"{new_conv.id}.{operands[1].id}"
        if len(operands) >= 3:
            operands[2].id = f"{new_conv.id}.{operands[2].id}"

        new_conv.in_channels = input_shape[1]
        new_conv.out_channels = weight_shape[0]
        new_conv.kernel_size = weight_shape[2]
        if self.padding is None:
            new_conv.padding = padding_expression(
                new_conv.kernel_size, new_conv.stride, new_conv.dilation
            )

        return new_conv

    def _forward_implementation(self, input, weight, bias=None):
        return conv2d(
            input,
            weight,
            bias,
            stride=lazy(self.stride),
            padding=lazy(self.padding),
            dilation=lazy(self.dilation),
            groups=lazy(self.groups),
            id=self.id,
        )

    def shape_fun(self):
        return conv_shape(
            *self.operands,
            dims=2,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


@parametrize
class Linear(Op):
    def __init__(self) -> None:
        super().__init__(name="Linear")

    def __call__(self, *operands) -> Any:
        new_linear = super().__call__(*operands)
        new_linear.in_features = operands[1].shape()[0]
        new_linear.out_features = operands[1].shape()[1]
        operands[1].id = f"{new_linear.id}.{operands[1].id}"
        if len(operands) >= 3:
            operands[2].id = f"{new_linear.id}.{operands[2].id}"
        return new_linear

    def shape_fun(self):
        return linear_shape(*self.operands)

    def _forward_implementation(self, input, weight, bias=None):
        input = torch.flatten(input, start_dim=1)
        return linear(
            input, weight, bias,
            id=self.id
        )


@parametrize
class Relu(Op):
    def __init__(self) -> None:
        super().__init__(name="Relu")

    # def _verify_operands(self, operands):
    #     assert len(operands) == 1

    def _forward_implementation(self, *operands):
        return relu(operands[0], id=self.id)

    def shape_fun(self):
        return identity_shape(*self.operands)


@parametrize
class Add(Op):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(name="Add")

    # def _verify_operands(self, operands):
    #     assert len(operands) == 2

    def __call__(self, *operands) -> Any:
        op = super().__call__(*operands)
        # self._verify_operands(op.operands)
        return op

    def _forward_implementation(self, input, other):
        return add(input, other, id=self.id)

    def shape_fun(self):
        return identity_shape(*self.operands)


# TODO: Add a version where the OP vanishes after it is connected and just leaves an edge between operands and successor
@parametrize
class Identity(Op):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(name="Identity")

    # def _verify_operands(self, *operands):
    #     assert len(operands) == 1

    def __call__(self, *operands) -> Any:
        op = super().__call__(*operands)
        # self._verify_operands(op.operands)
        return op

    def shape_fun(self):
        return self.operands[0].shape()

    def _forward_implementation(self, *operands):
        return operands[0]


@parametrize
class Requantize(Op):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(name="Quantize")
        self.quantize = FakeQuantize()

    @property
    def dtype(self):
        return self.quantize.dtype  # FIXME: we might wan't to switch to our own dtypes

    @property
    def scale(self) -> np.ndarray:
        return self.quantize.scale.numpy()

    @property
    def zero_point(self) -> np.ndarray:
        return self.quantize.zero_point.numpy()

    @property
    def ch_axis(self):
        return self.quantize.ch_axis

    def __call__(self, *operands) -> Any:
        op = super().__call__(*operands)
        return op

    def shape_fun(self):
        return self.operands[0].shape()

    def _forward_implementation(self, *operands):
        return self.quantize(operands[0])


@parametrize
class BatchNorm(Op):
    def __init__(self, track_running_stats=True) -> None:
        super().__init__(name="BatchNorm")
        self.track_running_stats = track_running_stats

    def __call__(self, *operands) -> Any:
        return super().__call__(*operands)

    def shape_fun(self):
        return self.operands[0].shape()

    def _forward_implementation(self, *operands):
        return batch_norm(
            operands[0],
            operands[1],
            operands[2],
            id=self.id,
            training=self._train,
            track_running_stats=self.track_running_stats,
        )


@parametrize
class MaxPooling(Op):
    def __init__(self, kernel_size, stride, dilation=1, padding=None) -> None:
        super().__init__(name="MaxPooling")
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        if padding is None:
            self.padding = padding_expression(
                self.kernel_size, self.stride, self.dilation
            )
        else:
            self.padding = padding

    def shape_fun(self):
        return pool_shape(
            *self.operands,
            dims=2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    def _forward_implementation(self, *operands):
        return max_pool(
            operands[0],
            kernel_size=lazy(self.kernel_size),
            stride=lazy(self.stride),
            padding=lazy(self.padding),
            dilation=lazy(self.dilation),
            id=self.id
        )


@parametrize
class AvgPooling(Op):
    def __init__(self, kernel_size, stride, dilation=1) -> None:
        super().__init__(name="AvgPooling")
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding_expression(self.kernel_size, self.stride, self.dilation)

    def shape_fun(self):
        return pool_shape(
            *self.operands,
            dims=2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    def _forward_implementation(self, *operands):
        return avg_pool(
            operands[0],
            kernel_size=lazy(self.kernel_size),
            stride=lazy(self.stride),
            padding=lazy(self.padding),
            id=self.id
        )


@parametrize
class MaxAvgPooling(Op):
    def __init__(self, kernel_size, stride, dilation=1) -> None:
        super().__init__(name="MaxAvgPooling")
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding_expression(self.kernel_size, self.stride, self.dilation)

    def shape_fun(self):
        return pool_shape(
            *self.operands,
            dims=2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    def _forward_implementation(self, *operands):
        return max_avg_pool(
            operands[0],
            kernel_size=lazy(self.kernel_size),
            stride=lazy(self.stride),
            padding=lazy(self.padding),
            id=self.id
        )


@parametrize
class AdaptiveAvgPooling(Op):
    def __init__(self, output_size=(1, 1)) -> None:
        super().__init__(name="AdaptiveAvgPooling")
        self.output_size = output_size
        if isinstance(output_size, int):
            self.dim = 1
        else:
            self.dim = len(output_size)

    def shape_fun(self):
        return adaptive_average_pooling_shape(
            *self.operands, output_size=self.output_size
        )

    def _forward_implementation(self, *operands):
        if self.dim == 1:
            return adaptive_avg_pooling1d(
                operands[0], output_size=self.output_size, id=self.id
            )
        else:
            return adaptive_avg_pooling2d(
                operands[0], output_size=self.output_size, id=self.id
            )


@parametrize
class Dropout(Op):
    def __init__(self, p) -> None:
        super().__init__(name="Dropout", p=p)
        self.p = p

    def __call__(self, *operands) -> Any:
        op = super().__call__(*operands)
        return op

    def _forward_implementation(self, input):
        return dropout(input, p=self.p, id=self.id)

    def shape_fun(self):
        return identity_shape(*self.operands)


@parametrize
class InterleaveChannels(Op):
    def __init__(self, step_size) -> None:
        super().__init__(name="InterleaveChannels")
        self.step_size = step_size

    def shape_fun(self):
        return self.operands[0].shape()

    def _forward_implementation(self, *operands):
        return interleave(operands[0], step_size=lazy(self.step_size))


@parametrize
class SelfAttention2d(Op):
    def __init__(self, num_heads, d_model) -> None:
        super().__init__(name="SelfAttention2d", num_heads=num_heads, d_model=d_model)
        self.num_heads = num_heads
        self.d_model = d_model

    # def _verify_operands(self, operands):
    #     assert len(operands) == 3

    def __call__(self, *operands) -> Any:
        new_attn = super().__call__(*operands)
        # self._verify_operands(new_attn.operands)
        return new_attn

    def _forward_implementation(self, *operands):
        q = operands[0]
        k = operands[1]
        v = operands[2]
        out = self_attention2d(
            q,
            k,
            v,
            num_heads=lazy(self.num_heads),
            d_model=lazy(self.d_model),
            id=self.id,
        )

        return out

    def shape_fun(self):
        input_shape = self.operands[0].shape()
        batch = input_shape[0]
        height = input_shape[-2]
        width = input_shape[-1]
        out_dim = self.num_heads * self.d_model

        return (batch, out_dim, height, width)


@parametrize
class ReluLinearAttention(Op):
    """
    Adapted from EfficientViT
    """

    def __init__(self, num_heads, d_model) -> None:
        super().__init__(
            name="ReluLinearAttention", num_heads=num_heads, d_model=d_model
        )
        self.num_heads = num_heads
        self.d_model = d_model

    # def _verify_operands(self, operands):
    #     assert len(operands) == 3

    def __call__(self, *operands) -> Any:
        new_attn = super().__call__(*operands)
        # self._verify_operands(new_attn.operands)
        return new_attn

    def _forward_implementation(self, *operands):
        q = operands[0]
        k = operands[1]
        v = operands[2]
        out = relu_linear_attention(
            q,
            k,
            v,
            num_heads=lazy(self.num_heads),
            d_model=lazy(self.d_model),
            id=self.id,
        )

        return out

    def shape_fun(self):
        input_shape = self.operands[0].shape()
        batch = input_shape[0]
        height = input_shape[-2]
        width = input_shape[-1]
        out_dim = self.num_heads * self.d_model

        return (batch, out_dim, height, width)


@parametrize
class Reshape(Op):
    def __init__(self, shape) -> None:
        super().__init__(name="Reshape")
        self.new_shape = shape

    def __call__(self, *operands) -> Any:
        new_reshape = super().__call__(*operands)
        return new_reshape

    def _forward_implementation(self, input):
        return reshape(input, self.new_shape)

    def shape_fun(self):
        if -1 in self.new_shape:
            input_shape = self.operands[0].shape()
            shape = list(self.new_shape)

            # Calculate the product of the input shape
            input_shape_prod = prod(input_shape)

            # Calculate the product of the shape dimensions excluding -1
            shape_prod = prod([dim for dim in shape if dim != -1])

            # Replace -1 with the calculated dimension
            shape[shape.index(-1)] = input_shape_prod // shape_prod

            return tuple(shape)

        return tuple(self.new_shape)


@parametrize
class Permute(Op):
    def __init__(self, dims) -> None:
        super().__init__(name="Permute")
        self.dims = dims

    def __call__(self, *operands) -> Any:
        new_permute = super().__call__(*operands)
        return new_permute

    def _forward_implementation(self, input):
        return permute(input, self.dims)

    def shape_fun(self):
        old_shape = self.operands[0].shape()
        new_shape = [old_shape[dim] for dim in self.dims]
        return new_shape
