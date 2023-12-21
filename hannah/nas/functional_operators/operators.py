#
# Copyright (c) 2023 Hannah contributors.
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

import torch
import torch.fx as fx
import torch.nn.functional as F

from hannah.models.functional_net_test.expressions import padding_expression
from hannah.nas.core.parametrized import is_parametrized

# from hannah.nas.functional_operators.fx_wrapped_functions import conv2d_forward, linear_forward
from hannah.nas.functional_operators.lazy import lazy
from hannah.nas.functional_operators.op import Choice, Op, Tensor
from hannah.nas.functional_operators.shapes import (
    adaptive_average_pooling2d_shape,
    conv_shape,
    identity_shape,
    linear_shape,
)
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter
from hannah.nas.parameters.parametrize import parametrize


@torch.fx.wrap
def conv2d(input, weight, stride, padding, dilation, groups, *, id):
    return F.conv2d(
        input=input,
        weight=weight,
        stride=lazy(stride),
        padding=lazy(padding),
        dilation=lazy(dilation),
        groups=lazy(groups),
    )


@torch.fx.wrap
def linear(input, weight, *, id):
    return F.linear(input=input, weight=weight.T)


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
def adaptive_avg_pooling(input, output_size=(1, 1), *, id):
    return F.adaptive_avg_pool2d(input, output_size=output_size)


@torch.fx.wrap
def max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode, *, id):
    return F.max_pool2d(
        input=input,
        kernel_size=lazy(kernel_size),
        stride=lazy(stride),
        padding=lazy(padding),
        dilation=lazy(dilation),
        ceil_mode=lazy(ceil_mode),
    )


@parametrize
class Conv1d(Op):
    def __init__(self, out_channels, kernel_size=1, stride=1, dilation=1) -> None:
        super().__init__(name="Conv1d")
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding_expression(self.kernel_size, self.stride, self.dilation)

    # def _verify_operands(self, operands):
    #     assert len(operands) == 2

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

    # FIXME: Use wrapped implementation
    def _forward_implementation(self, *operands):
        x = operands[0]  # .forward()
        weight = operands[1]  # .forward()
        return F.conv1d(
            input=x,
            weight=weight,
            stride=lazy(self.stride),
            padding=lazy(self.padding),
            dilation=lazy(self.dilation),
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
class Conv2d(Op):
    def __init__(self, stride=1, dilation=1, groups=1, padding=None) -> None:
        super().__init__(name="Conv2d", stride=stride, dilation=dilation, groups=groups)
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding

    # def _verify_operands(self, operands):
    #     assert len(operands) == 2

    def __call__(self, *operands) -> Any:
        new_conv = super().__call__(*operands)
        input_shape = operands[0].shape()
        weight_shape = operands[1].shape()
        operands[1].id = f"{new_conv.id}.{operands[1].id}"

        new_conv.in_channels = input_shape[1]
        new_conv.out_channels = weight_shape[0]
        new_conv.kernel_size = weight_shape[2]
        if self.padding is None:
            new_conv.padding = padding_expression(
                new_conv.kernel_size, new_conv.stride, new_conv.dilation
            )

        # new_conv.weight = Tensor(name=self.id + '.weight',
        #                          shape=(self.out_channels, new_conv.in_channels, self.kernel_size, self.kernel_size),
        #                          axis=('O', 'I', 'kH', 'kW'))

        # new_conv.operands.append(new_conv.weight)
        # if is_parametrized(new_conv.weight):
        #     new_conv._PARAMETERS[new_conv.weight.name] = new_conv.weight
        # new_conv._verify_operands(new_conv.operands)
        return new_conv

    def _forward_implementation(self, x, weight):
        return conv2d(
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
        return new_linear

    def shape_fun(self):
        return linear_shape(*self.operands)

    def _forward_implementation(self, input, weight):
        input = torch.flatten(input, start_dim=1)
        return linear(input, weight, id=self.id)


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
class Quantize(Op):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(name="Quantize")

    def __call__(self, *operands) -> Any:
        op = super().__call__(*operands)
        return op

    def shape_fun(self):
        return self.operands[0].shape()

    def _forward_implementation(self, *operands):
        return operands[0]  # TODO: Implement real quantization functionality


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
class AdaptiveAvgPooling(Op):
    def __init__(self, output_size=(1, 1)) -> None:
        super().__init__(name="AvgPooling")
        self.output_size = output_size

    def shape_fun(self):
        return adaptive_average_pooling2d_shape(
            *self.operands, output_size=self.output_size
        )

    def _forward_implementation(self, *operands):
        return adaptive_avg_pooling(
            operands[0], output_size=self.output_size, id=self.id
        )


@parametrize
class MaxPool2d(Op):
    def __init__(
        self, kernel_size=3, stride=1, padding=0, dilation=1, ceil_mode=False
    ) -> None:
        super().__init__(name="MaxPool2d")
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def shape_fun(self):
        return conv_shape(
            *self.operands,
            dims=2,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    def _forward_implementation(self, *operands):
        return max_pool2d(
            input=operands[0],
            kernel_size=lazy(self.kernel_size),
            stride=lazy(self.stride),
            padding=lazy(self.padding),
            dilation=lazy(self.dilation),
        )
