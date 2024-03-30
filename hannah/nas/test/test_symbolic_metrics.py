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
import torch
import torch.nn as nn

from hannah.callbacks.summaries import walk_model
from hannah.nas.expressions.metrics import (
    conv2d_macs,
    conv2d_weights,
    linear_macs,
    linear_weights,
)
from hannah.nas.expressions.shapes import conv2d_shape, linear_shape
from hannah.nas.functional_operators.shapes import padding_expression
from hannah.nas.parameters.lazy import Lazy
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter
from hannah.nas.parameters.parametrize import parametrize

conv2d = Lazy(nn.Conv2d, shape_func=conv2d_shape)
linear = Lazy(nn.Linear, shape_func=linear_shape)


@parametrize
class Convolution(nn.Module):
    def __init__(self, inputs) -> None:
        super().__init__()
        self.id = "convolution"
        self.inputs = inputs
        input_shape = inputs[0]
        self.in_channels = input_shape[1]
        self.out_channels = IntScalarParameter(4, 64, 4)
        self.kernel_size = CategoricalParameter([1, 3, 5])
        self.stride = CategoricalParameter([1, 2])

        self.conv = conv2d(
            self.id + ".conv",
            inputs=inputs,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding_expression(self.kernel_size, self.stride),
        )

    def initialize(self):
        self.tconv = self.conv.instantiate()

    def forward(self, x):
        out = self.tconv(x)
        return out

    @property
    def shape(self):
        return self.conv.shape

    @property
    def macs(self):
        return conv2d_macs(self.inputs[0], self.shape, self.conv.kwargs)

    @property
    def weights(self):
        return conv2d_weights(self.inputs[0], self.shape, self.conv.kwargs)


@parametrize
class Linear(nn.Module):
    def __init__(self, input, labels) -> None:
        super().__init__()
        self.input = input
        self.labels = labels

        in_features = self.input[1] * self.input[2] * self.input[3]
        self._linear = linear(
            "linear",
            inputs=[self.input],
            in_features=in_features,
            out_features=self.labels,
        )

    def initialize(self):
        self.linear = self._linear.instantiate()

    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.linear(out)
        return out

    @property
    def shape(self):
        return self._linear.shape

    @property
    def macs(self):
        return linear_macs(self.input, self.shape, self._linear.kwargs)

    @property
    def weights(self):
        return linear_weights(self.input, self.shape, self._linear.kwargs)


def test_conv_metrics():
    input_tensor = torch.ones(1, 3, 32, 32)
    conv = Convolution([input_tensor.shape])
    conv.initialize()
    conv.sample()
    out = conv(input_tensor)
    summary = walk_model(conv, input_tensor)

    mac_summary = summary["MACs"].item()
    mac_symbolic = conv.macs.evaluate()

    assert mac_summary == mac_symbolic

    weight_summary = summary["Weights volume"].item()
    weights_symbolic = conv.weights.evaluate()

    assert weight_summary == weights_symbolic
    print()


def test_linear_metrics():
    input_tensor = torch.ones(1, 3, 32, 32)
    conv = Convolution([input_tensor.shape])
    fc = Linear(conv.shape, 10)
    conv.initialize()
    fc.initialize()
    conv.sample()
    fc.sample()
    conv_out = conv(input_tensor)
    out = fc(conv_out)

    summary = walk_model(fc, conv_out)

    mac_summary = summary["MACs"].item()
    mac_symbolic = fc.macs.evaluate()

    assert mac_summary == mac_symbolic

    weight_summary = summary["Weights volume"].item()
    weights_symbolic = fc.weights.evaluate()

    assert weight_summary == weights_symbolic
    print()


if __name__ == "__main__":
    test_conv_metrics()
    test_linear_metrics()
