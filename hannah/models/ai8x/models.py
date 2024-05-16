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
"""A search space based on the cifar 10 NASNet search space for ai85x devices from: htt"""

from hannah.nas.expressions.types import Int
from hannah.nas.functional_operators.op import ChoiceOp, Tensor, scope
from hannah.nas.functional_operators.operators import (
    BatchNorm,
    Conv2d,
    Identity,
    Linear,
    MaxPooling,
    Relu,
    Requantize,
)
from hannah.nas.parameters.parameters import (
    CategoricalParameter,
    FloatScalarParameter,
    IntScalarParameter,
)


def max_pool(input, kernel_size=2, stride=2, padding=2):
    return MaxPooling(kernel_size=kernel_size, stride=stride, padding=padding)(input)


def conv2d(input, weight, stride=1, dilation=1, groups=1, padding=None):
    return Conv2d(stride=stride, dilation=dilation, groups=groups, padding=padding)(
        input, weight
    )


def relu(input):
    return Relu()(input)


def linear(input, weight):
    return Linear()(input, weight)


def identity(input):
    return Identity()(input)


def choice(input, choices, switch=None):
    return ChoiceOp(*choices, switch=switch)(input)


def dynamic_depth(*exits, switch):
    return ChoiceOp(*exits, switch=switch)()


def batch_norm(input):
    n_chans = input.shape()[1]
    running_mu = Tensor(name="running_mean", shape=(n_chans,), axis=("c",))
    running_std = Tensor(name="running_std", shape=(n_chans,), axis=("c",))
    return BatchNorm()(input, running_mu, running_std)


def quantize_weight(weight):
    weight = Requantize()(weight)
    return weight


@scope
def block(input, expand_ratio, reduce_ratio):
    max_pool1 = max_pool(input, kernel_size=2, stride=2, padding=1)
    identity1 = identity(input)
    # conv_in = choice(input, [max_pool1, identity1])

    conv_in = input

    weight1 = Tensor(
        "w1",
        (Int(conv_in.shape()[1] * expand_ratio), Int(conv_in.shape()[1]), 1, 1),
        axis=["O", "I", "kH", "kW"],
        grad=True,
    )
    weight1_quantized = quantize_weight(weight1)
    conv1 = conv2d(conv_in, weight1_quantized)
    bn1 = batch_norm(conv1)
    relu1 = relu(bn1)

    weight2 = Tensor(
        "w2",
        (Int(relu1.shape()[1] / reduce_ratio), relu1.shape()[1], 3, 3),
        axis=["O", "I", "kH", "kW"],
        grad=True,
    )
    weight2_quantized = quantize_weight(weight2)
    conv2 = conv2d(relu1, weight2_quantized)
    bn2 = batch_norm(conv2)
    relu2 = relu(bn2)

    return relu2


def search_space(name, input, num_classes=10, max_blocks=9):
    expand_ratio = IntScalarParameter(1, 3, name="expand_ratio")
    reduce_ratio = IntScalarParameter(1, 3, name="reduce_ratio")

    depth = IntScalarParameter(0, max_blocks, name="depth")
    input_channels = IntScalarParameter(8, 64, name="input_channels", step_size=8)

    weight = Tensor(
        "w1",
        (input_channels, input.shape()[1], 3, 3),
        axis=["O", "I", "kH", "kW"],
        grad=True,
    )
    weight = quantize_weight(weight)

    out = conv2d(input, weight)

    blocks = []
    for i in range(max_blocks + 1):
        out = block(out, expand_ratio.new(), reduce_ratio.new())
        blocks.append(out)

    out = dynamic_depth(*blocks, switch=depth)

    linear_weight = Tensor(
        "w2",
        (out.shape()[1] * out.shape()[2] * out.shape()[3], num_classes),
        axis=["O", "I"],
        grad=True,
    )
    linear_weight = quantize_weight(linear_weight)
    out = linear(out, linear_weight)

    return out
