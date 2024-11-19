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
from hannah.nas.functional_operators.op import ChoiceOp, Tensor, scope, search_space
from hannah.nas.functional_operators.operators import (
    BatchNorm,
    Conv2d,
    Identity,
    Linear,
    AvgPooling,
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


def avg_pool(input, kernel_size=2, stride=2, dilation=1):
    return AvgPooling(kernel_size=kernel_size, stride=stride, dilation=dilation)(input)


def conv2d(input, weight, bias=None, stride=1, dilation=1, groups=1, padding=None):
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
    # weight = Requantize()(weight)
    return weight


def quantize_activation(activation):
    # activation = Requantize()(activation)
    return activation


@scope
def block(input, channels: int, kernel_size: int):
    max_pool1 = max_pool(input, kernel_size=2, stride=2, padding=0)

    weight = Tensor(
        "w1",
        (Int(channels), Int(max_pool1.shape()[1]), kernel_size, kernel_size),
        axis=["O", "I", "kH", "kW"],
        grad=True,
    )
    weight_quantized = quantize_weight(weight)
    bias = Tensor(
        "b1",
        (Int(channels),),
        axis=["O", "I", "kH", "kW"],
        grad=True,
    )
    bias_quantized = quantize_weight(bias)
    conv = conv2d(max_pool1, weight_quantized, bias_quantized)
    act = quantize_activation(relu(conv))

    return act


@search_space
def ai8x_search_space(name, input, num_classes=10, max_blocks=9, rng=None):
    # do exactly 5 blocks so we get an input shape [1,1] for linear
    max_blocks = 5  # min(max_blocks, 4)
    depth = IntScalarParameter(0, max_blocks - 1, name="depth", rng=rng)
    channel_multiplier = IntScalarParameter(
        1, 16, name="channels", step_size=1, rng=rng
    )

    assert (
        len(input.shape()) == 4
    ), f"a shape of [1, C, W, H] is required, and {input.shape()} was provided"
    assert input.shape()[0] == 1, "it is required to have shape[0] == 1"

    out = input

    blocks = []
    for i in range(max_blocks):
        out = block(out, i + 1 * channel_multiplier, 3)

        blocks.append(out)

    # out = dynamic_depth(*blocks, switch=depth)

    linear_weight = Tensor(
        "lin_weight_w2",
        (out.shape()[1] * out.shape()[2] * out.shape()[3], num_classes),
        axis=["O", "I"],
        grad=True,
    )
    linear_weight = quantize_weight(linear_weight)
    out = linear(out, linear_weight)

    return out
