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
def block(
    input, channels: int, kernel3x3: bool, kspo: tuple[int, int, int, str | None]
):
    max_pool1 = max_pool(input, kernel_size=kspo[0], stride=kspo[1], padding=0)
    avg_pool1 = avg_pool(input, kernel_size=kspo[0], stride=kspo[1])
    identity1 = identity(input)
    # conv_in = choice(input, [max_pool1, avg_pool1, identity1])
    if kspo[3] in ["max"]:
        conv_in = max_pool1
    elif kspo[3] in ["avg"]:
        conv_in = avg_pool1
    elif kspo[3] in [None]:
        conv_in = identity1
    else:
        raise Exception("wrong operation")

    kernelsize = 3 if kernel3x3 else 1  # else 1x1
    weight1 = Tensor(
        "w1",
        (Int(channels), Int(conv_in.shape()[1]), kernelsize, kernelsize),
        axis=["O", "I", "kH", "kW"],
        grad=True,
    )
    weight1_quantized = quantize_weight(weight1)
    bias1 = Tensor(
        "b1",
        (Int(channels),),
        axis=["C"],
        grad=True,
    )
    bias1_quantized = quantize_weight(bias1)
    conv1 = conv2d(conv_in, weight1_quantized, bias1_quantized)

    bn1 = batch_norm(conv1)
    bn1 = conv1
    act1 = quantize_activation(relu(bn1))

    return act1


@search_space
def ai8x_search_space(name, input, num_classes=10, max_blocks=9, rng=None):
    kernel3x3 = CategoricalParameter([True, False], name="kernel3x3", rng=rng)
    pool_kernel = CategoricalParameter([2], name="pool_kernel", rng=rng)
    pool_stride = CategoricalParameter([1, 2], name="pool_stride", rng=rng)
    pool_pad = CategoricalParameter([0], name="pool_pad", rng=rng)  # has to stay 0
    pool_op = CategoricalParameter(["max"], name="pool_op", rng=rng)  # , "avg", None
    channels = CategoricalParameter(
        # single passes = 1 to 64
        # [i for i in range(1, 64 + 1)],
        # multipasses = 1*64, 2*64, 3*64, 4*64
        # + [j * i for j in range(1, 4 + 1) for i in range(64, 64 + 1)],
        [2**i for i in range(6 + 1)],  # 1,2,4,8,16,32,64
        name="channels",
        rng=rng,
    )
    max_blocks = min(max_blocks, 4)
    depth = IntScalarParameter(0, max_blocks, name="depth", rng=rng)
    input_channels = IntScalarParameter(
        8, 64, name="input_channels", step_size=8, rng=rng
    )

    assert (
        len(input.shape()) == 4
    ), f"a shape of [1, C, W, H] is required, and {input.shape()} was provided"
    assert input.shape()[0] == 1, "it is required to have shape[0] == 1"

    out = block(input, channels.new(), kernel3x3.new(), (2, 2, 1, "max"))

    blocks = []
    channelslist = []
    for i in range(max_blocks + 1):
        channelslist.append(channels.new())
        kspo = (pool_kernel.new(), pool_stride.new(), 1, pool_op.new())
        out = block(out, channelslist[-1], kernel3x3.new(), kspo)
        max_shape = out.shape()[1] * out.shape()[2] * out.shape()[3]
        out.cond(
            max_shape > 0 and max_shape <= 1024,
            allowed_params=[
                p for k, p in out.parametrization().items() if p.name in ["channel"]
            ]
            + [kspo[0], kspo[1], kspo[3], kernel3x3],
        )
        blocks.append(out)

    out = dynamic_depth(*blocks, switch=depth)

    # maybe fix input = output channels so that izer wont complain
    out = block(out, num_classes, True, (2, 2, 0, None))

    linear_weight = Tensor(
        "lin_weight_w2",
        (out.shape()[1] * out.shape()[2] * out.shape()[3], num_classes),
        axis=["O", "I"],
        grad=True,
    )
    linear_weight = quantize_weight(linear_weight)
    out = linear(out, linear_weight)

    out.cond(
        out.shape()[0] < 1025 and out.shape()[0] > 0,
        allowed_params=[
            p
            for k, p in out.parametrization().items()
            if p.name in ["channels", "depth"]
        ],
    )

    return out
