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
from typing import Any

from hannah.nas.functional_operators.operators import (
    Relu,
    Conv1d,
    Linear,
    AdaptiveAvgPooling,
)
from hannah.nas.parameters import CategoricalParameter, IntScalarParameter, parametrize
from hannah.nas.functional_operators.op import Tensor, Op, scope, ChoiceOp, search_space
from hannah.nas.functional_operators.shapes import conv_shape, padding_expression
from hannah.nas.functional_operators.lazy import lazy


import torch


def conv1d(input, out_channels, kernel_size, stride):
    in_channels = input.shape()[1]
    weight = Tensor(
        name="weight",
        shape=(out_channels, in_channels, kernel_size),
        axis=("O", "I", "k"),
        grad=True,
    )

    conv = Conv1d(stride=stride)(input, weight)
    return conv


def relu(input):
    return Relu()(input)


def adaptive_avg_pooling(input):
    return AdaptiveAvgPooling(output_size=1)(input)


def linear(input, num_classes):
    in_features = input.shape()[1]
    weight = Tensor(
        name="weight", shape=(in_features, num_classes), axis=("I", "O"), grad=True
    )
    return Linear()(input, weight)


@scope
def conv_relu(input, out_channels, kernel_size, stride):
    out = conv1d(
        input, out_channels=out_channels, stride=stride, kernel_size=kernel_size
    )
    out = relu(out)
    return out


@scope
def classifier_head(input, num_classes):
    out = adaptive_avg_pooling(input)
    out = linear(out, num_classes)
    return out


def dynamic_depth(*exits, switch):
    return ChoiceOp(*exits, switch=switch)()


@search_space
def space(name: str, input, num_classes: int, max_channels=512, max_depth=9):
    num_blocks = IntScalarParameter(0, max_depth, name="num_blocks")
    exits = []

    out = input

    for i in range(num_blocks.max + 1):
        kernel_size = CategoricalParameter([3, 5, 7, 9], name="kernel_size")
        stride = CategoricalParameter([1, 2], name="stride")
        out_channels = IntScalarParameter(
            16, max_channels, step_size=8, name="out_channels"
        )

        out = conv_relu(
            out, out_channels=out_channels, kernel_size=kernel_size, stride=stride
        )
        exits.append(out)

    out = dynamic_depth(*exits, switch=num_blocks)

    out = classifier_head(out, num_classes=num_classes)

    return out
