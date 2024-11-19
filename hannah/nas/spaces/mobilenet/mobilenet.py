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
from hannah.nas.dataflow.dataflow_graph import dataflow
from hannah.nas.dataflow.registry import op
from hannah.nas.dataflow.repeat import repeat
from hannah.nas.expressions.placeholder import DefaultFloat, DefaultInt
from hannah.nas.ops import weight_tensor
from hannah.nas.parameters.parameters import (
    CategoricalParameter,
    FloatScalarParameter,
    IntScalarParameter,
)
from hannah.nas.dataflow.ops import (
    conv2d,
    batch_nom,
    relu,
    dropout,
    pooling,
    linear,
    add,
    identity,
)  # noqa


@dataflow
def residual(input):
    return op("Identity", input)


@dataflow
def add(input, other):  # noqa: F811
    return op("Add", input, other)


@dataflow
def conv_bn_relu(
    input,
    out_channel,
    kernel_size=DefaultInt(1),
    stride=DefaultInt(1),
    groups=DefaultInt(1),
):
    input_tensor = input.tensor_type()
    weight = weight_tensor(
        shape=(out_channel, input_tensor["c"], kernel_size, kernel_size), name="weight"
    )
    conv = op("Conv2d", input, weight, stride=stride, groups=groups)
    bn = op("BatchNorm2d", conv)
    act = op("Relu", bn)
    return act


@dataflow
def depthwise_separable_convolution(
    input, out_channel, kernel_size, stride, expand_ratio
):
    input_tensor = input.tensor_type()
    expand_conv = conv_bn_relu(input, out_channel=input_tensor["c"].size * expand_ratio)
    depthwise_conv = conv_bn_relu(
        expand_conv,
        out_channel=input_tensor["c"].size * expand_ratio,
        kernel_size=kernel_size,
        stride=stride,
        groups=input_tensor["c"].size,
    )
    pointwise_conv = conv_bn_relu(depthwise_conv, out_channel=out_channel)
    return pointwise_conv


@dataflow
def stem(input, out_channel, stride, kernel_size=DefaultInt(3)):
    conv = conv_bn_relu(
        input, out_channel=out_channel, stride=stride, kernel_size=kernel_size
    )
    return conv


@dataflow
def inverted_block(input, out_channel, expand_ratio, stride):
    conv = depthwise_separable_convolution(
        input,
        out_channel=out_channel,
        kernel_size=DefaultInt(3),
        expand_ratio=expand_ratio,
        stride=stride,
    )
    # TODO: conditional residual connection (only if applicable?)
    res = residual(input)
    residual_add = add(conv, res)
    return residual_add


@dataflow
def classifier(input, classes):
    drp = op("Dropout2d", input, p=DefaultFloat(0.5))
    avg_pool = op("AdaptiveAveragePooling", drp, output_size=DefaultInt(1))
    fc = op("Linear", avg_pool, out_features=classes)
    return fc


@dataflow
def mobilenet(input, num_cells):
    out_channel = IntScalarParameter(min=4, max=128)
    expand_ratio = FloatScalarParameter(min=0.5, max=6.0)
    stride = CategoricalParameter(choices=[1, 2])
    s = stem(input, out_channel=out_channel.new(), stride=DefaultInt(2))
    graph = repeat(inverted_block, num_repeats=num_cells)(
        s,
        out_channel=out_channel.new(),
        expand_ratio=expand_ratio.new(),
        stride=stride.new(),
    )
    # graph = inverted_block(s, out_channel=out_channel.new(), expand_ratio=expand_ratio.new(), stride=stride.new())
    clf = classifier(graph, classes=DefaultInt(10))
    return clf
