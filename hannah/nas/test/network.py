#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
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
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.ops import (  # noqa: F401 (Import to load in registry)
    add,
    conv2d,
)
from hannah.nas.dataflow.registry import op
from hannah.nas.dataflow.tensor_expression import TensorExpression
from hannah.nas.ops import weight_tensor
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter


@dataflow
def conv_relu(
    input: TensorExpression,
    output_channel=IntScalarParameter(4, 64),
    kernel_size=CategoricalParameter([1, 3, 5]),
    stride=CategoricalParameter([1, 2]),
):
    input_tensor = input.tensor_type()
    weight = weight_tensor(
        shape=(output_channel, input_tensor["c"], kernel_size, kernel_size),
        name="weight",
    )

    c = op("Conv2d", input, weight, stride=stride)
    relu = OpType(c, name="Relu")
    return relu


@dataflow
def block(
    input: TensorExpression,
    expansion=IntScalarParameter(1, 6),
    output_channel=IntScalarParameter(4, 64),
    kernel_size=CategoricalParameter([1, 3, 5]),
    stride=CategoricalParameter([1, 2]),
):
    input_tensor = input.tensor_type()
    out = conv_relu(
        input, output_channel=output_channel, kernel_size=kernel_size, stride=stride
    )
    out = conv_relu(
        out, output_channel=output_channel.new(), kernel_size=kernel_size, stride=stride
    )
    out = conv_relu(
        out, output_channel=output_channel.new(), kernel_size=kernel_size, stride=stride
    )
    return out


@dataflow
def residual(input: TensorExpression, stride, output_channel):
    out = conv_relu(
        input,
        stride=stride,
        output_channel=output_channel.new(),
        kernel_size=CategoricalParameter([1, 3, 5]),
    )
    return out


@dataflow
def add(input: TensorExpression, other: TensorExpression):  # noqa
    out = op("Add", input, other)
    return out


@dataflow
def residual_block(input: TensorExpression, stride, output_channel):
    main_branch = block(input, stride=stride, output_channel=output_channel)
    residual_branch = residual(input, stride=stride, output_channel=output_channel)
    add_branches = add(main_branch, residual_branch)
    return add_branches
