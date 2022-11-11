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
import pytest

from hannah.nas.backend import TorchBackend
from hannah.nas.dataflow.dataflow_graph import dataflow
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor_type import TensorType
from hannah.nas.ops import axis, batched_image_tensor, float_t, tensor
from hannah.nas.parameters.parameters import (
    CategoricalParameter,
    FloatScalarParameter,
    IntScalarParameter,
)


@dataflow
def conv_relu(
    input: TensorType,
    output_channel=IntScalarParameter(4, 64),
    kernel_size=CategoricalParameter([1, 3, 5]),
    stride=CategoricalParameter([1, 2]),
):

    weight = tensor(
        (
            axis("o", size=output_channel),
            axis("i", size=input.tensor_type().axis["c"].size),
            axis("kh", size=kernel_size),
            axis("kw", size=kernel_size),
        ),
        dtype=float_t(),
        name="weight",
    )

    op = OpType("conv2d", input, weight, stride=stride)
    relu = OpType("relu", op)
    return relu


@dataflow
def block(
    input: TensorType,
    expansion=FloatScalarParameter(1, 6),
    output_channel=IntScalarParameter(4, 64),
    kernel_size=CategoricalParameter([1, 3, 5]),
    stride=CategoricalParameter([1, 2]),
):

    out = conv_relu(input, output_channel * expansion, kernel_size, stride)
    out = conv_relu(out, output_channel, 1, 1)
    return out


@pytest.mark.xfail
def test_convert_block():
    input = batched_image_tensor(name="input")
    out = block(input)
    network = block(out)

    backend = TorchBackend

    module = TorchBackend.build(network)

    # print(module)
    # print(module.code)
    # breakpoint()


if __name__ == "__main__":
    test_convert_block()
