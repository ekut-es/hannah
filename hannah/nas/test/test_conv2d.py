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

from hannah.nas.dataflow.dataflow_graph import dataflow
from hannah.nas.dataflow.ops import conv2d  # noqa  #Import to load in registry
from hannah.nas.dataflow.registry import op
from hannah.nas.expressions.placeholder import DefaultInt
from hannah.nas.ops import batched_image_tensor, weight_tensor
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter


@dataflow
def conv2d(  # noqa
    input,
    channel,
    kernel_size=DefaultInt(1),
    stride=DefaultInt(1),
    dilation=DefaultInt(1),
):
    weight = weight_tensor(
        shape=(channel, input["c"], kernel_size, kernel_size), name="weight"
    )
    padding = kernel_size // 2
    return op(
        "Conv2d", input, weight, dilation=dilation, stride=stride, padding=padding
    )


@dataflow
def chained_convs(
    input,
    channel,
    kernel_size=DefaultInt(1),
    stride=DefaultInt(1),
    dilation=DefaultInt(1),
):
    padding = kernel_size // 2
    weight1 = weight_tensor(
        shape=(channel, input["c"], kernel_size, kernel_size), name="weight"
    )
    conv1 = op(
        "Conv2d", input, weight1, dilation=dilation, stride=stride, padding=padding
    )

    weight2 = weight_tensor(
        shape=(channel, input["c"], kernel_size, kernel_size), name="weight"
    )
    conv2 = op(
        "Conv2d", conv1, weight2, dilation=dilation, stride=stride, padding=padding
    )

    return conv2


@pytest.mark.xfail()
def test_conv2d():
    inp = batched_image_tensor(name="input")

    kernel_size = CategoricalParameter([1, 3, 5])
    stride = CategoricalParameter([1, 2])

    conv = conv2d(
        inp, channel=IntScalarParameter(4, 64), kernel_size=kernel_size, stride=stride
    )

    conv["conv2d.0.Conv2d.0"].kernel_size.set_current(3)
    conv["conv2d.0.Conv2d.0"].stride.set_current(2)

    returned_tensor = conv.output.tensor_type()
    for name, ax in returned_tensor.tensor_type.axis.items():
        print("{}: {}".format(name, ax.size.evaluate()))
    print()


def test_chained_conv2d():
    inp = batched_image_tensor(name="input")

    ks = CategoricalParameter([1, 3, 5])
    ks.set_current(3)
    convs = chained_convs(inp, channel=IntScalarParameter(4, 64), kernel_size=ks)
    returned_tensor = convs.output.tensor_type()

    for name, ax in returned_tensor.axis.items():
        print("{}: {}".format(name, ax.size.evaluate()))
    print()


if __name__ == "__main__":
    # test_conv2d()
    test_chained_conv2d()
    print()
