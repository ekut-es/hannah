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
"A very simple device description containing supporting only conv2d -> relu -> max_pool2d"

from hannah.nas.expressions.placeholder import UndefinedInt
from hannah.nas.functional_operators.op import ChoiceOp
from hannah.nas.functional_operators.operators import Conv2d, MaxPool2d, Relu, Tensor
from hannah.nas.hardware_description import Device

Dyn = UndefinedInt("Dyn")


class SimpleDevice(Device):
    name = "simple_device"
    description = "A simple Abstract Hardware Device only supporting 2d convolutions with a stride of 1 and same padding"

    def __init__(self, relu=False):
        super().__init__()
        self._add_conv2d()
        if relu:
            self._add_relu()
        self._add_max_pool2d()

        self.add_memory(
            "local",
            size=1024 * 10,
            latency=1,
        )

    def _add_conv2d(self):
        input = Tensor(
            "input", shape=[None, None, None, None], axis=["N", "C", "H", "W"]
        )
        weight = Tensor(
            "weight",
            shape=[None, None, None, None],
            axis=["O", "I", "kH", "kW"],
            grad=True,
        )
        padding = Dyn
        conv = Conv2d(stride=1, padding=padding, dilation=1, groups=1)(input, weight)

        self.add_op(
            "conv2d",
            conv,
            [padding == weight.size(2) // 2, padding == weight.size(3) // 2],
        )

    def _add_relu(self):
        input = Tensor(
            "input", shape=[None, None, None, None], axis=["N", "C", "H", "W"]
        )
        relu = Relu()(input)

        self.add_op("relu", relu)

    def _add_max_pool2d(self):
        input = Tensor(
            "input", shape=[None, None, None, None], axis=["N", "C", "H", "W"]
        )
        max_pool = MaxPool2d(kernel_size=2, stride=2, padding=0)(input)

        self.add_op("max_pool2d", max_pool)


def get_device(name, relu=False, *args, **kwargs):
    simple_device = SimpleDevice()
    simple_device.name = name
    if relu:
        simple_device._add_relu()

    return simple_device
