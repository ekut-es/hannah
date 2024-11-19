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
from hannah.nas.expressions.placeholder import UndefinedInt
from hannah.nas.functional_operators.op import ChoiceOp
from hannah.nas.functional_operators.operators import Conv2d, Tensor
from hannah.nas.hardware_description import Device
from hannah.nas.functional_operators.op import context, search_space

class VanillaAccelerator(Device):
    name = "vanilla_accelerator"
    description = "A simple Abstract Hardware Device only supporting 2d convolutions with a stride of 1 and same padding"


    def __init__(self):
        super().__init__()
        self._add_conv2d()

        self.add_memory(
            "local",
            size=1024 * 10,
            latency=1,
        )

    def _add_conv2d(self):
        input = Tensor(
            "input", shape=[None, None, None, None], axis=["N", "C", "H", "W"]
        )  # NCHW tensor format
        weight = Tensor(
            "weight",
            shape=[None, None, None, None],
            axis=["O", "I", "kH", "kW"],
            grad=True,
        )
        padding = UndefinedInt("padding")
        conv = Conv2d(stride=1, padding=padding, dilation=1, groups=1)(input, weight)

        self.add_op(
            "conv2d",
            conv,
            [padding == weight.size(2) // 2, padding == weight.size(3) // 2],
        )
