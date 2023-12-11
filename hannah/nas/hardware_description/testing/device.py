#
# Copyright (c) 2023 Hannah contributors.
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
" A very simple device description containing supporting only conv2d -> relu -> max_pool2d "

from hannah.nas.functional_operators.operators import Conv2d, Tensor
from hannah.nas.hardware_description import Device


def get_device(name, *args, **kwargs):
    simple_device = Device(
        name,
        """
        A simple example device supporting the following operations:

        1. Conv2d
        2. ReLU
        3. MaxPool2d

        Operations need to appear in the order specified above.

        ReLU and MaxPool2d are optional.

        Conv2d inputs are int8 tensors, outputs are int32 tensors.
        ReLU and MaxPool2d inputs and outputs are int32 tensors.
        Final outpus are requantized to int8.
        """,
    )

    input = Tensor(
        "input", shape=["Dyn", "Dyn", "Dyn" "Dyn"], axis=["N", "C", "H", "W"]
    )
    weight = Tensor(
        "weight", shape=["Dyn", "Dyn", "Dyn", "Dyn"], axis=["O", "I", "kH", "kW"]
    )

    op = Conv2d(stride=1, padding=0, dilation=1, groups=1)(input, weight)

    simple_device.add_op("conv2d", op, [])

    return simple_device
