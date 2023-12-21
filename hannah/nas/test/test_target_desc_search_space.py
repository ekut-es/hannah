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
from hannah.nas.functional_operators.operators import Conv2d, Tensor
from hannah.nas.hardware_description.backend import HannahBackend
from hannah.nas.hardware_description.testing import get_device
from hannah.nas.parameters import CategoricalParameter


def get_simple_space(kernel_size=3):
    input = Tensor("input", shape=[1, 3, 32, 32], axis=["N", "C", "H", "W"])
    weight = Tensor(
        "weight", shape=[16, 3, kernel_size, kernel_size], axis=["O", "I", "kH", "kW"]
    )
    conv = Conv2d(stride=1, padding=1)(input, weight)

    return conv


def test_simple_device():
    simple_device = get_device("simple_device")

    backend = HannahBackend()
    hannah_target = backend.generate(simple_device)

    simple_space = get_simple_space()

    hannah_target.run(simple_space)

    for match in hannah_target.matches:
        print(match)


def test_simple_device_categorical():
    simple_device = get_device("simple_device")

    backend = HannahBackend()
    hannah_target = backend.generate(simple_device)

    simple_space = get_simple_space(CategoricalParameter("kernel_size", [3, 5, 7]))

    hannah_target.run(simple_space)

    for match in hannah_target.matches:
        print(match)


if __name__ == "__main__":
    test_simple_device()
