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
from hannah.nas.functional_operators.op import search_space
from hannah.nas.functional_operators.operators import Conv2d, Tensor

from ..device import Device


class EyerissDevice(Device):
    name = "eyeriss_v1"
    description = """
    Eyeriss version 1 hwa description.

    https://courses.cs.washington.edu/courses/cse550/21au/papers/CSE550.Eyeriss.pdf
    """

    def __init__(self, precision=8):
        super().__init__()

        self._add_conv2d()
        self._add_memories()

    def _add_conv2d(self):
        N = UndefinedInt("N")
        C = UndefinedInt("C")
        H = UndefinedInt("H")
        W = UndefinedInt("W")

        input = Tensor("input", shape=[N, C, H, W], axis=["N", "C", "H", "W"])

        OC = UndefinedInt("O")
        IC = UndefinedInt("I")
        kh = UndefinedInt("kh")
        kw = UndefinedInt("kw")

        weight = Tensor(
            "weight",
            shape=[OC, IC, kh, kw],
            axis=["O", "I", "kh", "kW"],
            grad=True,
        )

        sh = UndefinedInt("vertical_stride")
        sw = UndefinedInt("horizonzal_stride")

        conv = Conv2d(stride=(), padding=0, dilation=1, groups=1)(input, weight)

        self.add_op(
            "conv2d",
            conv,
            [
                sh <= 4,
                sh % 2 == 0,
                sh > 1,
                sw >= 1,
                sw <= 12,
                C <= 1,
                kh <= 12,
                kw <= 32,
                IC <= 1024,
                OC <= 1024,
            ],
        )

    def _add_memories(self):
        self.add_memory(
            "ifmap_buffer",
            wordwidth=16,
            size=12,
            latency=1,
            read_port=1,
            write_port=1,
            read_bw=1,
            write_bw=1,
        )

        self.add_memory(
            "weight_buffer",
            wordwidth=16,
            size=224,
            latency=1,
            read_port=1,
            write_port=1,
            read_bw=1,
            write_bw=1,
        )

        self.add_memory(
            "ps_buffer",
            wordwidth=16,
            size=32,
            latency=1,
            read_port=1,
            write_port=1,
            read_bw=1,
            write_bw=1,
        )

        self.add_memory(
            "global_buffer",
            wordwidth=16,
            size=1024 * 12,
            latency=1,
        )
