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
import pytest
import torch

from hannah.quantization.rounding import RoundingMode


@pytest.mark.parametrize(
    "mode,expected",
    # fmt: off
    [
        #                  [-1.9, -1.5, -1.0, -0.5, 0.0, 0.5, 1.5, 1.9, 0.49,1.51]
        ("DOWNWARD", [-2.0, -2.0, -1.0, -1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0]),
        ("UPWARD", [-2.0, -1.0, -1.0, -0.0, 0.0, 1.0, 2.0, 2.0, 0.0, 2.0]),
        ("ODD", [-2.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 2.0, 0.0, 2.0]),
        ("EVEN", [-2.0, -2.0, -1.0, -0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0]),
        ("ZERO", [-2.0, -1.0, -1.0, -0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0]),
        ("INFINITY", [-2.0, -2.0, -1.0, -1.0, 0.0, 1.0, 2.0, 2.0, 0.0, 2.0]),
        ("TRUNC_DOWN", [-2.0, -2.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
        ("TRUNC_UP", [-1.0, -1.0, -1.0, -0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 2.0]),
        ("TRUNC_ZERO", [-1.0, -1.0, -1.0, -0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
        ("TRUNC_INFINITY", [-2.0, -2.0, -1.0, -1.0, 0.0, 1.0, 2.0, 2.0, 1.0, 2.0]),
    ],
    # fmt: on
)
def test_rounding(mode, expected):
    data = torch.tensor([-1.9, -1.5, -1.0, -0.5, 0.0, 0.5, 1.5, 1.9, 0.49, 1.51])
    round = RoundingMode(mode)
    assert torch.allclose(round(data), torch.tensor(expected))


def test_stochastic_rounding():
    round = RoundingMode("STOCHASTIC")
    length = 100000

    for val in [-1.7, 0.3, 2.8, 10.5, 17.3789123]:
        data = torch.full((length,), val)
        result = round(data)
        average = float(torch.sum(result) / length)
        assert (val - 0.01) <= average <= (val + 0.01)
