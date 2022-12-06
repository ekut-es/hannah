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

from hannah.nas.hardware_description.device import Ultratrail
from hannah.nas.parameters import IntScalarParameter


@pytest.mark.xfail
def test_ultratrail_description():
    ultratrail = Ultratrail(
        weight_bits=IntScalarParameter(min=1, max=8),
        bias_bits=IntScalarParameter(min=1, max=8),
        activation_bits=IntScalarParameter(min=1, max=8),
        accumulator_bits=IntScalarParameter(min=1, max=32),
        max_weight_bits=IntScalarParameter(min=4, max=8),
    )

    print(ultratrail)


if __name__ == "__main__":
    test_ultratrail_description()
