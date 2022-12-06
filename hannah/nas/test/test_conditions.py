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
from hannah.nas.parameters import parametrize
from hannah.nas.parameters.parameters import IntScalarParameter


@parametrize
class Accelerator:
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b

    def __repr__(self):
        return "Accelerator(" + repr(self.a) + ", " + repr(self.b) + ")"


def test_condition():
    a_param = IntScalarParameter(0, 10)
    b_param = IntScalarParameter(0, 10)

    accelerator = Accelerator(a_param, b_param)
    accelerator.cond(a_param + b_param < 10)
    accelerator.cond(a_param > 5)
    try:
        accelerator_instance = accelerator.instantiate()
    except Exception:
        pass
    accelerator.set_current({"a": 6})
    accelerator_instance = accelerator.instantiate()
    try:
        accelerator.set_current({"a": 6, "b": 6})
    except Exception:
        pass

    assert accelerator_instance.a == 6
    assert accelerator_instance.b == 0

    other_a_param = IntScalarParameter(0, 10)
    other_b_param = IntScalarParameter(0, 10)
    other_accelerator = Accelerator(other_a_param, other_b_param)

    other_accelerator.cond(other_a_param == accelerator.a)

    other_accelerator.set_current({"a": 6})

    print()


if __name__ == "__main__":
    test_condition()
