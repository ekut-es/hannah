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

from hannah.nas.expressions.arithmetic import Add
from hannah.nas.expressions.placeholder import DefaultInt
from hannah.nas.parameters.parameters import IntScalarParameter


def test_print():
    add = Add(5, 5)
    print(str(add))

    add_param = Add(IntScalarParameter(0, 10), 10)
    print(add_param)


def test_evaluate():
    add = Add(5, 5)
    res = add.evaluate()
    assert res == 10

    add_param = Add(IntScalarParameter(0, 10), 10)
    res = add_param.evaluate()
    assert res in [10, 20]


def test_expressions():
    add = IntScalarParameter(0, 10) + 10
    res = add.evaluate()
    assert res in [10, 20]
    add2 = IntScalarParameter(0, 10) + IntScalarParameter(0, 10)
    res = add2.evaluate()
    assert res in [0, 20]


def test_and_expr():
    one = DefaultInt(1)
    zero = DefaultInt(0)

    assert (one & one).evaluate()
    assert not (one & zero).evaluate()


def test_le_expr():
    one = DefaultInt(1)
    zero = DefaultInt(0)

    assert (one < one).evaluate() is False
    assert (one < zero).evaluate() is False
    assert (zero < one).evaluate() is True
    assert (zero < zero).evaluate() is False


def test_complex_expressions():
    expr = IntScalarParameter(0, 10) + IntScalarParameter(0, 10) * (
        IntScalarParameter(0, 1) + 5
    )
    res = expr.evaluate()
    assert res in [0, 100]


@pytest.mark.parametrize(
    "x,y",
    [
        (IntScalarParameter(0, 0), 2),
        (IntScalarParameter(0, 0), IntScalarParameter(0, 0)),
        (DefaultInt(0), 2),
    ],
)
def test_unimplemeted(x, y):
    """Test that unimplemented methods raise unimplemented errors"""

    x = IntScalarParameter(0, 0)

    with pytest.raises(NotImplementedError):
        _ = x**2

    with pytest.raises(NotImplementedError):
        _ = divmod(x, 2)

    with pytest.raises(NotImplementedError):
        _ = x << 2

    with pytest.raises(NotImplementedError):
        _ = x >> 2

    with pytest.raises(NotImplementedError):
        _ = divmod(x, x)

    with pytest.raises(NotImplementedError):
        _ = x ^ x


if __name__ == "__main__":
    test_print()
    test_evaluate()
    test_expressions()
    test_and_expr()
    test_le_expr()
    test_complex_expressions()
