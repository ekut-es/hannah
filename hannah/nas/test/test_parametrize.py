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
from typing import Optional

from numpy.random import default_rng

from hannah.nas.parameters import FloatScalarParameter, IntScalarParameter, parametrize


@parametrize
class ParametrizedTest:
    def __init__(self, a: int, b: Optional[str] = None, c: str = "test"):
        self.a = a
        self.b = b
        self.c = c
        self.id = "parametrized_test"


@parametrize
class NestedParametrizedTest:
    def __init__(self, t: ParametrizedTest = None):
        self.t = t


def test_parametrization():
    parametrized_test = ParametrizedTest(a=IntScalarParameter(min=10, max=20, name="a"))
    assert isinstance(parametrized_test.a, IntScalarParameter)
    assert parametrized_test.b is None
    assert parametrized_test.c == "test"
    assert hasattr(parametrized_test, "_PARAMETERS") and isinstance(
        parametrized_test._PARAMETERS["a"], IntScalarParameter
    )
    assert (
        hasattr(parametrized_test, "_parametrized") and parametrized_test._parametrized
    )


def test_sample():
    rng = default_rng(seed=321)
    parametrized_test = ParametrizedTest(
        a=IntScalarParameter(min=10, max=20, name="a", rng=rng)
    )
    parametrized_test.sample()
    assert parametrized_test.a.current_value == 14
    assert parametrized_test.a.instantiate() == 14


def test_sample_float():
    rng = default_rng(seed=321)
    parametrized_test = ParametrizedTest(
        a=FloatScalarParameter(min=0.1, max=3, name="a", rng=rng)
    )
    parametrized_test.sample()
    assert parametrized_test.a.current_value == 2.010423416621207
    assert parametrized_test.a.instantiate() == 2.010423416621207


def test_sample_nested():
    rng = default_rng(seed=321)
    parametrized_test = NestedParametrizedTest(
        t=ParametrizedTest(a=IntScalarParameter(min=10, max=11, name="a", rng=rng))
    )
    parametrized_test.sample()

    assert (
        parametrized_test.t.a.current_value == 10
    )  # FIXME: dont rely on rng seed for deterministic random number generation across machines


def test_set_current():
    pass


def test_set_params():
    rng = default_rng(seed=321)
    parametrized_test = ParametrizedTest(
        a=IntScalarParameter(min=10, max=20, rng=rng, name="a")
    )
    parametrized_test.set_params(a=16)

    assert parametrized_test.a.current_value == 16


def test_set_params_nested():
    rng = default_rng(seed=321)
    parametrized_test = NestedParametrizedTest(
        t=ParametrizedTest(a=IntScalarParameter(min=10, max=20, rng=rng, name="a"))
    )
    other_test_object = ParametrizedTest(
        a=IntScalarParameter(min=0, max=3, rng=rng, name="a")
    )

    parametrized_test.set_params(t=other_test_object)
    assert parametrized_test.t == other_test_object

    marker = False
    try:
        parametrized_test.set_params(t=6)
    except Exception:
        marker = True
    assert marker

    parametrized_test.set_params(t={"a": 2})
    assert parametrized_test.t.a.current_value == 2

    marker = False
    try:
        parametrized_test.set_params(t={"a": 5})
    except Exception:
        marker = True
    assert marker


def test_instantiate():
    rng = default_rng(seed=321)
    parametrized_test = ParametrizedTest(
        a=IntScalarParameter(min=10, max=20, rng=rng, name="a")
    )
    parametrized_test.sample()
    instance = parametrized_test.instantiate()
    assert instance.a == 14

    parametrized_test.set_params(a=16)
    instance = parametrized_test.instantiate()

    assert isinstance(instance, ParametrizedTest)
    assert instance._parametrized is False
    assert instance.a == 16

    rng = default_rng(seed=321)
    parametrized_test = NestedParametrizedTest(
        t=ParametrizedTest(a=IntScalarParameter(min=10, max=20, rng=rng, name="a"))
    )
    other_test_object = ParametrizedTest(
        a=IntScalarParameter(min=0, max=3, rng=rng, name="a")
    )

    parametrized_test.set_params(t=other_test_object)
    parametrized_test.set_params(t={"a": 2})

    instance = parametrized_test.instantiate()

    assert isinstance(instance, NestedParametrizedTest)
    assert isinstance(instance.t, ParametrizedTest)
    assert instance._parametrized is False
    assert instance.t._parametrized is False
    assert instance.t.a == 2


if __name__ == "__main__":
    test_parametrization()
    test_sample()
    test_sample_float()
    test_sample_nested()
    test_set_current()
    test_set_params()
    test_set_params_nested()
    test_instantiate()
