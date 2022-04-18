from typing import Optional
from hannah.nas.parameters.parameters import parametrize, IntScalarParameter, FloatScalarParameter
from numpy.random import default_rng


@parametrize
class Test:
    def __init__(self, a: int, b: Optional[str] = None, c: str = "test"):
        self.a = a
        self.b = b
        self.c = c


@parametrize
class NestedTest:
    def __init__(self, t: Test = None):
        self.t = t


# conv = conv2d(wildcard(), wildcard(), out_channels=IntScalarParameter(32, 64))
# relu(conv)

# conv = op('conv', out_channels=IntScalarParameter(32, 64))
# relu = op('relu', conv)
def test_parametrization():
    parametrized_test = Test(a=IntScalarParameter(min=10, max=20))
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
    parametrized_test = Test(a=IntScalarParameter(min=10, max=20, rng=rng))
    parametrized_test.sample()
    assert parametrized_test.a.current_value == 13
    assert parametrized_test.a.instantiate() == 13


def test_sample_float():
    rng = default_rng(seed=321)
    parametrized_test = Test(a=FloatScalarParameter(min=0.1, max=3, rng=rng))
    parametrized_test.sample()
    assert parametrized_test.a.current_value == 2.010423416621207
    assert parametrized_test.a.instantiate() == 2.010423416621207


def test_sample_nested():
    rng = default_rng(seed=321)
    parametrized_test = NestedTest(t=Test(a=IntScalarParameter(min=10, max=20, rng=rng)))
    parametrized_test.sample()

    assert parametrized_test.t.a.current_value == 13


def test_set_current():
    pass


def test_set_params():
    rng = default_rng(seed=321)
    parametrized_test = Test(a=IntScalarParameter(min=10, max=20, rng=rng))
    parametrized_test.set_params(a=16)

    assert parametrized_test.a.current_value == 16


def test_set_params_nested():
    rng = default_rng(seed=321)
    parametrized_test = NestedTest(t=Test(a=IntScalarParameter(min=10, max=20, rng=rng)))
    other_test_object = Test(a=IntScalarParameter(min=0, max=3, rng=rng))

    parametrized_test.set_params(t=other_test_object)
    assert parametrized_test.t == other_test_object

    marker = False
    try:
        parametrized_test.set_params(t=6)
    except Exception:
        marker = True
    assert marker

    parametrized_test.set_params(t={'a': 2})
    assert parametrized_test.t.a.current_value == 2

    marker = False
    try:
        parametrized_test.set_params(t={'a': 5})
    except Exception:
        marker = True
    assert marker


def test_instantiate():
    rng = default_rng(seed=321)
    parametrized_test = Test(a=IntScalarParameter(min=10, max=20, rng=rng))
    parametrized_test.sample()
    instance = parametrized_test.instantiate()
    assert instance.a == 13

    parametrized_test.set_params(a=16)
    instance = parametrized_test.instantiate()

    assert isinstance(instance, Test)
    assert instance._parametrized is False
    assert instance.a == 16

    rng = default_rng(seed=321)
    parametrized_test = NestedTest(t=Test(a=IntScalarParameter(min=10, max=20, rng=rng)))
    other_test_object = Test(a=IntScalarParameter(min=0, max=3, rng=rng))

    parametrized_test.set_params(t=other_test_object)
    parametrized_test.set_params(t={'a': 2})

    instance = parametrized_test.instantiate()

    assert isinstance(instance, NestedTest)
    assert isinstance(instance.t, Test)
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
