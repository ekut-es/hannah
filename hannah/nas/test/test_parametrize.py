from typing import Optional
from hannah.nas.parameters.parameters import parametrize, IntScalarParameter


@parametrize
class Test:
    def __init__(self, a: int, b: Optional[str] = None, c: str = "test"):
        self.a = a
        self.b = b
        self.c = c


# conv = conv2d(wildcard(), wildcard(), out_channels=IntScalarParameter(32, 64))
# relu(conv)

# conv = op('conv', out_channels=IntScalarParameter(32, 64))
# relu = op('relu', conv)
def test_parametrization():
    print("test parametrize")
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
    pass


def test_set_params():
    pass


def test_instantiate():
    pass


if __name__ == "__main__":
    test_parametrization()
    # test_sample()
    # test_set_params()
    # test_instantiate()


# parametrized_test = Test(a=IntScalarParameter(min=10, max=20))
# parametrized_test.set_params(a=5, b=7)

# test = parametrized_test.instantiate()

# assert test.a == 7
# assert isinstance(test.a, int) ## ????
# # test.sample()

# test2_parametrized = Test2(a2=IntScalarParameter(2, 6))
# parametrized_test_hierarchical = Test(a=IntScalarParameter(32, 54), test2=test2)


# parametrized_test.cond(parametrized_test.a <= parametrized_test2.a2)


# parametrized_test_hierarchical.set_params(a=5)
# test2_parametrized.set_params(a2=6)
