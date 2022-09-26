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

    assert (one & one).evaluate() == True
    assert (one & zero).evaluate() == False


def test_le_expr():
    one = DefaultInt(1)
    zero = DefaultInt(0)

    assert (one < one).evaluate() == False
    assert (one < zero).evaluate() == False
    assert (zero < one).evaluate() == True
    assert (zero < zero).evaluate() == False


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
