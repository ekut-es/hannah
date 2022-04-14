from hannah.nas.expressions.arithmetic import Add
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


if __name__ == "__main__":
    test_print()
