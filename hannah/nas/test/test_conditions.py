from hannah.nas.core.parametrized import is_parametrized
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

    accelerator = (
        Accelerator(a_param, b_param).cond(a_param + b_param < 10).cond(a_param < 5)
    )

    print(accelerator.instantiate())
    accelerator.set_current(a=5)
    accelerator.set_current(a=12, b=10)


if __name__ == "__main__":
    test_condition()
