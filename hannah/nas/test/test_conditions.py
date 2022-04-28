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
    accelerator.set_current({'a': 6})
    accelerator_instance = accelerator.instantiate()
    try:
        accelerator.set_current({'a': 6, 'b': 6})
    except Exception:
        pass

    assert accelerator_instance.a == 6
    assert accelerator_instance.b == 0


if __name__ == "__main__":
    test_condition()
