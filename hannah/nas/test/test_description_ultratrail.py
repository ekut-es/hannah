from hannah.nas.hardware_description.device import Ultratrail
from hannah.nas.parameters import IntScalarParameter


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
