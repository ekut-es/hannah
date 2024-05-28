from hannah.nas.functional_operators.operators import Relu, Conv2d
from hannah.nas.functional_operators.op import scope, search_space, Tensor
from hannah.nas.parameters.parameters import IntScalarParameter, CategoricalParameter


@search_space
def simple_search_space():
    input = Tensor(name='input',
                   shape=(1, 3, 32, 32),
                   axis=('N', 'C', 'H', 'W'))

    weight_0 = Tensor(name='weight', shape=(IntScalarParameter(min=8, max=64, name='out_channels'), 3, 1, 1), axis=("O", "I", "kH", "kW"))

    conv_0 = Conv2d(stride=CategoricalParameter(name='stride', choices=[1, 2]))(input, weight_0)
    relu_0 = Relu()(conv_0)

    weight_1 = Tensor(name='weight', shape=(IntScalarParameter(min=32, max=64, name='out_channels'), conv_0.shape()[1], 3, 3), axis=("O", "I", "kH", "kW"))
    conv_1 = Conv2d(stride=CategoricalParameter(name='stride', choices=[1, 2]))(relu_0, weight_1)
    relu_1 = Relu()(conv_1)
    return relu_1


def test_parameter_scoping():
    out = simple_search_space()
    params = out.parametrization()
    assert len(params) == 4
    assert 'simple_search_space_0.Conv2d_1.stride' in params
    assert 'simple_search_space_0.Conv2d_1.weight.out_channels' in params
    assert 'simple_search_space_0.Conv2d_0.stride' in params
    assert 'simple_search_space_0.Conv2d_1.weight.out_channels' in params


if __name__ == "__main__":
    test_parameter_scoping()