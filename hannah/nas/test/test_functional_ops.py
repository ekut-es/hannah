from hannah.nas.expressions.arithmetic import Ceil
from hannah.nas.expressions.types import Int
from hannah.nas.functional_operators.op import ChoiceOp, OptionalOp, Tensor, scope, nodes_in_scope, search_space
from hannah.nas.functional_operators.operators import Add, Conv2d, Relu, Identity
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter
from functools import partial


def conv_relu(input, out_channels, kernel_size, stride):
    """ Example for a functional block containing conv and relu
    """
    out = conv2d(input, out_channels=out_channels, stride=stride, kernel_size=kernel_size)
    out = relu(out)
    return out


def add(input, other):
    return Add()(input, other)


def conv2d(input, out_channels, kernel_size=1, stride=1, dilation=1):
    in_channels = input.shape()[1]
    weight = Tensor(name='weight',
                    shape=(out_channels, in_channels, kernel_size, kernel_size),
                    axis=('O', 'I', 'kH', 'kW'))

    conv = Conv2d(stride, dilation)(input, weight)
    return conv


def relu(input):
    return Relu()(input)


@scope
def double_conv(input):
    kernel_size = CategoricalParameter([1, 3, 5], name='kernel_size')
    out_channels = IntScalarParameter(min=4, max=64, name='out_channels')
    stride = CategoricalParameter([1, 2], name='stride')

    net = conv_relu(input, out_channels=out_channels.new(), kernel_size=kernel_size.new(), stride=stride.new())
    net = conv_relu(net, out_channels=out_channels.new(), kernel_size=kernel_size.new(), stride=stride.new())
    return net


@scope
def residual(input, out_channels, in_size, out_size):
    stride = Int(Ceil(in_size / out_size))
    out = conv_relu(input, out_channels, kernel_size=1, stride=stride)
    return out


@scope
def res_block(input):
    kernel_size = CategoricalParameter([1, 3, 5], name='kernel_size')
    out_channels = IntScalarParameter(min=4, max=64, name='out_channels')
    stride = CategoricalParameter([1, 2], name='stride')

    net = conv_relu(input, out_channels=out_channels.new(), kernel_size=kernel_size.new(), stride=2)
    block_channels = out_channels.new()
    main_branch = conv_relu(net, out_channels=block_channels, kernel_size=kernel_size, stride=stride)
    res = residual(net, out_channels=block_channels, in_size=net.shape()[2], out_size=main_branch.shape()[2])

    net = add(main_branch, res)
    return net


def test_functional_ops():
    @search_space
    def test_space():
        kernel_size = CategoricalParameter([1, 3, 5], name='kernel_size')
        out_channels = IntScalarParameter(min=4, max=64, name='out_channels')
        stride = CategoricalParameter([1, 2], name='stride')

        input = Tensor(name='input', shape=(1, 3, 32, 32), axis=('N', 'C', 'H', 'W'))
        weight = Tensor(name='weight',
                        shape=(out_channels, 3, kernel_size, kernel_size),
                        axis=('O', 'I', 'kH', 'kW'))
        net = Conv2d(stride=stride)(input, weight)
        net = Relu()(net)
        return net
    net = test_space()


def test_functional_ops_chained():
    @search_space
    def test_space():
        kernel_size = CategoricalParameter([1, 3, 5], name='kernel_size')
        out_channels = IntScalarParameter(min=4, max=64, name='out_channels')
        stride = CategoricalParameter([1, 2], name='stride')

        input = Tensor(name='input', shape=(1, 3, 32, 32), axis=('N', 'C', 'H', 'W'))
        weight0 = Tensor(name='weight',
                         shape=(out_channels, 3, kernel_size, kernel_size),
                         axis=('O', 'I', 'kH', 'kW'))
        net = Conv2d(stride=stride)(input, weight0)
        net = Relu()(net)

        weight1 = Tensor(name='weight',
                         shape=(out_channels, 3, kernel_size, kernel_size),
                         axis=('O', 'I', 'kH', 'kW'))

        net = Conv2d(stride=stride.new())(net, weight1)
        net = Relu()(net)
        return net
    net = test_space()

    # print(net.parametrization(flatten=True))


def test_shape_propagation():
    @search_space
    def test_space():
        kernel_size = CategoricalParameter([1, 3, 5], name='kernel_size')
        out_channels = IntScalarParameter(min=4, max=64, name='out_channels')
        stride = CategoricalParameter([1, 2], name='stride')
        stride.current_value = 1

        input = Tensor(name='input', shape=(1, 3, 32, 32), axis=('N', 'C', 'H', 'W'))
        net = conv2d(input, out_channels=out_channels.new(), stride=stride.new(), kernel_size=kernel_size.new())
        net = relu(net)

        net = conv2d(net, out_channels=out_channels.new(), stride=stride.new(), kernel_size=kernel_size.new())
        net = relu(net)

        stride.current_value = 2
        out_channels.current_value = 48

        net = conv2d(net, out_channels=out_channels.new(), stride=stride.new(), kernel_size=kernel_size.new())
        net = relu(net)
        return net
    net = test_space()

    shape = net.shape()
    assert shape[0] == 1
    assert shape[1].evaluate() == 48
    assert shape[2].evaluate() == 16
    assert shape[3].evaluate() == 16


def test_blocks():
    @search_space
    def test_space():
        kernel_size = CategoricalParameter([1, 3, 5], name='kernel_size')
        out_channels = IntScalarParameter(min=4, max=64, name='out_channels')

        input = Tensor(name='input', shape=(1, 3, 32, 32), axis=('N', 'C', 'H', 'W'))
        net = conv_relu(input, out_channels=out_channels.new(), kernel_size=kernel_size.new(), stride=1)

        net = conv_relu(net, out_channels=32, kernel_size=kernel_size.new(), stride=2)
        return net
    net = test_space()


def test_operators():
    # TODO: Make op registry and test ops
    pass


def test_multibranches():
    @search_space
    def test_space():
        kernel_size = CategoricalParameter([1, 3, 5], name='kernel_size')
        out_channels = IntScalarParameter(min=4, max=64, name='out_channels')

        input = Tensor(name='input', shape=(1, 3, 32, 32), axis=('N', 'C', 'H', 'W'))
        net = conv_relu(input, out_channels=out_channels.new(), kernel_size=kernel_size.new(), stride=2)

        stride = CategoricalParameter([1, 2], name='stride')
        stride.current_value = 1
        block_channels = out_channels.new()
        block_channels.current_value = 48
        main_branch = conv_relu(net, out_channels=block_channels, kernel_size=kernel_size, stride=stride)
        res = residual(net, out_channels=block_channels, in_size=net.shape()[2], out_size=main_branch.shape()[2])
        net = add(main_branch, res)
        stride.current_value = 2
        return net
    net = test_space()


def test_scoping():

    input = Tensor(name='input', shape=(1, 3, 32, 32), axis=('N', 'C', 'H', 'W'))

    @search_space
    def test_space(input):
        net = double_conv(input)
        net = double_conv(net)
        net = double_conv(net)
        return net
    net = test_space(input)

    test_scope = 'test_space_0.double_conv_1.Conv2d_0'
    scopes = []
    for node in nodes_in_scope(net, [input]):
        scopes.append(node.id)
    assert test_scope in scopes


def test_parametrization():
    # input = Tensor(name='input', shape=(1, 3, 32, 32), axis=('N', 'C', 'H', 'W'))
    # net = res_block(input)
    pass


def test_choice():
    @scope
    def choice_block():
        kernel_size = CategoricalParameter([1, 3, 5], name='kernel_size')
        out_channels = IntScalarParameter(min=4, max=64, name='out_channels')
        stride = CategoricalParameter([1, 2], name='stride')

        input = Tensor(name='input', shape=(1, 3, 32, 32), axis=('N', 'C', 'H', 'W'))
        conv = conv2d(input, out_channels=out_channels.new(), stride=stride.new(), kernel_size=kernel_size.new())
        identity = Identity()
        relu = Relu()
        optional_conv = partial(conv2d, out_channels=out_channels.new(), stride=stride.new(), kernel_size=kernel_size.new())
        net = ChoiceOp(identity, relu, optional_conv)(conv)
        net = conv2d(net, out_channels=out_channels.new(), stride=stride.new(), kernel_size=kernel_size.new())
        return net

    @search_space
    def test_space():
        return choice_block()

    net = test_space()
    assert 'test_space_0.choice_block_0.ChoiceOp_0.choice' in net.parametrization(flatten=True)
    net.parametrization(flatten=True)['test_space_0.choice_block_0.ChoiceOp_0.choice'].set_current(0)
    net.parametrization(flatten=True)['test_space_0.choice_block_0.ChoiceOp_0.choice'].set_current(1)
    net.parametrization(flatten=True)['test_space_0.choice_block_0.ChoiceOp_0.choice'].set_current(2)


def test_optional_op():
    @scope
    def optional_block():
        kernel_size = CategoricalParameter([1, 3, 5], name='kernel_size')
        out_channels = IntScalarParameter(min=4, max=64, name='out_channels')
        stride = CategoricalParameter([1, 2], name='stride')

        input = Tensor(name='input', shape=(1, 3, 32, 32), axis=('N', 'C', 'H', 'W'))
        net = conv2d(input, out_channels=out_channels.new(), stride=stride.new(), kernel_size=kernel_size.new())
        net = OptionalOp(Relu())(net)
        net = conv2d(net, out_channels=out_channels.new(), stride=stride.new(), kernel_size=kernel_size.new())
        return net

    @search_space
    def test_space():
        return optional_block()

    net = test_space()


def test_dynamic_depth():
    @scope
    def dynamic_depth_block(input, depth):
        kernel_size = CategoricalParameter([1, 3, 5], name='kernel_size')
        out_channels = IntScalarParameter(min=4, max=64, name='out_channels')

        exit0 = conv_relu(input, out_channels.new(), kernel_size.new(), 2)
        exit1 = conv_relu(exit0, out_channels.new(), kernel_size.new(), 2)
        exit2 = conv_relu(exit1, out_channels.new(), kernel_size.new(), 2)

        net = ChoiceOp(exit0, exit1, exit2, switch=depth)
        net = conv_relu(net, 32, 1, 1)

        return net

    @search_space
    def test_space():
        input = Tensor(name='input', shape=(1, 3, 32, 32), axis=('N', 'C', 'H', 'W'))
        depth_choice = IntScalarParameter(min=0, max=2, name="depth_choice")
        depth_choice.current_value = 2

        net = dynamic_depth_block(input, depth_choice)
        return net

    net = test_space()
    # net = conv_relu(net, 32, 1, 1)


if __name__ == '__main__':
    test_functional_ops()
    test_functional_ops_chained()
    test_shape_propagation()
    test_blocks()
    test_operators()
    test_multibranches()
    test_scoping()
    # test_parametrization()
    test_choice()
    test_optional_op()
    test_dynamic_depth()
    # test_visualization()
