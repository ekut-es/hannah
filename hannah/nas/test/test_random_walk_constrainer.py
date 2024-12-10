from hannah.nas.functional_operators.op import Tensor, scope, search_space
from hannah.nas.functional_operators.utils.visit import get_active_parameters
from hannah.nas.constraints.random_walk import RandomWalkConstraintSolver
from hannah.models.embedded_vision_net.operators import adaptive_avg_pooling, add, conv_relu, dynamic_depth, linear
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter


@scope
def conv_block(input, stride, kernel_size, out_channels):
    out = conv_relu(input, stride=stride, kernel_size=kernel_size, out_channels=out_channels)
    out = conv_relu(out, stride=1, kernel_size=kernel_size, out_channels=out_channels)
    return out


@scope
def parallel_blocks(input, stride, kernel_size, out_channels):
    out_0 = conv_block(input, stride=stride, kernel_size=kernel_size, out_channels=out_channels)
    out_1 = conv_block(input, stride=stride, kernel_size=kernel_size, out_channels=out_channels)
    out = add(out_0, out_1)
    return out


@scope
def classifier_head(input, num_classes):
    out = adaptive_avg_pooling(input)
    out = linear(out, num_classes)
    return out


@search_space
def space(input):
    out_channels = IntScalarParameter(4, 64, 4, name="out_channels")
    kernel_size = CategoricalParameter([1, 3, 5, 7], name="kernel_size")
    stride = IntScalarParameter(1, 2, name="stride")
    depth = IntScalarParameter(0, 2, name="depth")
    out = conv_relu(input, out_channels=out_channels.new(), kernel_size=kernel_size.new(), stride=stride.new())
    block_0 = parallel_blocks(out, out_channels=out_channels.new(), kernel_size=kernel_size.new(), stride=stride.new())
    block_1 = parallel_blocks(block_0, out_channels=out_channels.new(), kernel_size=kernel_size.new(), stride=stride.new())
    block_2 = parallel_blocks(block_1, out_channels=out_channels.new(), kernel_size=kernel_size.new(), stride=stride.new())
    out = dynamic_depth(block_0, block_1, block_2, switch=depth)
    out = classifier_head(out, num_classes=10)
    return out


def test_get_active_params():
    input = Tensor(name='input', shape=(1, 3, 32, 32), axis=('N', 'C', 'H', 'W'))
    out = space(input)
    active_params = list(get_active_parameters(out).keys())
    assert len(active_params) == 7
    for p in active_params:
        assert "parallel_blocks_1" not in p and "parallel_blocks_2" not in p
    out.parametrization()['space_0.ChoiceOp_0.depth'].set_current(1)
    active_params = get_active_parameters(out)
    assert len(active_params) == 10
    for p in active_params:
        assert "parallel_blocks_2" not in p
    out.parametrization()['space_0.ChoiceOp_0.depth'].set_current(2)
    active_params = get_active_parameters(out)
    assert len(active_params) == 13


def test_right_direction():
    rw_solver = RandomWalkConstraintSolver()
    dir = "<"
    assert not rw_solver.right_direction(10, 15, dir)
    assert rw_solver.right_direction(26, 17, dir)

    dir = ">"
    assert rw_solver.right_direction(10, 15, dir)
    assert not rw_solver.right_direction(26, 17, dir)

    dir = "<="
    assert not rw_solver.right_direction(10, 15, dir)
    assert rw_solver.right_direction(26, 17, dir)

    dir = ">="
    assert rw_solver.right_direction(10, 15, dir)
    assert not rw_solver.right_direction(26, 17, dir)


if __name__ == '__main__':
    test_get_active_params()
    test_right_direction()

