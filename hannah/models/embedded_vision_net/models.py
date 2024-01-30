from functools import partial
from hannah.models.embedded_vision_net.expressions import expr_product, extract_macs_recursive, extract_weights_recursive
from hannah.nas.expressions.arithmetic import Ceil
from hannah.nas.expressions.logic import And
from hannah.nas.expressions.types import Int
from hannah.nas.expressions.utils import extract_parameter_from_expression
from hannah.nas.functional_operators.executor import BasicExecutor
from hannah.nas.functional_operators.op import Tensor, get_nodes, scope
from hannah.models.embedded_vision_net.operators import adaptive_avg_pooling, add, conv2d, conv_relu, depthwise_conv2d, dynamic_depth, pointwise_conv2d, linear, relu, batch_norm, choice, identity
# from hannah.nas.functional_operators.visualizer import Visualizer
from hannah.nas.parameters.parameters import CategoricalParameter, FloatScalarParameter, IntScalarParameter
from hannah.models.embedded_vision_net.blocks import block, cwm_block, classifier_head, stem
from hannah.nas.parameters.parametrize import set_parametrization


def backbone(input, num_classes=10, max_channels=512):
    out_channels = IntScalarParameter(16, max_channels, step_size=8, name='out_channels')
    kernel_size = CategoricalParameter([3, 5, 7, 9], name='kernel_size')
    stride = CategoricalParameter([1, 2], name='stride')
    expand_ratio = IntScalarParameter(1, 6, name='expand_ratio')
    reduce_ratio = IntScalarParameter(2, 4, name='reduce_ratio')
    depth = IntScalarParameter(0, 2, name='depth')

    num_blocks = IntScalarParameter(0, 9, name='num_blocks')
    exits = []

    stem_kernel_size = CategoricalParameter([3, 5], name="kernel_size")
    stem_channels = IntScalarParameter(min=16, max=64, step_size=4, name="out_channels")
    out = stem(input, stem_kernel_size, stride.new(), stem_channels)
    for i in range(num_blocks.max+1):
        out = block(out, depth=depth.new(), stride=stride.new(), out_channels=out_channels.new(), kernel_size=kernel_size.new(),
                    expand_ratio=expand_ratio.new(), reduce_ratio=reduce_ratio.new())
        exits.append(out)

    out = dynamic_depth(*exits, switch=num_blocks)

    output_fmap = out.shape()[2]

    out = classifier_head(out, num_classes=num_classes)

    stride_params = [v for k, v in out.parametrization(flatten=True).items() if k.split('.')[-1] == 'stride']
    out.cond(output_fmap > 1, allowed_params=stride_params)

    return out


def search_space(name, input, num_classes: int, max_channels=512, constraints: list[dict] = []):
    arch = backbone(input, num_classes, max_channels)
    # arch.weights = extract_weights_recursive(arch)
    # arch.macs = extract_weights_recursive(arch)
    # arch.cond(And(arch.macs < 128000000, arch.weights < 550000))
    for con in constraints:
        if con.name == "weights":
            arch.weights = extract_weights_recursive(arch)
            weight_params = extract_parameter_from_expression(arch.weights)
            weight_params = [p for p in weight_params if 'stride' not in p.name and 'groups' not in p.name]
            if "lower" in con and "upper" in con:
                upper = arch.weights < con.upper
                lower = arch.weights > con.lower
                arch.cond(And(lower, upper), weight_params)
            elif "upper" in con:
                arch.cond(arch.weights < con.upper, weight_params)
            elif "lower" in con:
                arch.cond(arch.weights > con.lower, weight_params)
        elif con.name == "macs":
            arch.macs = extract_macs_recursive(arch)
            mac_params = extract_parameter_from_expression(arch.macs)
            mac_params = [p for p in mac_params if 'stride' not in p.name and 'groups' not in p.name]
            if "lower" in con and "upper" in con:
                upper = arch.macs < con.upper
                lower = arch.macs > con.lower
                arch.cond(And(lower, upper), mac_params)
            elif "upper" in con:
                arch.cond(arch.macs < con.upper, mac_params)
            elif "lower" in con:
                arch.cond(arch.macs > con.lower, mac_params)
        else:
            raise NotImplementedError(f"Constraint {con.name} not implemented")
    return arch


def search_space_with_param_init(name, input, num_classes, max_channels, constraints, param_path, task_name, index):
    import pandas as pd
    from pathlib import Path

    space = search_space(name, input, num_classes, max_channels, constraints)
    params = pd.read_pickle(Path(param_path))
    # params = pd.read_pickle(Path("~/projects/hannah/experiments/embedded_vision_net_ri/parameters.pkl"))
    parameters = params[task_name][index]
    set_parametrization(parameters, space.parametrization(flatten=True))
    return space


def search_space_cwm(name,  input, num_classes=10):
    channel_width_multiplier = CategoricalParameter([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], name="channel_width_multiplier")
    kernel_size = CategoricalParameter([3, 5, 7, 9], name='kernel_size')
    stride = CategoricalParameter([1, 2], name='stride')
    expand_ratio = IntScalarParameter(2, 6, name='expand_ratio')
    reduce_ratio = IntScalarParameter(3, 6, name='reduce_ratio')
    depth = IntScalarParameter(0, 2, name='depth')
    num_blocks = IntScalarParameter(0, 5, name='num_blocks')
    exits = []

    stem_kernel_size = CategoricalParameter([3, 5], name="kernel_size")
    stem_channels = IntScalarParameter(min=16, max=32, step_size=4, name="out_channels")
    out = stem(input, stem_kernel_size, stride.new(), stem_channels)
    for i in range(num_blocks.max+1):
        out = cwm_block(out,
                        depth=depth.new(),
                        stride=stride.new(),
                        channel_width_multiplier=channel_width_multiplier.new(),
                        kernel_size=kernel_size.new(),
                        expand_ratio=expand_ratio.new(),
                        reduce_ratio=reduce_ratio.new())
        exits.append(out)

    out = dynamic_depth(*exits, switch=num_blocks)
    out = classifier_head(out, num_classes=num_classes)

    strides = [v for k, v in out.parametrization(flatten=True).items() if k.split('.')[-1] == 'stride']
    total_stride = expr_product(strides)
    out.cond(input.shape()[2] / total_stride > 1, )

    multipliers = [v for k, v in out.parametrization(flatten=True).items() if k.split('.')[-1] == 'channel_width_multiplier']
    max_multiplication = expr_product(multipliers)
    out.cond(max_multiplication < 4)
    return out


def model(name, param_path, task_name, index, input_shape, labels):
    import pandas as pd
    from pathlib import Path

    params = pd.read_pickle(Path(param_path))
    input = Tensor(name='input', shape=input_shape, axis=("N", "C", "H", "W"))
    space = search_space(name=name, input=input, num_classes=labels)
    # params = pd.read_pickle(Path("~/projects/hannah/experiments/embedded_vision_net_ri/parameters.pkl"))
    parameters = params[task_name][index]
    set_parametrization(parameters, space.parametrization(flatten=True))
    mod = BasicExecutor(space)
    mod.initialize()
    return mod

