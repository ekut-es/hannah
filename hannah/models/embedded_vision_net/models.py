from functools import partial
from hannah.models.embedded_vision_net.expressions import expr_product
from hannah.nas.expressions.arithmetic import Ceil
from hannah.nas.expressions.types import Int
from hannah.nas.functional_operators.executor import BasicExecutor
from hannah.nas.functional_operators.op import Tensor, get_nodes, scope
from hannah.models.embedded_vision_net.operators import adaptive_avg_pooling, add, conv2d, conv_relu, depthwise_conv2d, dynamic_depth, pointwise_conv2d, linear, relu, batch_norm, choice, identity
# from hannah.nas.functional_operators.visualizer import Visualizer
from hannah.nas.parameters.parameters import CategoricalParameter, FloatScalarParameter, IntScalarParameter
from hannah.models.embedded_vision_net.blocks import block, cwm_block, classifier_head, stem


def search_space(name,  input, num_classes=10):
    out_channels = IntScalarParameter(32, 64, step_size=4, name='out_channels')
    kernel_size = CategoricalParameter([3, 5, 7, 9], name='kernel_size')
    stride = CategoricalParameter([1, 2], name='stride')
    expand_ratio = IntScalarParameter(1, 3, name='expand_ratio')
    reduce_ratio = IntScalarParameter(2, 4, name='reduce_ratio')

    depth = IntScalarParameter(0, 2, name='depth')

    num_blocks = IntScalarParameter(0, 6, name='num_blocks')
    exits = []

    out = input
    for i in range(num_blocks.max+1):
        out = block(out, depth=depth.new(), stride=stride.new(), out_channels=out_channels.new(), kernel_size=kernel_size.new(),
                    expand_ratio=expand_ratio.new(), reduce_ratio=reduce_ratio.new())
        exits.append(out)

    out = dynamic_depth(*exits, switch=num_blocks)
    out = classifier_head(out, num_classes=num_classes)

    strides = [v for k, v in out.parametrization(flatten=True).items() if k.split('.')[-1] == 'stride']
    total_stride = expr_product(strides)
    out.cond(input.shape()[2] / total_stride > 1)

    return out


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
    out.cond(input.shape()[2] / total_stride > 1)

    multipliers = [v for k, v in out.parametrization(flatten=True).items() if k.split('.')[-1] == 'channel_width_multiplier']
    max_multiplication = expr_product(multipliers)
    out.cond(max_multiplication < 4)
    return out
