from functools import partial
from hannah.models.embedded_vision_net.expressions import expr_product
from hannah.nas.expressions.arithmetic import Ceil
from hannah.nas.expressions.types import Int
from hannah.nas.functional_operators.op import scope
from hannah.models.embedded_vision_net.operators import adaptive_avg_pooling, add, conv2d, conv_relu, depthwise_conv2d, dynamic_depth, grouped_conv2d, interleave_channels, pointwise_conv2d, linear, relu, batch_norm, choice, identity, max_pool, avg_pool
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter


@scope
def pooling(input, kernel_size, stride):
    avgp = partial(avg_pool, kernel_size=kernel_size, stride=stride)
    maxp = partial(max_pool, kernel_size=kernel_size, stride=stride)
    pool_choice = CategoricalParameter([0, 1], name='pool_mode')
    out = choice(input, avgp, maxp, switch=pool_choice)
    return out


@scope
def grouped_pointwise(input, out_channels):
    pw_k = grouped_conv2d(input, out_channels, kernel_size=1, stride=1)
    out = interleave_channels(pw_k, step_size=pw_k.groups)
    pw_l = grouped_conv2d(out, out_channels, kernel_size=1, stride=1, groups=pw_k.groups)
    out = add(pw_l, pw_k)
    return out


@scope
def expansion(input, expanded_channels):
    pw = partial(pointwise_conv2d, out_channels=expanded_channels)
    grouped_pw = partial(grouped_pointwise, out_channels=expanded_channels)
    return choice(input, pw, grouped_pw)
    # return pointwise_conv2d(input, out_channels=expanded_channels)


@scope
def spatial_correlation(input, out_channels, kernel_size, stride=1):
    return depthwise_conv2d(input, out_channels=out_channels, kernel_size=kernel_size, stride=stride)


@scope
def reduction(input, out_channels):
    pw = partial(pointwise_conv2d, out_channels=out_channels)
    grouped_pw = partial(grouped_pointwise, out_channels=out_channels)
    return choice(input, pw, grouped_pw)
    # return pointwise_conv2d(input, out_channels=out_channels)


@scope
def reduce_expand(input, out_channels, reduce_ratio, kernel_size, stride):
    in_channels = input.shape()[1]
    reduced_channels = Int(in_channels / reduce_ratio)

    out = reduction(input, reduced_channels)
    # out.add_param('reduce_ratio', reduce_ratio)
    out = batch_norm(out)
    out = relu(out)
    out = conv2d(out, reduced_channels, kernel_size, stride)
    out = batch_norm(out)
    out = relu(out)
    out = expansion(out, out_channels)
    out = batch_norm(out)
    out = relu(out)
    return out


@scope
def expand_reduce(input, out_channels, expand_ratio, kernel_size, stride):
    in_channels = input.shape()[1]
    expanded_channels = Int(expand_ratio * in_channels)
    out = expansion(input, expanded_channels)
    out = batch_norm(out)
    out = relu(out)
    out = spatial_correlation(out, kernel_size=kernel_size, stride=stride, out_channels=expanded_channels)
    out = batch_norm(out)
    out = relu(out)
    out = reduction(out, out_channels)
    out = batch_norm(out)
    out = relu(out)
    return out


# FIXME: integrate this into reduce_expand?
@scope
def sandglass_block(input, out_channels, reduce_ratio, kernel_size, stride):
    in_channels = input.shape()[1]
    reduced_channels = Int(in_channels / reduce_ratio)
    out = depthwise_conv2d(input, out_channels=in_channels, kernel_size=kernel_size, stride=1)
    out = batch_norm(out)
    out = relu(out)
    out = reduction(out, out_channels=reduced_channels)
    out = expansion(out, expanded_channels=out_channels)
    out = relu(out)
    out = batch_norm(out)
    out = depthwise_conv2d(out, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
    return out


@scope
def pattern(input, stride, out_channels, kernel_size, expand_ratio, reduce_ratio):
    convolution = partial(conv_relu, stride=stride, kernel_size=kernel_size, out_channels=out_channels)
    red_exp = partial(reduce_expand, out_channels=out_channels, reduce_ratio=reduce_ratio, kernel_size=kernel_size, stride=stride)
    exp_red = partial(expand_reduce, out_channels=out_channels, expand_ratio=expand_ratio, kernel_size=kernel_size, stride=stride)
    pool = partial(pooling, kernel_size=kernel_size, stride=stride)
    sandglass = partial(sandglass_block, out_channels=out_channels, reduce_ratio=reduce_ratio, kernel_size=kernel_size, stride=stride)

    out = choice(input, convolution, exp_red, red_exp, pool, sandglass)
    return out


@scope
def residual(input, main_branch_output_shape):
    input_shape = input.shape()
    in_fmap = input_shape[2]
    out_channels = main_branch_output_shape[1]
    out_fmap = main_branch_output_shape[2]
    stride = Int(Ceil(in_fmap / out_fmap))

    out = conv2d(input, out_channels=out_channels, kernel_size=1, stride=stride, padding=0)
    out = batch_norm(out)
    out = relu(out)
    return out


@scope
def block(input, depth, stride, out_channels, kernel_size, expand_ratio, reduce_ratio):
    assert isinstance(depth, IntScalarParameter), "block depth must be of type IntScalarParameter"
    out = input
    exits = []
    for i in range(depth.max+1):
        out = pattern(out,
                      stride=stride.new() if i == 0 else 1,
                      out_channels=out_channels.new(),
                      kernel_size=kernel_size.new(),
                      expand_ratio=expand_ratio.new(),
                      reduce_ratio=reduce_ratio.new())
        exits.append(out)

    out = dynamic_depth(*exits, switch=depth)
    res = residual(input, out.shape())
    out = add(out, res)

    return out


@scope
def cwm_block(input, depth, stride, channel_width_multiplier, kernel_size, expand_ratio, reduce_ratio):
    assert isinstance(depth, IntScalarParameter), "block depth must be of type IntScalarParameter"
    out = input
    exits = []
    in_channels = out.shape()[1]
    for i in range(depth.max+1):
        out = pattern(out,
                      stride=stride.new() if i == 0 else 1,
                      out_channels=Int(channel_width_multiplier * in_channels),
                      kernel_size=kernel_size.new(),
                      expand_ratio=expand_ratio.new(),
                      reduce_ratio=reduce_ratio.new())
        exits.append(out)

    out = dynamic_depth(*exits, switch=depth)
    res = residual(input, out.shape())
    out = add(out, res)

    return out


@scope
def stem(input, kernel_size, stride, out_channels):
    out = conv2d(input, out_channels, kernel_size, stride)
    out = batch_norm(out)
    out = relu(out)
    return out


@scope
def classifier_head(input, num_classes):
    out = adaptive_avg_pooling(input)
    out = linear(out, num_classes)
    return out
