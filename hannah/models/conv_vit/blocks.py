from functools import partial
from torch.nn import functional as F

from hannah.nas.parameters.parameters import IntScalarParameter
from hannah.nas.expressions.arithmetic import Ceil
from hannah.nas.expressions.types import Int
from hannah.nas.functional_operators.op import scope

from hannah.models.conv_vit.operators import (
    conv2d, grouped_conv2d, batch_norm, relu, linear, add,
    max_pool, avg_pool, adaptive_avg_pooling, combine_pool,
    choice, dynamic_depth
)
from hannah.models.conv_vit.attention import attention2d, relu_lin_attention


@scope
def classifier_head(input, num_classes):
    out = adaptive_avg_pooling(input)
    out = linear(out, num_classes)

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
def pooling_pattern(input, kernel_size, stride):
    maxpool = partial(
        max_pool, kernel_size=kernel_size, stride=stride,
    )
    avgpool = partial(
        avg_pool, kernel_size=kernel_size, stride=stride,
    )
    combine = partial(
        combine_pool, kernel_size=kernel_size, stride=stride
    )

    out = choice(input, maxpool, avgpool, combine)

    return out


@scope
def conv_layer(input, out_channels, kernel_size, stride):
    out = conv2d(input, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
    out = batch_norm(out)
    out = relu(out)

    out = conv2d(out, out_channels=out_channels, kernel_size=1, stride=1)
    out = batch_norm(out)
    out = relu(out)

    return out


@scope
def inv_bottleneck(input, expand_ratio, out_channels, kernel_size, stride):
    in_channels = input.shape()[1]
    expanded_channels = Int(expand_ratio * in_channels)

    out = conv2d(input, expanded_channels, kernel_size=1, stride=1, padding=0)
    out = batch_norm(out)
    out = relu(out)

    out = grouped_conv2d(out, expanded_channels, kernel_size=kernel_size, stride=stride, groups=expanded_channels)
    out = batch_norm(out)
    out = relu(out)

    out = conv2d(out, out_channels, kernel_size=1, stride=1, padding=0)
    out = batch_norm(out)
    out = relu(out)

    res = residual(input, out.shape())
    out = add(out, res)

    return out


@scope
def bottleneck(input, reduce_ratio, out_channels, kernel_size, stride):
    in_channels = input.shape()[1]
    reduced_channels = Int(in_channels / reduce_ratio)

    out = conv2d(input, reduced_channels, kernel_size=1, stride=1, padding=0)
    out = batch_norm(out)
    out = relu(out)

    out = grouped_conv2d(out, reduced_channels, kernel_size=kernel_size, stride=stride, groups=reduced_channels)
    out = batch_norm(out)
    out = relu(out)

    out = conv2d(out, out_channels, kernel_size=1, stride=1, padding=0)
    out = batch_norm(out)
    out = relu(out)

    res = residual(input, out.shape())
    out = add(out, res)

    return out


@scope
def conv_residual(input, out_channels, kernel_size, stride):
    out = conv2d(input, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
    out = batch_norm(out)
    out = relu(out)

    out = conv2d(out, out_channels=out_channels, kernel_size=1, stride=1)
    out = batch_norm(out)
    out = relu(out)

    res = residual(input, out.shape())
    out = add(out, res)

    return out


@scope
def conv_pattern(input, out_channels, kernel_size, stride, channel_ratio):
    conv = partial(
        conv_layer,
        out_channels=out_channels, kernel_size=kernel_size, stride=stride,
    )
    conv_res = partial(
        conv_residual,
        out_channels=out_channels, kernel_size=kernel_size, stride=stride,
    )
    conv_bot = partial(
        bottleneck,
        reduce_ratio=channel_ratio,
        out_channels=out_channels, kernel_size=kernel_size, stride=stride,
    )
    conv_inv_bot = partial(
        inv_bottleneck,
        expand_ratio=channel_ratio,
        out_channels=out_channels, kernel_size=kernel_size, stride=stride
    )

    out = choice(input, conv, conv_res, conv_bot, conv_inv_bot)

    return out


# @scope
def attention_layer(input, num_heads, d_model, out_channels, use_lin_attn=False):
    if use_lin_attn:
        out = relu_lin_attention(input, num_heads, d_model)
    else:
        out = attention2d(input, num_heads, d_model)
    out = conv2d(out, out_channels, kernel_size=1, stride=1, padding=0)
    out = batch_norm(out)
    out = relu(out)

    res = residual(input, out.shape())
    out = add(out, res)

    return out


@scope
def feed_forward(input, out_channels):
    out = conv2d(input, out_channels, kernel_size=1, stride=1, padding=0)
    out = batch_norm(out)
    out = relu(out)

    res = residual(input, out.shape())
    out = add(out, res)

    return out


@scope
def vit_pattern(input, out_channels, num_heads, d_model, use_lin_attn=False):
    out = attention_layer(
        input, num_heads, d_model, out_channels, use_lin_attn=use_lin_attn
    )
    ff = partial(feed_forward, out_channels=out_channels)
    out = choice(out, ff)

    return out


@scope
def cnn_vit(
    input,
    out_channels, kernel_size, stride, channel_ratio,
    num_heads, d_model, use_lin_attn=False
):
    out = conv_pattern(input, out_channels, kernel_size, stride, channel_ratio)
    out = vit_pattern(out, out_channels, num_heads, d_model, use_lin_attn=use_lin_attn)

    return out


@scope
def pool_vit(
    input,
    out_channels, kernel_size, stride,
    num_heads, d_model, use_lin_attn=False
):
    out = pooling_pattern(input, kernel_size=kernel_size, stride=stride)
    out = vit_pattern(out, out_channels, num_heads, d_model, use_lin_attn=use_lin_attn)

    return out


@scope
def stem(input, out_channels, kernel_size, stride, channel_ratio, pool_size):
    out = conv2d(input, out_channels, kernel_size, stride)
    out = batch_norm(out)
    out = relu(out)

    out = conv_pattern(out, out_channels, kernel_size, stride, channel_ratio)
    out = pooling_pattern(out, kernel_size=pool_size, stride=2)

    return out


@scope
def pattern(
    input,
    out_channels, kernel_size, stride, channel_ratio,
    num_heads, d_model, use_lin_attn=False
):
    cvit = partial(
        cnn_vit,
        out_channels=out_channels,
        kernel_size=kernel_size, stride=stride, channel_ratio=channel_ratio,
        num_heads=num_heads, d_model=d_model, use_lin_attn=use_lin_attn
    )
    pvit = partial(
        pool_vit,
        out_channels=out_channels,
        kernel_size=kernel_size, stride=stride,
        num_heads=num_heads, d_model=d_model, use_lin_attn=use_lin_attn
    )
    out = choice(input, cvit, pvit)

    return out


@scope
def block(
    input, depth,
    out_channels, kernel_size, stride, channel_ratio,
    num_heads, d_model, use_lin_attn
):
    assert isinstance(depth, IntScalarParameter), "block depth must be of type IntScalarParameter"
    out = input
    exits = []
    for i in range(depth.max+1):
        out = pattern(
            out,
            out_channels=out_channels.new(),
            kernel_size=kernel_size.new(),
            stride=stride.new() if i == 0 else 1,
            channel_ratio=channel_ratio.new(),
            num_heads=num_heads.new(),
            d_model=d_model.new(),
            use_lin_attn=use_lin_attn,
        )
        exits.append(out)

    out = dynamic_depth(*exits, switch=depth)
    res = residual(input, out.shape())
    out = add(out, res)

    return out
