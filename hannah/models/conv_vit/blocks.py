from functools import partial
from torch.nn import functional as F

from hannah.nas.parameters.parameters import IntScalarParameter
from hannah.nas.expressions.arithmetic import Ceil
from hannah.nas.expressions.types import Int
from hannah.nas.functional_operators.op import scope
from hannah.nas.functional_operators.lazy import lazy

from hannah.models.conv_vit.operators import (
    conv2d, batch_norm, relu, linear, add,
    max_pool, adaptive_avg_pooling,
    choice, dynamic_depth, grouped_conv2d
)
from hannah.models.conv_vit.attention import attention2d


@scope
def stem(input, kernel_size, stride, out_channels):
    out = conv2d(input, out_channels, kernel_size, stride)
    out = batch_norm(out)
    out = relu(out)
    out = max_pool(out, kernel_size=3, stride=2)

    return out


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
def conv_layer(input, out_channels, kernel_size, stride):
    out = conv2d(input, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
    out = relu(out)

    out = conv2d(out, out_channels=out_channels, kernel_size=1, stride=1)
    out = batch_norm(out)
    out = relu(out)

    return out


@scope
def embedding(input, expand_ratio, out_channels, kernel_size, stride):
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
def attention_layer(input, num_heads, d_model, out_channels):
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
def transformer_cell(input, expand_ratio, out_channels, kernel_size, stride, num_heads, d_model):
    out = embedding(input, expand_ratio, out_channels, kernel_size, stride)
    out = attention_layer(out, num_heads, d_model, out_channels)
    out = feed_forward(out, out_channels)

    return out


@scope
def attention_cell(input, out_channels, kernel_size, stride, num_heads, d_model):
    out = conv_layer(input, out_channels, kernel_size, stride)
    out = attention_layer(out, num_heads, d_model, out_channels)

    return out


@scope
def pattern(input, expand_ratio, kernel_size, stride, num_heads, d_model, out_channels):
    attn = partial(
        attention_cell,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        num_heads=num_heads,
        d_model=d_model
    )
    trf = partial(
        transformer_cell,
        expand_ratio=expand_ratio,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        num_heads=num_heads,
        d_model=d_model
    )

    out = choice(input, attn, trf)

    return out


@scope
def block(input, depth, expand_ratio, kernel_size, stride, num_heads, d_model, out_channels):
    assert isinstance(depth, IntScalarParameter), "block depth must be of type IntScalarParameter"
    out = input
    exits = []
    for i in range(depth.max+1):
        out = pattern(
            out,
            expand_ratio=expand_ratio.new(),
            kernel_size=kernel_size.new(),
            stride=stride.new() if i == 0 else 1,
            num_heads=num_heads.new(),
            d_model=d_model.new(),
            out_channels=out_channels.new()
        )
        exits.append(out)

    out = dynamic_depth(*exits, switch=depth)
    res = residual(input, out.shape())
    out = add(out, res)

    return out
