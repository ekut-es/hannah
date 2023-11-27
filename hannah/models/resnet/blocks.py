from functools import partial
from hannah.models.embedded_vision_net.expressions import expr_product
from hannah.nas.expressions.arithmetic import Ceil
from hannah.nas.expressions.types import Int
from hannah.nas.functional_operators.op import scope
from hannah.models.embedded_vision_net.operators import adaptive_avg_pooling, add, conv2d, conv_relu, depthwise_conv2d, dynamic_depth, pointwise_conv2d, linear, relu, batch_norm, choice, identity
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter


@scope
def conv_relu_bn(input, out_channels, kernel_size, stride):
    out = conv2d(input, out_channels, kernel_size, stride)
    out = batch_norm(out)
    out = relu(out)
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
def block(input, depth, out_channels, kernel_size, stride):
    assert isinstance(depth, IntScalarParameter), "block depth must be of type IntScalarParameter"
    out = input
    exits = []
    for i in range(depth.max+1):
        out = conv_relu_bn(out,
                           out_channels=out_channels.new(),
                           kernel_size=kernel_size.new(),
                           stride=stride.new() if i == 0 else 1)
        exits.append(out)

    out = dynamic_depth(*exits, switch=depth)
    res = residual(input, out.shape())
    out = add(out, res)

    return out


@scope
def classifier_head(input, num_classes):
    out = choice(input, adaptive_avg_pooling)
    out = linear(out, num_classes)
    return out
