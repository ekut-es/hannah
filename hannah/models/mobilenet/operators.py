from hannah.models.mobilenet.expressions import padding_expression
from hannah.nas.functional_operators.op import Tensor, scope
from hannah.nas.functional_operators.operators import AdaptiveAvgPooling, BatchNorm, Conv2d, Linear, Relu
from hannah.nas.expressions.types import Int


def relu(input):
    return Relu()(input)


@scope
def batch_norm(input):
    n_chans = input.shape()[1]
    running_mu = Tensor(name='running_mean', shape=(n_chans,), axis=('c',))
    running_std = Tensor(name='running_std', shape=(n_chans,), axis=('c',))
    return BatchNorm()(input, running_mu, running_std)


# def conv_1x1(input, out_channels, stride):
#     in_channels = input.shape()[1]
#     weight = Tensor(name='weight',
#                     shape=(out_channels, in_channels, 1, 1),
#                     axis=('O', 'I', 'kH', 'kW'),
#                     grad=True)

#     conv = Conv2d(stride=stride, dilation=1, groups=in_channels, padding=padding)(input, weight)
#     return conv


# def conv_3x3(input, out_channels):
#     pass


def conv2d(input, out_channels, kernel_size=1, stride=1, dilation=1, groups=1):
    in_channels = input.shape()[1]
    weight = Tensor(name='weight',
                    shape=(out_channels, in_channels, kernel_size, kernel_size),
                    axis=('O', 'I', 'kH', 'kW'),
                    grad=True)
    padding = padding_expression(kernel_size, stride, dilation)

    conv = Conv2d(stride=stride, dilation=dilation, groups=groups, padding=padding)(input, weight)
    return conv


@scope
def depthwise_conv2d(input, out_channels, kernel_size, stride, dilation=1):
    in_channels = 1
    conv = conv2d(input, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=in_channels)
    return conv


@scope
def pointwise_conv2d(input, out_channels):
    conv = conv2d(input, out_channels=out_channels, kernel_size=1)
    return conv


@scope
def inverted_residual(input, out_channels, stride, expand_ratio):
    in_channels = input.shape()[1]
    hidden_dim = Int(in_channels * expand_ratio)
    
    out = input
    if isinstance(expand_ratio, int) and expand_ratio == 1:
        out = depthwise_conv2d(out, in_channels, kernel_size=3, stride=stride)
        out = batch_norm(out)
        out = relu(out)
        out = pointwise_conv2d(out, out_channels)
        out = batch_norm(out)
    else:
        out = pointwise_conv2d(out, hidden_dim)
        out = batch_norm(out)
        out = relu(out)
        out = depthwise_conv2d(out, hidden_dim, kernel_size=3, stride=stride)     
        out = batch_norm(out)
        out = relu(out)
        out = pointwise_conv2d(out, out_channels=out_channels)
        out = relu(out)
    return out


def adaptive_avg_pooling(input):
    return AdaptiveAvgPooling()(input)


def linear(input, out_features):
    input_shape = input.shape()
    in_features = input_shape[1] * input_shape[2] * input_shape[3]
    weight = Tensor(name='weight',
                    shape=(in_features, out_features),
                    axis=('in_features', 'out_features'),
                    grad=True)

    out = Linear()(input, weight)
    return out
