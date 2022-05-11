from torch import TensorType
from ...core.expression import Expression
from ...parameters import CategoricalParameter, FloatScalarParameter, IntScalarParameter
from ...ops import conv, bn, relu, swish, leaky_relu
from .activation import act


def mbconv_block(input : TensorType, activation_type: Expression[str], expand_ratio:  Expression[float], output_channels: Expression[int], stride: Expression[int], kernel_size: Expression[int]):
    input_channels = input.dim("C")
    hidden_dim = input_channels*expand_ratio
    pw = act(bn(conv(input, kernel_size=1, output_channels=hidden_dim)), activation_type=activation_type)
    dw = act(bn(conv(pw, kernel_size=kernel_size, output_channels=hidden_dim, stride=stride, padding="same")), activation_type=activation_type)
    pw_linear = bn(conv(dw, kernel_size=1, output_channels=output_channels))

    output = if_then_else(output_channels == input_channels and stride == 1, pw_linear + input, pw_linear)

    return output


def mbconv_block(input : TensorType, expand_ratio:  Expression[float], output_channels: Expression[int], stride: Expression[int], kernel_size: Expression[int]):
    input_channels = input.dim("C")
    hidden_dim = input_channels*expand_ratio

    act = CategoricalParameter([relu(pw), swish(pw), leaky_relu(pw)])
    pw = act(bn(conv(input, kernel_size=1, output_channels=hidden_dim)))

    dw = bn(conv(pw, kernel_size=kernel_size, output_channels=hidden_dim, stride=stride, padding="same"))
    dw = CategoricalParameter([relu(pw), swish(pw), leaky_relu(pw)])
    pw_linear = bn(conv(dw, kernel_size=1, output_channels=output_channels))

    output = if_then_else(output_channels == input_channels and stride == 1, pw_linear + input, pw_linear)

    return output

def mbconv_block(input : TensorType, expand_ratio:  Expression[float], output_channels: Expression[int], stride: Expression[int], kernel_size: Expression[int],
                 negative_slope: Expression[float] = FloatScalarParameter(0.05, 0.15)):
    input_channels = input.dim("C")
    hidden_dim = input_channels*expand_ratio
    pw = bn(conv(input, kernel_size=1, output_channels=hidden_dim))
    pw = CategoricalParameter([relu(pw), swish(pw), leaky_relu(pw, negative_slope=negative_slope)])
    dw = bn(conv(pw, kernel_size=kernel_size, output_channels=hidden_dim, stride=stride, padding="same"))
    dw = CategoricalParameter([relu(pw), swish(pw), leaky_relu(pw, negative_slope=negative_slope)])
    pw_linear = bn(conv(dw, kernel_size=1, output_channels=output_channels))

    output = if_then_else(output_channels == input_channels and stride == 1, pw_linear + input, pw_linear)

    return output

