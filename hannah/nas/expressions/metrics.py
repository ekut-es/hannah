from hannah.nas.core.expression import Expression
from hannah.nas.expressions.arithmetic import Floor


def conv2d_macs(input, output, kwargs):
    volume_ofm = output[0] * output[1] * output[2] * output[3]
    return volume_ofm * (kwargs['in_channels'] / kwargs.get('groups', 1) * kwargs['kernel_size'] * kwargs['kernel_size'])


def conv2d_weights(input, output, kwargs):
    return kwargs['out_channels'] * kwargs['in_channels'] / kwargs.get('groups', 1) * kwargs['kernel_size'] * kwargs['kernel_size']


def linear_macs(input, output, kwargs):
    return kwargs['in_features'] * kwargs['out_features']


def linear_weights(input, output, kwargs):
    return linear_macs(input, output, kwargs)
