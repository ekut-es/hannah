from hannah.nas.core.expression import Expression
from hannah.nas.expressions.arithmetic import Floor
# from hannah.nas.search_space.utils import get_same_padding


def identity_shape(input_shape):
    return input_shape


def conv2d_shape(input_shape, out_channels, kernel_size, stride=1, padding='same', dilation=1):
    def _calc_output_dim(input_size, padding, dilation, kernel_size, stride) -> Expression:
        if padding == 'same':
            padding = kernel_size // 2
        ax = Floor(((input_size + padding * 2 - dilation * (kernel_size - 1) - 1) / stride) + 1)
        return ax

    # assuming NCHW layout
    batch = input_shape[0]
    output_height = _calc_output_dim(input_shape[2], padding, dilation, kernel_size, stride)
    output_width = _calc_output_dim(input_shape[3], padding, dilation, kernel_size, stride)

    return (batch, out_channels, output_height, output_width)
