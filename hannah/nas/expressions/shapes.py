from hannah.nas.core.expression import Expression
from hannah.nas.expressions.arithmetic import Floor
# from hannah.nas.search_space.utils import get_same_padding


def identity_shape(*input_shapes, kwargs):
    return input_shapes[0]


def conv2d_shape(*input_shapes, kwargs):
    def _calc_output_dim(input_size, padding, dilation, kernel_size, stride) -> Expression:
        if padding == 'same':
            padding = kernel_size // 2
        ax = Floor(((input_size + padding * 2 - dilation * (kernel_size - 1) - 1) / stride) + 1)
        return ax

    # assuming NCHW layout
    batch = input_shapes[0][0]
    output_height = _calc_output_dim(input_shapes[0][2], kwargs.get("padding", "same"), kwargs.get("dilation", 1), kwargs.get("kernel_size", 1), kwargs.get("stride", 1))
    output_width = _calc_output_dim(input_shapes[0][3], kwargs.get("padding", "same"), kwargs.get("dilation", 1), kwargs.get("kernel_size", 1), kwargs.get("stride", 1))

    return (batch, kwargs["out_channels"], output_height, output_width)
