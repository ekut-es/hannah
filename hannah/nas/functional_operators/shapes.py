from hannah.nas.core.expression import Expression
from hannah.nas.expressions.arithmetic import Floor


def identity_shape(*operands):
    return operands[0].shape()


def conv_shape(*operands, dims, stride, padding, dilation):
    def _calc_output_dim(input_size, padding, dilation, kernel_size, stride) -> Expression:
        if padding == 'same':
            padding = kernel_size // 2
        ax = Floor(((input_size + padding * 2 - dilation * (kernel_size - 1) - 1) / stride) + 1)
        return ax

    # assuming NCHW layout
    input_shape = operands[0].shape()
    weight_shape = operands[1].shape()

    batch = input_shape[0]
    out_channels = weight_shape[0]
    kernel_size = weight_shape[2]

    num_spatial_dims = dims
    spatial_dims = []
    for i in range(2, num_spatial_dims + 2):
        output_dim = _calc_output_dim(input_shape[i], padding, dilation, kernel_size, stride)
        spatial_dims.append(output_dim)
    return (batch, out_channels, *spatial_dims)


def linear_shape(*operands):
    batch = operands[0].shape()[0]
    out_features = operands[1].shape()[1]
    return (batch, out_features)


def adaptive_average_pooling2d_shape(*operands, output_size):
    dims = operands[0].shape()
    # NOTE: dims might be SymbolicSequence. Symbolic sequence has in its symbolic state no fixed length, making it
    # necessary to know and define which dimensions hold values.
    new_dims = [dims[0], dims[1]]
    new_dims += [output_size[0], output_size[1]]

    return tuple(new_dims)
