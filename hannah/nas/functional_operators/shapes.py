from hannah.nas.core.expression import Expression
from hannah.nas.expressions.arithmetic import Ceil, Floor


def padding_expression(kernel_size, stride, dilation=1):
    """Symbolically calculate padding such that for a given kernel_size, stride and dilation
    the padding is such that the output dimension is kept the same(stride=1) or halved(stride=2).
    Note: If the input dimension is 1 and stride = 2, the calculated padding will result in
    an output with also dimension 1.

    Parameters
    ----------
    kernel_size : Union[int, Expression]
    stride : Union[int, Expression]
    dilation : Union[int, Expression], optional
        _description_, by default 1

    Returns
    -------
    Expression
    """
    # r = 1 - (kernel_size % 2)
    p = (dilation * (kernel_size - 1) - stride + 1) / 2
    return Ceil(p)


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


def pool_shape(*operands, dims, kernel_size, stride, padding, dilation):
    def _calc_output_dim(input_size, padding, dilation, kernel_size, stride) -> Expression:
        if padding == 'same':
            padding = kernel_size // 2
        ax = Floor(((input_size + padding * 2 - dilation * (kernel_size - 1) - 1) / stride) + 1)
        return ax

    input_shape = operands[0].shape()
    batch = input_shape[0]
    out_channels = input_shape[1]

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


def adaptive_average_pooling_shape(*operands, output_size):
    dims = operands[0].shape()
    # NOTE: dims might be SymbolicSequence. Symbolic sequence has in its symbolic state no fixed length, making it
    # necessary to know and define which dimensions hold values.
    new_dims = [dims[0], dims[1]]
    if isinstance(output_size, int):
        output_size = [output_size]
        
    new_dims.extend(output_size)

    return tuple(new_dims)
