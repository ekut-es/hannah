from hannah.nas.expressions.logic import And, If
from hannah.nas.expressions.arithmetic import Ceil


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
