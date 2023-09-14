from hannah.nas.expressions.logic import And, If
from hannah.nas.expressions.arithmetic import Ceil


def padding_expression(kernel_size, stride, dilation = 1):
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


def expr_product(expressions: list):
    res = None
    for expr in expressions:
        if res:
            res = res * expr
        else:
            res = expr
    return res


def expr_sum(expressions: list):
    res = None
    for expr in expressions:
        if res:
            res = res + expr
        else:
            res = expr
    return res


# FIXME: replace in search space with depth_aware_sum
def num_layer_constraint(depth_expr, num_blocks):
    num = num_blocks
    ct = 1
    res = None
    for expr in depth_expr:
        if res:
            res = res + (expr * If(num >= ct, 1, 0))
        else:
            res = expr
        ct += 1
    return res

def depth_aware_downsampling(strides, params):
    res = strides.pop(-1)  # get stride of stem
    for stride in strides:
        block_num = int(stride.id.split(".")[1])
        pattern_num = int(stride.id.split(".")[3])
        block_depth = params[f"block.{block_num}.depth"]
        num_blocks = params["num_blocks"]

        if res:
            res = res * If(num_blocks > block_num, If(block_depth > pattern_num, stride, 1), 1)
        else:
            res = If(num_blocks > block_num, If(block_depth > pattern_num, stride, 1), 1)
    return res


def depth_aware_sum(param_list, depth_param):
    ct = 1
    res = None
    for expr in param_list:
        if res:
            res = res + (expr * If(depth_param >= ct, 1, 0))
        else:
            res = expr
        ct += 1
    return res
