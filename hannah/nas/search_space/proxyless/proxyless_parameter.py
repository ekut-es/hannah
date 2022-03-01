from hannah.nas.search_space.symbolic_operator import (
    Parameter,
    SymbolicOperator,
    Context,
)
import numpy as np


def restricted_stride(parameter: Parameter, op: SymbolicOperator, ctx: Context):
    if not hasattr(ctx, 'reduction_strides'):
        ctx.reduction_strides = 0
    if not hasattr(ctx, 'max_reductions'):
        ctx.max_reductions = 2
    if ctx.reduction_strides >= ctx.max_reductions:
        parameter.values = [1] * len(parameter.values)
    else:
        ctx.reduction_strides += 1


def stochastic_channel_expansion(parameter: Parameter, op: SymbolicOperator, ctx: Context):
    in_channels = ctx.input.shape[1]
    if not hasattr(ctx, 'expansion_prob'):
        ctx.expansion_prob = 0.2
    rnd = np.random.rand()
    if rnd < ctx.expansion_prob:
        out_channels = 2*in_channels
    else:
        out_channels = in_channels
    return out_channels
