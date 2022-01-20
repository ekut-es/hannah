from hannah.nas.search_space.symbolic_operator import Parameter, SymbolicOperator, Context


def infer_in_channel(parameter: Parameter, op: SymbolicOperator, ctx: Context):
    in_channels = ctx.input.shape[1]
    return in_channels


# example for modified choice param
def restricted_stride(parameter: Parameter, op: SymbolicOperator, ctx: Context):
    padding = op.params['padding'].get(op, ctx)
    # print("paddong", padding)
    if padding == 'same':
        stride = 1
    else:
        # print("P", parameter)
        # print("m", mod)
        idx = ctx.config.get(op.name).get(parameter.name)
        stride = parameter.values[idx]
    return stride


def reduce_channels_by_edge_number(parameter, op, ctx):
    out_channels = ctx.input.shape[1] / ctx.config['in_edges']
    return int(out_channels)


def keep_channels(parameter, op, ctx):
    out_channels = ctx.input.shape[1]
    return out_channels


def multiply_by_stem(parameter, op, ctx):
    return ctx.input.shape[1] * ctx.config['stem_multiplier']


def double_channels(parameter, op, ctx):
    channels = ctx.input.shape[1]
    return channels * 2


def reduce_and_double(parameter, op, ctx):
    out_channels = ctx.input.shape[1] / ctx.config['in_edges']
    return int(out_channels * 2)
