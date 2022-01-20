from copy import deepcopy
from abc import ABC


class SymbolicOperator:
    def __init__(self, name, mod, **kwargs) -> None:
        self.name = name
        self.mod = mod
        self.params = {}
        for k, v in kwargs.items():
            self.params[k] = v

    def instantiate(self, ctx):
        args = {}
        for key, param in self.params.items():
            args[key] = param.get(self, ctx)
        return self.mod(**args)

    def new(self):
        return deepcopy(self)

    def get_params(self):
        return self.mod_args

    def __repr__(self):
        return 'SymOp {}'.format(self.name)


class Parameter(ABC):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name

    def get(self, mod_name, ctx):
        pass


class Choice(Parameter):
    def __init__(self, name, *args, func=None) -> None:
        super().__init__(name)
        self.values = list(args)
        self.func = func

    def get(self, mod, ctx):
        if self.func:
            result = self.infer(mod, ctx)
        else:
            idx = ctx.config.get(mod.name).get(self.name)
            result = self.values[idx]
        return result

    def infer(self, mod, ctx):
        return self.func(self, mod, ctx)

    def __repr__(self) -> str:
        return str(self.values)


class Variable(Parameter):
    def __init__(self, name, func) -> None:
        super().__init__(name)
        self.func = func

    def get(self, mod, ctx):
        return self.infer(mod, ctx)

    def infer(self, mod, ctx):
        return self.func(self, mod, ctx)


class Constant(Parameter):
    def __init__(self, name, value) -> None:
        super().__init__(name)
        self.value = value

    def get(self, mod, ctx):
        return self.value


class Context:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.input = None
        self.outputs = {}
        self.relabel_dict = {}

    def set_input(self, x):
        self.input = x


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
    out_channels = ctx.input.shape[1] * ctx.config['in_edges']
    return out_channels


def keep_channels(parameter, op, ctx):
    out_channels = ctx.input.shape[1]
    return out_channels


def infer_padding(parameter, op, ctx):
    stride = op.params['stride'].get(op, ctx)
    if stride == 1:
        padding = 'same'
    elif stride == 2:
        padding = 0
    return padding


def multiply_by_stem(parameter, op, ctx):
    return ctx.input.shape[1] * ctx.config['stem_multiplier']


def double_channels(parameter, op, ctx):
    channels = ctx.input.shape[1]
    return channels * 2
