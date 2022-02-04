from copy import deepcopy
from abc import ABC
import numpy as np


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

    def get_config_dims(self):
        param_cfg = {}
        for k, v in self.params.items():
            try:
                param_cfg[v.name] = v.get_config_dims()
            except Exception:
                pass
        return {self.name: param_cfg}

    def __repr__(self):
        return 'SymOp {}'.format(self.name)


class Parameter(ABC):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name

    def get(self, mod_name, ctx):
        pass


class Choice(Parameter):
    def __init__(self, name, *args) -> None:
        super().__init__(name)
        self.values = list(args)

    def get(self, mod, ctx):
        idx = ctx.config.get(mod.name).get(self.name)
        result = self.values[idx]
        return result

    def get_config_dims(self):
        return list(range(len(self.values)))

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


class FloatRange(Parameter):
    def __init__(self, name, min, max) -> None:
        self.name = name
        self.min = min
        self.max = max
        super().__init__(name)


class FloatRangeVector(Parameter):
    def __init__(self, name, min, max, size, init=None) -> None:
        assert max >= min
        self.name = name
        self.min = min
        self.max = max
        self.size = size
        if init:
            self.value = init
        else:
            self.value = np.ones(self.size) * (self.min + ((self.max - self.min) / 2))
        super().__init__(name)

    def get(self, mod, ctx):
        val = ctx.config.get(mod.name).get(self.name)
        # val = np.clip(val, self.min, self.max)
        return val

    def get_config_dims(self):
        return {'min': self.min, 'max': self.max, 'size': self.size}


class Context:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.input = None
        self.outputs = {}
        self.relabel_dict = {}

    def set_input(self, x):
        self.input = x

    def set_cfg(self, cfg):
        self.config = cfg


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
