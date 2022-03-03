from copy import deepcopy
from abc import ABC
import torch


class SymbolicOperator:
    def __init__(self, name, target_cls, **kwargs) -> None:
        self.name = name
        self.target_cls = target_cls
        self.params = {}
        for k, v in kwargs.items():
            if not isinstance(v, Parameter):
                v = Constant(str(k), v)
            self.params[k] = v

    def instantiate(self, ctx):
        args = {}
        for key, param in self.params.items():
            args[key] = param.get(self, ctx)
        mod = self.target_cls(**args)
        return mod

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

    def update_name(self, new_name):
        self.name = new_name

    def __repr__(self):
        return 'SymOp {}'.format(self.name)


class Parameter(ABC):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name

    def get(self, mod_name, ctx):
        pass


class OneOf(Parameter):
    def __init__(self, name, *ops) -> None:
        super().__init__(name)
        self.ops = {op.name: op for op in ops}

    def get(self, mod, ctx):
        op_name = ctx.config.get(mod.name).get(self.name)
        op = self.ops[op_name].instantiate(ctx)
        return op

    def get_config_dims(self):
        dims = {}
        for op_name, op in self.ops.items():
            dims[op_name] = op.get_config_dims()[op_name]
        return dims


class Choice(Parameter):
    def __init__(self, name, *args, func=None) -> None:
        super().__init__(name)
        self.values = list(args)
        self.func = func

    def get(self, mod, ctx):
        if self.func:
            self.infer(mod, ctx)
        idx = ctx.config.get(mod.name).get(self.name)
        result = self.values[idx]
        return result

    def infer(self, mod, ctx):
        self.func(self, mod, ctx)

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
            self.value = torch.ones(self.size) * (self.min + ((self.max - self.min) / 2))
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
