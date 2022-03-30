from copy import deepcopy
from abc import ABC
import torch
from hannah.nas.search_space.utils import get_same_padding
from sympy import solve, symbols, floor, ceiling


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

    def new(self, name=None):
        new_module = deepcopy(self)
        if name:
            new_module.update_name(name)
        return new_module

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

    def update_parameters(self, **kwargs):
        for k, v in kwargs.items():
            if not isinstance(v, Parameter):
                v = Constant(str(k), v)
            self.params[k] = v

    # def fan_out(self):
    #     for k, v in self.params.items():
    #         try:
    #             dims = v.get_config_dims()
    #         except Exception:
    #             pass

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

    def __repr__(self) -> str:
        return "Variable: {}".format(self.func)


class Constant(Parameter):
    def __init__(self, name, value) -> None:
        super().__init__(name)
        self.value = value

    def get(self, mod, ctx):
        return self.value

    def __repr__(self) -> str:
        return "Constant: {}".format(self.value)


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
    kernel_size = op.params['kernel_size'].get(op, ctx)
    if stride == 1:
        padding = 'same'
    elif stride == 2:
        padding = get_same_padding(kernel_size)
    return padding


def infer_padding_symbolic(parameter, op, ctx):
    out, stride, kernel_size, pad, dil, inp = symbols('out stride kernel_size pad dil inp', integer=True)
    stride_param = [value.get(op, ctx) for key, value in op.params.items() if 'stride' in key]
    if len(stride_param) > 1:
        raise Exception('More than one stride in module currently not supported')
    else:
        stride_param = stride_param[0]
    kernel_param = [value.get(op, ctx) for key, value in op.params.items() if 'kernel_size' in key]
    if len(kernel_param) > 1:
        raise Exception('More than one kernel_size in module currently not supported')
    else:
        kernel_param = kernel_param[0]
    dilation_param = [value.get(op, ctx) for key, value in op.params.items() if 'dilation' in key]
    if len(dilation_param) > 1:
        raise Exception('More than one dilation in module currently not supported')
    else:
        dilation_param = dilation_param[0]

    constraints = [out - floor(((inp + 2 * pad - dil * (kernel_size - 1) - 1) / stride) + 1),   # general output size formula
                   out - ceiling(inp / stride),                                                 # output must be same or half size of input
                   stride - stride_param,
                   kernel_size - kernel_param,
                   inp - ctx.input.shape[2],                                                    # note the dimension, non-square 2d inputs not supported
                   dil - dilation_param]
    sol = solve(constraints, dict=True)
    return sol[0][pad]


def multiply_by_stem(parameter, op, ctx):
    return ctx.input.shape[1] * ctx.config['stem_multiplier']


def double_channels(parameter, op, ctx):
    channels = ctx.input.shape[1]
    return channels * 2
