from torch import nn as nn
from copy import deepcopy
from abc import ABC, abstractmethod


class SymbolicOperator:
    def __init__(self, name, mod: nn.Module, **kwargs) -> None:
        self.name = name
        self.mod = mod
        self.params = {}
        for k, v in kwargs.items():
            self.params[k] = v

    def instantiate(self, ctx):
        args = {}
        for key, param in self.params.items():
            args[key] = param.get(self.name, ctx)

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

    def get(self, name, ctx):
        pass


class Choice(Parameter):
    def __init__(self, name, *args) -> None:
        super().__init__(name)
        self.values = list(args)

    def get(self, name, ctx):
        idx = ctx.get('config').get(name).get(self.name)
        return self.values[idx]

    def __repr__(self) -> str:
        return str(self.values)


class Variable(Parameter):
    def __init__(self, name, infer_func) -> None:
        super().__init__(name)
        self.infer_func = infer_func

    def get(self, name, ctx):
        return self.infer(ctx)

    def infer(self, ctx):
        return self.infer_func(self.name, ctx)


class Context:
    def __init__(self, config: dict) -> None:
        self.values = {}
        self.values['config'] = config

    def get(self, key):
        return self.values[key]
