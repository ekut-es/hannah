# from hannah.nas.search_space.symbolic_space import Space
from hannah.nas.search_space.symbolic_operator import SymbolicOperator
import hannah.conf  # noqa
from torch.fx import symbolic_trace
import inspect
import torch.nn as nn
import torch


class MethodWrapper(nn.Module):
    def __init__(self, method, args, kwargs) -> None:
        super().__init__()
        # self.name = name
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        xargs = []
        if isinstance(x, list):
            x, *xargs = x
        return getattr(x, self.method)(*xargs, *self.args, **self.kwargs)


class FunctionWrapper(nn.Module):
    def __init__(self, function, args=[], kwargs={}) -> None:
        super().__init__()
        # self.name = name
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        if isinstance(x, list):
            return self.function(*x, *self.args, **self.kwargs)
        else:
            return self.function(x, *self.args, **self.kwargs)

    def __repr__(self):
        return "Wrapped({})".format(self.function)


class TorchConverter:
    def __init__(self) -> None:
        pass

    def convert_model_to_space(self, model, space):
        gm = symbolic_trace(model)
        modules = dict(gm.named_modules())
        symops = {}

        for node in gm.graph.nodes:
            if node.op == 'call_module':
                original_module = modules[node.target]
                sig = inspect.signature(original_module.__init__)
                args = {}
                for p in sig.parameters.values():
                    if hasattr(original_module, p.name):
                        args[p.name] = getattr(original_module, p.name)
                    else:
                        try:
                            args[p.name] = p.default
                        except Exception as e:
                            print(str(e))
                symop = SymbolicOperator(node.name,
                                         original_module.__class__,
                                         **args)
                symops[node.name] = symop
            elif node.op == 'call_function':
                args = [a for a in node.args if not isinstance(a, torch.fx.Node)]
                symop = SymbolicOperator(node.name,
                                         FunctionWrapper,
                                         function=node.target,
                                         args=args,
                                         kwargs=node.kwargs)
                symops[node.name] = symop
            elif node.op == 'call_method':
                args = [a for a in node.args if not isinstance(a, torch.fx.Node)]
                symop = SymbolicOperator(node.name,
                                         MethodWrapper,
                                         method=node.target,
                                         args=args,
                                         kwargs=node.kwargs)

                symops[node.name] = symop
            if node.name in symops:
                space.add_node(symops[node.name])
                inputs = node.all_input_nodes
                for inp in inputs:
                    if inp.name in symops:
                        space.add_edge(symops[inp.name], symops[node.name])

        return space
