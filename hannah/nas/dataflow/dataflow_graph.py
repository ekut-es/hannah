from typing import Iterable

from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.optional_op import OptionalOp
from hannah.nas.dataflow.tensor_type import TensorType


# TODO:
class DataFlowGraph:
    def __init__(self, outputs, name: str = "") -> None:
        self.outputs = outputs
        self.name = name
        self.inputs = collect_leaf_nodes(self.outputs)

    def __str__(self):
        ret = "dataflow(name=" + self.name + "\n"
        # for output in self.outputs:
        ret += str(self.outputs) + "\n"

        ret += "\n)"
        return ret

    def __repr__(self) -> str:
        return str(self)


def dataflow(func):
    def wrapper_func(*args, **kwargs):
        args = expose_dataflow_outputs(args)
        outputs = func(*args, **kwargs)
        name = func.__name__
        if isinstance(outputs, Iterable):
            outputs = tuple(outputs)
        else:
            outputs = (outputs,)
        outputs = expose_dataflow_outputs(outputs)
        return DataFlowGraph(outputs=outputs, name=name)

    return wrapper_func


def expose_dataflow_outputs(args: tuple):
    exposed_args = []
    for arg in args:
        if isinstance(arg, DataFlowGraph):
            exposed_args.extend(arg.outputs)
        else:
            exposed_args.append(arg)
    return tuple(exposed_args)


def collect_leaf_nodes(g):
    def _propagate(x, leafs):
        if isinstance(x, Iterable):
            for i in x:
                leafs = _propagate(i, leafs)
            return leafs
        elif isinstance(x, DataFlowGraph):
            leafs.extend(x.inputs)
            return leafs
        elif isinstance(x, OpType):
            for inp in x.operands:
                leafs = _propagate(inp, leafs)
            return leafs
        elif isinstance(x, OptionalOp):
            leafs = _propagate(x.op, leafs)
            return leafs
        elif isinstance(x, TensorType):
            return leafs + [x]

    leafs = _propagate(g, [])
    return leafs
