from typing import Iterable
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor_type import TensorTuple
from hannah.nas.dataflow.dataflow_utils import find_first_op_in_dfg
from hannah.nas.dataflow.tensor_expression import TensorExpression
from hannah.nas.dataflow.tensor import Tensor


class DataFlowGraph(TensorExpression):
    def __init__(self, *operands, output, name: str = "dataflow") -> None:
        super().__init__(*operands, tensor_type=None, name=name)
        self.inputs = []
        self.operand_to_input_map = {}
        if self.operands:
            for i, o in enumerate(self.operands):
                inp = Tensor(name='input')
                self.inputs.append(inp)
                self.operand_to_input_map[o] = inp

        self.output = output
        self.link_users()

    def link_users(self):
        def _rewire_to_placeholder(operand, node, placeholder):
            if operand in node.operands:
                last_output = find_first_op_in_dfg(operand)
                last_output.users.remove(node)

                placeholder.users.append(node)
            elif isinstance(node, DataFlowGraph):
                _rewire_to_placeholder(operand, node.output, placeholder)
            elif isinstance(node, OpType):
                for o in node.operands:
                    _rewire_to_placeholder(operand, o, placeholder)

        for operand, inp in self.operand_to_input_map.items():
            last_output = find_first_op_in_dfg(operand)
            last_output.users.append(self)
            self.users.append(inp)
            _rewire_to_placeholder(operand, self.output, inp)

    def __repr__(self) -> str:
        return "DataFlowGraph(id={})".format(self.id)


def dataflow(func):
    def wrapper_func(*args, **kwargs):
        name = func.__name__
        operands = args
        output = func(*args, **kwargs)

        if isinstance(output, Iterable):
            output = TensorTuple(output, name=name+".output")

        dfg = DataFlowGraph(*operands, output=output, name=name)
        return dfg

    return wrapper_func
