

from copy import deepcopy
from hannah.nas.dataflow.dataflow_graph import DataFlowGraph
from hannah.nas.dataflow.repeat import Repeat
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor import Tensor
from hannah.nas.dataflow.tensor_expression import TensorExpression


class DataFlowAnalysis:
    def __init__(self) -> None:
        pass

    def initial_env(self):
        pass

    def create_worklist(self):


    def  analyze(self, expr: TensorExpression, env = None):
        worklist = self.create_worklist(expr)
        if env is  None:
            env = self.initial_env()
        while worklist:
            expr = worklist.pop()
            if isinstance(expr, Repeat):
                changed = self.visit_repeat(expr, env)
            elif isinstance(expr, DataFlowGraph):
                changed = self.visit_dataflowgraph(expr, env)
            elif isinstance(expr, OpType):
                changed = self.visit_optype(expr, env)
            elif isinstance(expr, Tensor):
                changed = self.visit_tensor(expr, env)


            if changed:
                self.extend_worklist(expr)


    def visit_dataflowgraph(self, dfg, env) -> bool:
        old_env = deepcopy(env)
        self.analyze(dfg, env)
        return old_env == env

    def visit_repeat(self, repeat, env) -> bool:
        pass

    def visit_optype(self, op, env) -> bool:
        pass

    def visit_tensor(self, tensor, env) -> bool:

        pass

