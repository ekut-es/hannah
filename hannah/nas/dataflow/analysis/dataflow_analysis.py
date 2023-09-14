#
# Copyright (c) 2022 Hannah contributors.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from copy import deepcopy

from hannah.nas.dataflow.dataflow_graph import DataFlowGraph
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.repeat import Repeat
from hannah.nas.dataflow.tensor import Tensor
from hannah.nas.dataflow.tensor_expression import TensorExpression


class DataFlowAnalysis:
    def __init__(self) -> None:
        pass

    def initial_env(self):
        pass

    def create_worklist(self):
        pass

    def analyze(self, expr: TensorExpression, env=None):
        worklist = self.create_worklist(expr)
        if env is None:
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
