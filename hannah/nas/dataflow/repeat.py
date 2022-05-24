from typing import Any
from hannah.nas.parameters.parameters import Parameter
from hannah.nas.dataflow.dataflow_graph import DataFlowGraph
from types import MethodType


class Repeater:
    def __init__(self, block, num_repeats) -> None:
        self.block = block
        self.num_repeats = num_repeats

    def __str__(self):
        ret = "Repeater"
        return ret

    def __call__(self, input) -> Any:
        out = self.block(input)

        # create Repeat (i.e. child) instance from DataFlowGraph instance
        out = Repeat(out.inputs, out.output, self.num_repeats, out.name)
        return out


class Repeat(DataFlowGraph):
    def __init__(self, inputs, output, num_repeats, name: str = "dataflow") -> None:
        super().__init__(inputs, output, name)
        self.num_repeats = num_repeats

    def dfg_line_representation(self, indent, input_names):
        return '\t'*indent + self.id + " (repeats: {})".format(self.num_repeats) + ':'


def repeat(block, num_repeats=1):
    return Repeater(block=block, num_repeats=num_repeats)
