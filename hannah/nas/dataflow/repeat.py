from typing import Any
from hannah.nas.dataflow.dataflow_graph import DataFlowGraph, delete_users


class Repeater:
    def __init__(self, block, num_repeats) -> None:
        self.block = block
        self.num_repeats = num_repeats

    def __str__(self):
        ret = "Repeater"
        return ret

    def __call__(self, *args, **kwargs) -> Any:
        out_block = self.block(*args, **kwargs)
        operands = out_block.operands
        output = out_block.output
        name = out_block.name

        delete_users(out_block, out_block)
        del out_block


        # create Repeat (i.e. child) instance from DataFlowGraph instance
        out = Repeat(*operands, output=output, num_repeats=self.num_repeats, name=name)
        return out


class Repeat(DataFlowGraph):
    def __init__(self, *operands, output, num_repeats, name: str = "dataflow") -> None:
        super().__init__(*operands, output=output, name=name)
        self.num_repeats = num_repeats

    def dfg_line_representation(self, indent, input_names):
        return '\t'*indent + self.id + " (repeats: {})".format(self.num_repeats) + ':'

    def __repr__(self) -> str:
        return "DataFlowGraph(id={}) - repeats: ({})".format(self.id, self.num_repeats)


def repeat(block, num_repeats=1):
    return Repeater(block=block, num_repeats=num_repeats)
