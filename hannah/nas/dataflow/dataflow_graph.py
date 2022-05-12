from typing import Iterable


def dataflow(func):
    def wrapper_func(*args, **kwargs):
        input = args  # get inputs
        # TODO: get all leaf nodes
        # output = 'output'  # get outputs
        output = func(*args, **kwargs)
        name = func.__name__
        if isinstance(output, Iterable):
            outputs = tuple(output)
        else:
            outputs = (output,)

        print(outputs)

        return DataFlowGraph(inputs=input, outputs=output, name=name)

    return wrapper_func


# TODO:
class DataFlowGraph:
    def __init__(self, inputs, outputs, name: str = "") -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.name = name

    def __str__(self):
        ret = "dataflow(name=" + self.name + "\n"
        # for output in self.outputs:
        ret += str(self.outputs) + "\n"

        ret += "\n)"
        return ret
