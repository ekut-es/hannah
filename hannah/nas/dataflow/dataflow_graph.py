
def dataflow(func):
    def wrapper_func(*args, **kwargs):
        input = args  # get inputs
        # TODO: get all leaf nodes
        # output = 'output'  # get outputs
        output = func(*args)

        return DataFlowGraph(inputs=input, outputs=output)
    return wrapper_func


# TODO:
class DataFlowGraph():
    def __init__(self, inputs, outputs) -> None:
        self.inputs = inputs
        self.outputs = outputs