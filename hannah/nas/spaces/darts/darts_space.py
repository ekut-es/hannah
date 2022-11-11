from hannah.nas.dataflow.dataflow_graph import dataflow
from hannah.nas.dataflow.registry import op
from hannah.nas.expressions.placeholder import DefaultInt
from hannah.nas.ops import weight_tensor
# from hannah.nas.dataflow.ops import conv2d, sum, concat


@dataflow
def stem(input):
    return input


@dataflow
def sum_node(*inputs):
    return op('Sum', *inputs)


@dataflow
def concat_node(*inputs):
    return op('Concat', *inputs, axis=1, out_axis_name="c")


@dataflow
def input_node(input):
    return op('Identity', input)


@dataflow
def op_node(input):
    input_tensor = input.tensor_type()
    output_channel = DefaultInt(32)  # FIXME:
    kernel_size = DefaultInt(3)      # FIXME:
    stride = DefaultInt(1)           # FIXME:
    weight = weight_tensor(shape=(output_channel, input_tensor['c'], kernel_size, kernel_size), name='weight')
    c = op("Conv2d", input, weight, stride=stride)
    return c


@dataflow
def darts_cell(input, input_prev, num_nodes=4, reduction=True):
    input_0 = input_node(input)
    input_1 = input_node(input_prev)

    nodes = [input_0, input_1]

    for _ in range(num_nodes):
        connections = []
        for node in nodes:
            op_connection = op_node(node)
            connections.append(op_connection)
        s = sum_node(*connections)
        nodes.append(s)

    # input nodes are not connected to concat/output node
    nodes = nodes[2:]
    c = concat_node(*nodes)

    return c


@dataflow
def darts_space(input, num_cells=4):
    output = input_node(input)
    previous = input_node(input)
    for _ in range(num_cells):
        tmp = output
        output = darts_cell(output, previous)
        previous = tmp

    return output
