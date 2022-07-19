from hannah.nas.dataflow.dataflow_graph import dataflow, flatten
from hannah.nas.dataflow.ops import conv2d, add  # noqa: F401 (Import to load in registry)
from hannah.nas.dataflow.registry import op
from hannah.nas.expressions.placeholder import DefaultInt
from hannah.nas.ops import batched_image_tensor, weight_tensor
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter


@dataflow
def conv(input, channel, kernel_size=DefaultInt(1), stride=DefaultInt(1), dilation=DefaultInt(1)):
    weight = weight_tensor(shape=(channel, input['c'], kernel_size, kernel_size), name='weight')
    padding = kernel_size // 2
    return op("Conv2d", input, weight, dilation=dilation, stride=stride, padding=padding)


@dataflow
def convs(input, channel, kernel_size=DefaultInt(1), stride=DefaultInt(1), dilation=DefaultInt(1)):
    padding = kernel_size // 2
    input_tensor = input.output_tensor()

    weight1 = weight_tensor(shape=(channel, input_tensor['c'], kernel_size, kernel_size), name='weight')
    conv1 = op("Conv2d", input, weight1, dilation=dilation, stride=stride, padding=padding)
    # conv1_tensor = conv1.output_tensor()

    # weight2 = weight_tensor(shape=(channel, conv1_tensor['c'], kernel_size, kernel_size), name='weight')
    weight2 = weight_tensor(shape=(channel, input_tensor['c'], kernel_size, kernel_size), name='weight')
    conv2 = op("Conv2d", conv1, weight2, dilation=dilation, stride=stride, padding=padding)

    return conv2


@dataflow
def add(input, other):
    return op('Add', input, other)


def traverse_users(node):
    print(node)
    for user in node.users:
        traverse_users(user)

def test_dfg_removal():
    inp = batched_image_tensor(name="input")

    kernel_size = CategoricalParameter([1, 3, 5])
    stride = CategoricalParameter([1, 2])
    channel = IntScalarParameter(4, 64)
    channel.set_current(52)
    graph = conv(inp, channel=channel, kernel_size=kernel_size, stride=stride)


def test_chained_convs_removal():
    inp = batched_image_tensor(name="input")

    ks = CategoricalParameter([1, 3, 5])
    ks.set_current(3)
    graph = convs(inp, channel=IntScalarParameter(4, 64), kernel_size=ks)
    # flattened_graph = flatten(graph)

    print()


def test_chained_dfg_removal():
    inp = batched_image_tensor(name="input")

    ks = CategoricalParameter([1, 3, 5])
    ks.set_current(3)
    graph = convs(inp, channel=IntScalarParameter(4, 64), kernel_size=ks)
    graph1 = convs(graph, channel=IntScalarParameter(4, 64), kernel_size=ks)
    flattened_graph = flatten(graph1)
    # traverse_users(inp)
    print()


def test_parallel_branch_dfg_removal():
    inp = batched_image_tensor(name="input")

    ks = CategoricalParameter([1, 3, 5])
    ks.set_current(3)
    graph_0 = convs(inp, channel=IntScalarParameter(4, 64), kernel_size=ks)
    graph_1 = convs(inp, channel=IntScalarParameter(4, 64), kernel_size=ks)

    graph_add = add(graph_0, graph_1)
    flattened_graph = flatten(graph_add)
    traverse_users(inp)
    print()



if __name__ == '__main__':
    # test_dfg_removal()
    # test_chained_convs_removal()
    # test_chained_dfg_removal()
    test_parallel_branch_dfg_removal()
    print()
