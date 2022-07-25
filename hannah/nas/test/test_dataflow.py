from requests import delete
from hannah.nas.dataflow.dataflow_graph import dataflow, DataFlowGraph, delete_users
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.registry import op
from hannah.nas.dataflow.tensor_type import TensorType
from hannah.nas.expressions.placeholder import DefaultInt
from hannah.nas.parameters.parameters import CategoricalParameter, FloatScalarParameter, IntScalarParameter
from hannah.nas.ops import axis, tensor, batched_image_tensor, float_t, weight_tensor
from hannah.nas.dataflow.ops import conv2d, add  # noqa: F401 (Import to load in registry)


@dataflow
def conv_relu(input: TensorType,
              output_channel=IntScalarParameter(4, 64),
              kernel_size=CategoricalParameter([1, 3, 5]),
              stride=CategoricalParameter([1, 2])):
    input_tensor = input.output_tensor()
    weight = weight_tensor(shape=(output_channel, input_tensor['c'], kernel_size, kernel_size), name='weight')

    c = op("Conv2d", input, weight, stride=stride)
    relu = OpType(c, name='Relu')
    return relu


@dataflow
def block(input: TensorType,
          expansion=FloatScalarParameter(1, 6),
          output_channel=IntScalarParameter(4, 64),
          kernel_size=CategoricalParameter([1, 3, 5]),
          stride=CategoricalParameter([1, 2])):

    out = conv_relu(input, output_channel=output_channel*expansion, kernel_size=kernel_size, stride=DefaultInt(1))
    out = conv_relu(out, output_channel=output_channel, kernel_size=DefaultInt(1), stride=stride)
    return out


@dataflow
def add(input: TensorType, other: TensorType):
    out = op('Add', input, other)
    return out


def test_dataflow():
    input = batched_image_tensor(name='input')
    out = conv_relu(input)
    print(out)
    assert isinstance(out, DataFlowGraph)


def test_dataflow_linking():
    input = batched_image_tensor(name='input')
    out = conv_relu(input)
    out = conv_relu(out)
    print(out)
    assert isinstance(out, DataFlowGraph)


def test_dataflow_block():
    input = batched_image_tensor(name='input')
    out = block(input, stride=DefaultInt(1))
    out = block(out, stride=DefaultInt(2))
    print(out)

    assert isinstance(out, DataFlowGraph)


def test_parallel_blocks():
    input = batched_image_tensor(name='input')
    graph_0 = block(input)
    graph_1 = block(input)
    graph = add(graph_0, graph_1)
    print(graph)



if __name__ == '__main__':
    test_dataflow()
    test_dataflow_linking()
    test_dataflow_block()
    test_parallel_blocks()
