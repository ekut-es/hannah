from hannah.nas.dataflow.dataflow_graph import dataflow, DataFlowGraph, flatten
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.registry import op
from hannah.nas.dataflow.tensor_expression import TensorExpression
from hannah.nas.expressions.placeholder import DefaultInt
from hannah.nas.parameters.parameters import CategoricalParameter, FloatScalarParameter, IntScalarParameter
from hannah.nas.ops import batched_image_tensor, weight_tensor
from hannah.nas.dataflow.ops import conv2d, add  # noqa: F401 (Import to load in registry)


@dataflow
def conv_relu(input: TensorExpression,
              output_channel=IntScalarParameter(4, 64),
              kernel_size=CategoricalParameter([1, 3, 5]),
              stride=CategoricalParameter([1, 2])):
    input_tensor = input.tensor_type()
    weight = weight_tensor(shape=(output_channel, input_tensor['c'], kernel_size, kernel_size), name='weight')

    c = op("Conv2d", input, weight, stride=stride)
    relu = OpType(c, name='Relu')
    return relu


@dataflow
def block(input: TensorExpression,
          expansion=FloatScalarParameter(1, 6, name='expansion'),
          output_channel=IntScalarParameter(4, 64),
          kernel_size=CategoricalParameter([1, 3, 5]),
          stride=CategoricalParameter([1, 2])):

    out = conv_relu(input, output_channel=output_channel.new()*expansion.new(), kernel_size=kernel_size.new(), stride=DefaultInt(1))
    out = conv_relu(out, output_channel=output_channel.new(), kernel_size=DefaultInt(1), stride=stride.new())
    return out


@dataflow
def add(input: TensorExpression, other: TensorExpression):  # noqa
    out = op('Add', input, other)
    return out


def test_dataflow():
    input = batched_image_tensor(name='input')
    out = conv_relu(input)
    assert isinstance(out, DataFlowGraph)


def test_dataflow_linking():
    input = batched_image_tensor(name='input')
    out = conv_relu(input)
    out = conv_relu(out)
    assert isinstance(out, DataFlowGraph)


def test_dataflow_block():
    input = batched_image_tensor(name='input')
    out = block(input)
    out = block(out)

    assert isinstance(out, DataFlowGraph)


def test_parallel_blocks():
    input = batched_image_tensor(name='input')
    graph_0 = block(input, stride=IntScalarParameter(min=1, max=2))
    graph_1 = block(input, stride=DefaultInt(2))
    graph = add(graph_0, graph_1)

    assert isinstance(graph, DataFlowGraph)


def test_flatten():
    input = batched_image_tensor(name='input')
    graph_0 = block(input, stride=IntScalarParameter(min=1, max=2))
    graph_1 = block(input, stride=DefaultInt(2))
    graph = add(graph_0, graph_1)
    flattened_graph = flatten(graph)

    assert isinstance(flattened_graph, OpType)


def test_parameter_extraction():
    input = batched_image_tensor(name='input')
    out = block(input, stride=IntScalarParameter(min=1, max=2))
    out = block(out)
    # flattened_graph = flatten(out)
    params = out.parametrization(include_empty=True, flatten=True)

    assert isinstance(out, DataFlowGraph)
    assert 'block.0.conv_relu.1.Conv2d.0.stride' in params and isinstance(params['block.0.conv_relu.1.Conv2d.0.stride'], IntScalarParameter)


if __name__ == '__main__':
    test_dataflow()
    test_dataflow_linking()
    test_dataflow_block()
    test_parallel_blocks()
    test_parameter_extraction()
