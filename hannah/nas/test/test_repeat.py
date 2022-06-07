from hannah.nas.dataflow.dataflow_graph import dataflow, DataFlowGraph
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor_type import TensorType
from hannah.nas.expressions.placeholder import UndefinedInt
from hannah.nas.parameters.parameters import CategoricalParameter, FloatScalarParameter, IntScalarParameter
from hannah.nas.ops import axis, tensor, batched_image_tensor, float_t
from hannah.nas.dataflow.repeat import repeat


@dataflow
def conv_relu(input: TensorType,
              output_channel=IntScalarParameter(4, 64),
              kernel_size=CategoricalParameter([1, 3, 5]),
              stride=CategoricalParameter([1, 2])):

    weight = tensor((axis('o', size=output_channel),
                     axis('i', size=UndefinedInt()),
                     axis('kh', size=kernel_size),
                     axis('kw', size=kernel_size)),
                    dtype=float_t(),
                    name='weight')

    op = OpType(input, weight, stride=stride, name='conv2d')
    relu = OpType(op, name='relu')
    return relu


@dataflow
def block(input: TensorType,
          expansion=FloatScalarParameter(1, 6),
          output_channel=IntScalarParameter(4, 64),
          kernel_size=CategoricalParameter([1, 3, 5]),
          stride=CategoricalParameter([1, 2])):

    out = conv_relu(input,
                    output_channel=output_channel*expansion,
                    kernel_size=kernel_size,
                    stride=stride)
    out = conv_relu(out,
                    output_channel=output_channel,
                    kernel_size=1,
                    stride=1)
    return out


def test_repeat():
    input = batched_image_tensor(name='input')
    graph = repeat(block, num_repeats=IntScalarParameter(min=1, max=5))(input)
    graph = block(graph)
    print()

    assert isinstance(graph, DataFlowGraph)


if __name__ == '__main__':
    test_repeat()
