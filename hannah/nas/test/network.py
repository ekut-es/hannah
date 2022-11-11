from hannah.nas.dataflow.dataflow_graph import dataflow
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.registry import op
from hannah.nas.dataflow.tensor_expression import TensorExpression
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter
from hannah.nas.ops import weight_tensor
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
          expansion=IntScalarParameter(1, 6),
          output_channel=IntScalarParameter(4, 64),
          kernel_size=CategoricalParameter([1, 3, 5]),
          stride=CategoricalParameter([1, 2])):
    input_tensor = input.tensor_type()
    out = conv_relu(input, output_channel=input_tensor['c'] * expansion, kernel_size=kernel_size, stride=stride)
    out = conv_relu(out, output_channel=output_channel.new(), kernel_size=kernel_size, stride=stride)
    out = conv_relu(out, output_channel=output_channel.new(), kernel_size=kernel_size, stride=stride)
    return out


@dataflow
def residual(input: TensorExpression,
             stride,
             output_channel):
    out = conv_relu(input, stride=stride, output_channel=output_channel.new(), kernel_size=CategoricalParameter([1, 3, 5]))
    return out


@dataflow
def add(input: TensorExpression, other: TensorExpression):  # noqa
    out = op('Add', input, other)
    return out


@dataflow
def residual_block(input: TensorExpression, stride, output_channel):
    main_branch = block(input, stride=stride, output_channel=output_channel)
    residual_branch = residual(input, stride=stride, output_channel=output_channel)
    add_branches = add(main_branch, residual_branch)
    return add_branches
