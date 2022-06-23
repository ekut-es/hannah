from hannah.nas.core.expression import Expression
from hannah.nas.dataflow.axis_type import AxisType
from hannah.nas.dataflow.data_type import IntType
from hannah.nas.dataflow.dataflow_graph import dataflow
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor_expression import TensorExpression
from hannah.nas.dataflow.tensor_type import TensorType
from hannah.nas.expressions.placeholder import DefaultInt, UndefinedInt
from hannah.nas.parameters.parameters import CategoricalParameter, IntScalarParameter
from hannah.nas.ops import axis, batched_image_tensor, tensor, weight_tensor
from hannah.nas.dataflow.register_ops import add_op, add_shape_func, add_conversion
from hannah.nas.dataflow.registry import op
from hannah.nas.expressions.arithmetic import Floor


@add_op
class Conv2d:
    input: TensorExpression
    weight: TensorExpression
    # FIXME: DefaultInt or int??
    kernel_size: Expression
    dilation: Expression = DefaultInt(1)
    stride : Expression = DefaultInt(1)
    padding: Expression = DefaultInt(0)


@add_shape_func("Conv2d")
def conv2d_shape(op: OpType):
    # FIXME: Proprietary floor function?

    def _calc_output_dim(out_dim_name, input_dim, padding, dilation, kernel, stride) -> AxisType:
        input_size = input_dim.size
        kernel_size = kernel.size
        ax = AxisType(name=out_dim_name, size=Floor(((input_size + DefaultInt(2) * padding - dilation * (kernel_size - 1) - 1) / stride) + 1))
        return ax
    input = op.input.output_tensor()
    weight = op.weight.output_tensor()

    batch = input['n']
    out_channel = op.weight['o'].new('c')  # FIXME: Get channels from weight or conv2d op. Same with kernel_size?
    output_height = _calc_output_dim('h', input['h'], op.padding, op.dilation, weight['kh'], op.stride)
    output_width = _calc_output_dim('w', input['w'], op.padding, op.dilation, weight['kw'], op.stride)

    # FIXME: Just take inputs dtype?
    return TensorType((batch, out_channel, output_height, output_width), dtype=input.tensor_type.dtype)



# @add_conversion("Conv2D", target="torch")
# def conv2d_torch(op: OpType):
#     pass


@dataflow
def conv2d(input, channel, kernel_size=DefaultInt(1), stride=DefaultInt(1), dilation=DefaultInt(1), padding=DefaultInt(0)):
    weight = weight_tensor(shape=(channel, input['c'], kernel_size, kernel_size), name='weight')
    return op("Conv2d", input, weight, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding)


@dataflow
def chained_convs(input, channel, kernel_size=DefaultInt(1), stride=DefaultInt(1), dilation=DefaultInt(1), padding=DefaultInt(0)):
    weight1 = weight_tensor(shape=(channel, input['c'], kernel_size, kernel_size), name='weight')
    conv1 = op("Conv2d", input, weight1, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding)

    weight2 = weight_tensor(shape=(channel, input['c'], kernel_size, kernel_size), name='weight')
    conv2 = op("Conv2d", conv1, weight2, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding)

    return conv2


if __name__ == '__main__':
    inp = batched_image_tensor(name="input")

    conv = conv2d(inp, channel=IntScalarParameter(4, 64), kernel_size=CategoricalParameter([1, 3, 5]))

    conv['conv2d.0.Conv2d.0'].kernel_size.set_current(3)

    returned_tensor = conv.output.output_tensor()
    for name, ax in returned_tensor.tensor_type.axis.items():
        print("{}: {}".format(name, ax.size.evaluate()))

    ks = CategoricalParameter([1, 3, 5])
    ks.set_current(5)
    convs = chained_convs(inp, channel=IntScalarParameter(4, 64), kernel_size=ks)
    returned_tensor = convs.output.output_tensor()

    for name, ax in returned_tensor.tensor_type.axis.items():
        print("{}: {}".format(name, ax.size.evaluate()))

    print()
