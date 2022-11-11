from hannah.nas.core.expression import Expression
from hannah.nas.dataflow.axis_type import AxisType
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor_expression import TensorExpression
from hannah.nas.dataflow.tensor_type import TensorType
from hannah.nas.expressions.placeholder import DefaultInt, IntRange
from hannah.nas.dataflow.register_ops import add_op, add_shape_func, add_conversion
from hannah.nas.expressions.arithmetic import Floor
import torch.nn.functional as F
from torch.nn import Conv2d as torch_conv
from hannah.nas.parameters.parameters import IntScalarParameter

from hannah.nas.parameters.parametrize import parametrize


@add_op
class Conv2d:
    input: TensorExpression
    weight: TensorExpression
    dilation: Expression = DefaultInt(1)
    stride : Expression = DefaultInt(1)
    padding: Expression = IntScalarParameter(0, 10)
    groups: Expression = DefaultInt(1)


@add_shape_func("Conv2d")
def conv2d_shape(op: OpType):
    def _calc_output_dim(out_dim_name, input_dim, padding, dilation, kernel, stride) -> AxisType:
        input_size = input_dim.size
        kernel_size = kernel.size
        ax = AxisType(name=out_dim_name, size=Floor(((input_size + padding * 2 - dilation * (kernel_size - 1) - 1) / stride) + 1))
        return ax
    input = op.operands[0].tensor_type()
    weight = op.operands[1].tensor_type()

    batch = input['n']
    out_channel = weight['o'].new('c')
    output_height = _calc_output_dim('h', input['h'], op.padding, op.dilation, weight['kh'], op.stride)
    output_width = _calc_output_dim('w', input['w'], op.padding, op.dilation, weight['kw'], op.stride)

    # FIXME: Just take inputs dtype?
    return TensorType((batch, out_channel, output_height, output_width), dtype=input.dtype)


# @add_conversion("Conv2d", target="torch")
# def conv2d_torch(op: OpType):
#     # kernel_size = op.kernel_size
#     # dilation = op.dilation
#     # stride = op.stride
#     # padding = op.padding

#     # input_tensor = op.input.output_tensor()
#     # output_tensor = op.output_tensor()

#     # torch_op = torch_conv(in_channels=input_tensor['c'].size.evaluate(),
#     #                       out_channels=output_tensor['c'].size.evaluate(),
#     #                       kernel_size=kernel_size,
#     #                       stride=stride,
#     #                       padding=padding,
#     #                       dilation=dilation)

#     # # def conv2d_func(input, weight):
#     #     return F.conv2d(input, weight, None, stride.evaluate())
#     torch_op = None
#     return torch_op
