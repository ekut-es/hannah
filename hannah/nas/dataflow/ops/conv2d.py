from torch import conv3d
from hannah.nas.core.expression import Expression
from hannah.nas.dataflow.axis_type import AxisType
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor_expression import TensorExpression
from hannah.nas.dataflow.tensor_type import TensorType
from hannah.nas.expressions.placeholder import DefaultInt
from hannah.nas.dataflow.register_ops import add_op, add_shape_func, add_conversion
from hannah.nas.expressions.arithmetic import Floor
from torch.nn import Conv2d as torch_conv


@add_op
class Conv2d:
    input: TensorExpression
    weight: TensorExpression
    # FIXME: DefaultInt or int??
    kernel_size: Expression = DefaultInt(3)
    dilation: Expression = DefaultInt(1)
    stride : Expression = DefaultInt(1)
    padding: Expression = DefaultInt(0)


@add_shape_func("Conv2d")
def conv2d_shape(op: OpType):
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


@add_conversion("Conv2d", target="torch")
def conv2d_torch(op: OpType):
    kernel_size = op.kernel_size.evaluate()
    dilation = op.dilation.evaluate()
    stride = op.stride.evaluate()
    padding = op.padding.evaluate()

    input_tensor = op.input.output_tensor()
    output_tensor = op.output_tensor()

    torch_op = torch_conv(in_channels=input_tensor['c'].size.evaluate(),
                          out_channels=output_tensor['c'].size.evaluate(),
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          dilation=dilation)
    return torch_op
