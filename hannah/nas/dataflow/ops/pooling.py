from typing import Union
from hannah.nas.core.expression import Expression
from hannah.nas.dataflow.axis_type import AxisType
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor_expression import TensorExpression
from hannah.nas.dataflow.register_ops import add_op, add_shape_func
from hannah.nas.dataflow.tensor_type import TensorType
from hannah.nas.expressions.placeholder import IntRange
from hannah.nas.parameters.parameters import IntScalarParameter


@add_op
class AdaptiveAveragePooling:
    input: TensorExpression
    output_size: Expression


@add_shape_func("AdaptiveAveragePooling")
def add_shape(op: OpType):
    tensor = op.operands[0].tensor_type()
    new_h = AxisType(name='h', size=op.output_size)
    new_w = AxisType(name='w', size=op.output_size)
    output_tensor_axis = (tensor.axis['n'].new(), tensor.axis['c'].new(), new_h, new_w)
    return TensorType(output_tensor_axis, dtype=tensor.dtype)
