from typing import Union
from hannah.nas.core.expression import Expression
from hannah.nas.dataflow.axis_type import AxisType
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor_expression import TensorExpression
from hannah.nas.dataflow.tensor_type import TensorType
from hannah.nas.expressions.placeholder import IntRange
from hannah.nas.dataflow.register_ops import add_op, add_shape_func
from hannah.nas.expressions.arithmetic import Floor
from hannah.nas.parameters.parameters import IntScalarParameter


@add_op
class Linear:
    input: TensorExpression
    out_features: Expression


@add_shape_func("Linear")
def conv2d_shape(op: OpType):
    input_tensor = op.operands[0].tensor_type()
    out_axis = AxisType(name='features', size=op.out_features)
    return TensorType((input_tensor.axis['n'].new(), out_axis), dtype=input.dtype)
