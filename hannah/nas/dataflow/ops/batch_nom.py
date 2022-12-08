from copy import deepcopy
from hannah.nas.core.expression import Expression
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor_expression import TensorExpression
from hannah.nas.dataflow.register_ops import add_op, add_shape_func
from hannah.nas.expressions.placeholder import Categorical, FloatRange, DefaultFloat, DefaultBool
from hannah.nas.parameters.parameters import CategoricalParameter, FloatScalarParameter


@add_op
class BatchNorm2d:
    input: TensorExpression
    eps: Expression = DefaultFloat(0.00001)
    momentum: Expression = DefaultFloat(0.1)
    affine: Expression = DefaultBool(True)


@add_shape_func("BatchNorm2d")
def add_shape(op: OpType):
    return deepcopy(op.operands[0].tensor_type())
