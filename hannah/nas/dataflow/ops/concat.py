from typing import List
from hannah.nas.dataflow.axis_type import AxisType
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.register_ops import add_op, add_shape_func
from hannah.nas.dataflow.tensor_expression import TensorExpression
from hannah.nas.dataflow.tensor_type import TensorType


@add_op
class Concat:
    inputs: List[TensorExpression]
    axis: int
    out_axis_name: str


@add_shape_func("Concat")
def add_shape(op: OpType):
    tensors = []
    for operand in op.operands:
        tensors.append(operand.tensor_type())

    for tensor in tensors:
        assert tensors[0].dim() == tensor.dim()

    ax = []
    # constraints = []
    ax_new = AxisType(name=op.out_axis_name)
    for tensor in tensors:
        for i, (ax1, ax2) in enumerate(zip(tensors[0].axis.values(), tensor.axis.values())):
            if i != op.axis:
                con = ax1.size == ax2.size
                op.cond(con)
                ax.append(ax1.new())
            else:
                if ax_new.size:
                    ax_new.size = ax2.size
                else:
                    ax_new.size = ax_new.size + ax2.size
                ax.append(ax_new)

    ax = tuple(ax)
    return TensorType(ax, dtype=tensors[0].dtype)
