from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor_type import TensorType
from hannah.nas.dataflow.register_ops import add_op, add_shape_func

from hannah.nas.dataflow.tensor_expression import TensorExpression


@add_op
class Add:
    input: TensorExpression
    other: TensorExpression


@add_shape_func("Add")
def add_shape(op: OpType):
    input = op.input.output_tensor()
    other = op.other.output_tensor()

    assert input.dim == other.dim
    ax = []
    for ax1, ax2 in zip(input.tensor_type.axis, other.tensor_type.axis):
        con = input.tensor_type.axis[ax1].size == other.tensor_type.axis[ax2].size
        assert con.evaluate(), """Tensor axis sizes do not match: Axis {} with dimension
                               {} and axis {} with dimension {}""".format(ax1,
                                                                          input.tensor_type.axis[ax1].size,
                                                                          ax2,
                                                                          other.tensor_type.axis[ax2].size)
        ax.append(input.tensor_type.axis[ax1].new())

    ax = tuple(ax)
    return TensorType(ax, input.tensor_type.dtype)
