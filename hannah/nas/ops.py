from torch import negative
from hannah.nas.dataflow.axis_type import AxisType
from hannah.nas.dataflow.data_type import IntType
from hannah.nas.dataflow.tensor_type import TensorType
from hannah.nas.dataflow.op_type import OpType
from hannah.nas.parameters.parameters import CategoricalParameter, FloatScalarParameter, IntScalarParameter, UndefinedInt
from hannah.nas.dataflow.dataflow_graph import dataflow

@dataflow
def conv(input):
    kernel_size = UndefinedInt()
    stride = DefaultInt(1)
    weight = Tensor(Undefined, input.size(1), kernel_size, kernel_size)
    return OpType(input, weight, stride)

# TODO: Fix parametrize for op functions
@dataflow
def leaky_relu(input):
    return OpType(input, negative_slope=DefaultFloat(0.0001))


@dataflow
def conv_block(input : TensorType, kernel_size : int = 4):
    out = add(conv(out, kernel_size=kernel_size, stride=CategoricalParameter(1, 2)),
              conv(out, kernel_size=DefaultInt(4), name='residual'))
    out = leaky_relu(out)
    return out


@dataflow
def network(input):
    out = inp
    with Repeat(IntScalarParameter(1, 10), name="blocks"):
            with Parametrize({"leaky_relu.negative_slope": FloatScalarParameter(0.000001, 0.1)}):
                out = conv_block(out, kernel_size=4)
    return out


if __name__ == '__main__':

    inp = Tensor()
    my_network = network(inp)
