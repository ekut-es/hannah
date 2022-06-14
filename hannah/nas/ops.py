from typing import Optional, Tuple, Union

from hannah.nas.dataflow.tensor import Tensor

from .dataflow.axis_type import AxisType
from .dataflow.compression_type import CompressionType
from .dataflow.data_type import DataType, FloatType, IntType
from .dataflow.dataflow_graph import dataflow
from .dataflow.op_type import OpType
from .dataflow.optional_op import OptionalOp
from .dataflow.quantization_type import QuantizationType
from .dataflow.tensor_type import TensorType
from .expressions.placeholder import DefaultInt, UndefinedInt
from .hardware_description.memory_type import MemoryType
import hannah.nas.dataflow.registry as registry


def int_t(signed: bool = True, bits: int = 8):
    return IntType(signed=signed, bits=bits)


def float_t(signed=True, significand_bits=23, exponent_bits=8):
    return FloatType(
        signed=signed, significand_bits=significand_bits, exponent_bits=exponent_bits
    )


def axis(
    name: str, size: Optional[int] = None, compression: Optional[CompressionType] = None
):
    return AxisType(name=name, size=size, compression=compression)


def memory(size: Optional[int] = None, name: Optional[str] = ""):
    return MemoryType(size=size, name=name)


def quantization(
    axis: Optional[AxisType] = None,
    scale: Optional[float] = None,
    zero_point: Optional[float] = None,
):
    return QuantizationType(axis=axis, scale=scale, zero_point=zero_point)


def tensor(
    axis: Tuple[AxisType, ...],
    dtype: DataType,
    quantization: Optional[QuantizationType] = None,
    memory: Optional[MemoryType] = None,
    name: str = "",
):
    tensor_type = TensorType(axis=axis, dtype=dtype, quantization=quantization, memory=memory, name=name)
    return Tensor(tensor_type=tensor_type, name=name)


def batched_image_tensor(dtype=float_t(), name=""):
    return tensor((axis("n"), axis("c"), axis("h"), axis("w")), dtype=dtype, name=name)


def weight_tensor(dtype=float_t(), name=""):
    return tensor((axis("o"), axis("i"), axis("kh"), axis("kw")), dtype=dtype, name=name)


@dataflow
def conv(input):
    kernel_size = UndefinedInt()
    stride = DefaultInt(1)
    weight = tensor(
        (
            axis("o", UndefinedInt()),
            axis("i", UndefinedInt()),
            axis("kh", kernel_size),
            axis("kw", kernel_size),
        ),
        dtype=IntType(),
    )
    return registry.op("conv", input, weight, stride=stride)


@dataflow
def avg_pool(input: TensorType):
    window_size = UndefinedInt()
    stride = UndefinedInt()
    return OpType("avg_pool", input, window_size=window_size, stride=stride)


@dataflow
def requantize(input: TensorType, dtype: DataType, quantization: QuantizationType):
    return OpType("requantize", input, dtype=dtype, quantization=quantization)


@dataflow
def add(input: TensorType, other: TensorType):
    return OpType("add", input, other)


@dataflow
def leaky_relu(input: TensorType, negative_slope: float = 0.0001):
    return OpType("leaky_relu", input, negative_slope=negative_slope)


@dataflow
def relu(input: TensorType):
    return OpType("relu", input)


@dataflow
def broadcast(input: TensorType, axis: int = 1):
    return OpType("broadcast", input, axis=axis)


@dataflow
def optional(op: Union[OpType, TensorType], default: Union[OpType, TensorType]):
    return OptionalOp(op, default)


# @dataflow
# def conv_block(input: TensorType, kernel_size: int = 4):
#     out = add(
#         conv(out, kernel_size=kernel_size, stride=CategoricalParameter(1, 2)),
#         conv(out, kernel_size=DefaultInt(4), name="residual"),
#     )
#     out = leaky_relu(out)
#     return out


# @dataflow
# def network(input: TensorType, blocks: Optional[int] = None):
#     out = inp
#     with Repeat(blocks):
#         with Parametrize(
#             {"leaky_relu.negative_slope": FloatScalarParameter(0.000001, 0.1)}
#         ):
#             out = conv_block(out, kernel_size=4)
#     return out
