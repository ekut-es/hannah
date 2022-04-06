from typing import Optional, Tuple

from .compression_type import CompressionType
from .memory_type import MemoryType

from .quantization_type import QuantizationType
from .op_type import OpType
from .data_type import DataType, FloatType, IntType
from .tensor_type import TensorType
from .axis_type import AxisType
from .optional import Optional


def int(signed: bool = True, bits: int = 8):
    return IntType(signed=signed, bits=bits)


def float(signed=True, significand_bits=23, exponent_bits=8):
    return FloatType(
        signed=signed, significand_bits=significand_bits, exponent_bits=exponent_bits
    )


def axis(
    name: str, size: Optional[int] = None, compression: Optional[CompressionType] = None
):
    return AxisType(name=name, size=size, compression=compression)


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
):
    return TensorType(axis=axis, dtype=dtype, quantization=quantization, memory=memory)


def op(name, *operands, **attributes):
    return OpType(name, *operands, **attributes)


def optional(op: OpType):
    return Optional(op)
