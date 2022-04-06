from typing import Optional, Tuple

from hannah.nas.hardware_description.compression_type import CompressionType
from hannah.nas.hardware_description.memory_type import MemoryType

from hannah.nas.hardware_description.quantization_type import QuantizationType
from hannah.nas.hardware_description.op_type import OpType
from hannah.nas.hardware_description.data_type import DataType, FloatType, IntType
from hannah.nas.hardware_description.tensor_type import TensorType
from hannah.nas.hardware_description.axis_type import AxisType
from hannah.nas.hardware_description.optional_op import OptionalOp


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
):
    return TensorType(axis=axis, dtype=dtype, quantization=quantization, memory=memory)


def op(name, *operands, **attributes):
    return OpType(name, *operands, **attributes)


def optional(op: OpType):
    return OptionalOp(op)


if __name__ == "__main__":
    # UltraTrail Description

    ## DataTypes
    weight_type = int_t(bits=6)
    bias_type = int_t(bits=8)
    activation_type = int_t(bits=8)
    accumulator_type = int_t(bits=20)

    ## Memories
    fmem1 = memory(size=2 ** 16)
    fmem2 = memory(size=2 ** 16)
    fmem3 = memory(size=2 ** 16)
    lmem = memory(size=2 ** 16)
    bmem = memory(size=2 ** 16)

    ## Tensors
    feature_tensor = tensor(
        (axis("N", 1), axis("C"), axis("W"), axis("c", 8)),
        dtype=activation_type,
        quantization=quantization(None, scale=1 / 2 ** 7, zero_point=0.0),
    )

    weight_tensor = tensor(
        (axis("O"), axis("I"), axis("W"), axis("o", 8), axis("i", 8)),
        dtype=weight_type,
        quantization=quantization(None, scale=1 / 2 ** 5, zero_point=0.0),
    )

    linear_weight_tensor = tensor(
        (axis("O"), axis("I"), axis("o", 8), axis("i", 8)),
        dtype=weight_type,
        quantization=quantization(None, scale=1 / 2 ** 7, zero_point=0.0),
    )

    bias_tensor = tensor(
        (axis("N", 1), axis("C"), axis("c", 8)),
        dtype=bias_type,
        quantization=quantization(None, scale=1 / 2 ** 7, zero_point=0.0),
    )

    accumulator_tensor = tensor(
        (axis("N", 1), axis("C"), axis("W"), axis("c", 8)),
        dtype=accumulator_type,
        quantization=quantization(None, scale=1 / 2 ** 14, zero_point=0.0),
    )

    conv = op("conv", feature_tensor, weight_tensor, out_type=accumulator_tensor)
    # linear = op("linear", feature_tensor, linear_weight_tensor, out_type=accumulator_tensor)
    # bias_add = optional(op("add", choice(conv, linear), bias_tensor, out_type=accumulator_tensor))
    bias_add = optional(op("add", conv, bias_tensor, out_type=accumulator_tensor))
    residual_add = optional(
        op("add", bias_add, accumulator_tensor, out_dtype=accumulator_type)
    )
    relu = optional(op("relu", residual_add, out_dtype=accumulator_type))
    requantize = op("requantize", relu, out_type=feature_tensor)

    print("")

