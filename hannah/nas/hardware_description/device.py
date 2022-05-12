from abc import ABC, abstractmethod
from typing import List

from ..dataflow.dataflow_graph import DataFlowGraph, dataflow
from ..dataflow.op_type import OpType
from ..expressions.placeholder import IntRange, UndefinedFloat, UndefinedInt
from ..hardware_description.memory_type import MemoryType
from ..ops import (
    add,
    avg_pool,
    axis,
    broadcast,
    int_t,
    optional,
    quantization,
    relu,
    requantize,
    tensor,
)
from ..parameters.parametrize import parametrize


class Device(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._ops = []
        self._memories = []

    @property
    def ops(self) -> List[DataFlowGraph]:
        return self._ops

    @property
    def memories(self) -> List[MemoryType]:
        return self._memories

    def __str__(self):
        res = self.__class__.__name__ + ":\n"
        res += "Ops:\n"
        for op in self.ops:
            res += str(op) + "\n"
        for memory in self.memories:
            res += str(memory) + "\n"
        return res


@dataflow
def ut_op(
    weight_bits: int = 8,
    bias_bits: int = 8,
    activation_bits: int = 8,
    accumulator_bits: int = 8,
    max_weight_bits: int = 8,
    max_kernel_size: int = 2**4,
    max_input_length: int = 2**7,
    max_input_channel_block: int = 2**4,
    max_output_channel_block: int = 2**4,
    stride_range: int = 2**3,
):

    input_data_type = int_t(signed=True, bits=activation_bits)
    input_quantization = quantization(scale=UndefinedFloat(), zero_point=0)
    input = tensor(
        (
            axis("n", size=1),
            axis("c", size=UndefinedInt()),
            axis("w", size=UndefinedInt()),
        ),
        dtype=input_data_type,
        quantization=input_quantization,
    )

    weight_data_type = int_t(signed=True, bits=weight_bits)
    weight_quantization = quantization(scale=UndefinedFloat(), zero_point=0)

    weight = tensor(
        (
            axis("o", size=UndefinedInt()),
            axis("i", size=UndefinedInt()),
            axis("kw", size=IntRange(1, max_kernel_size)),
        ),
        dtype=weight_data_type,
        quantization=weight_quantization,
    )

    res_input = tensor(
        (
            axis("n", size=1),
            axis("c", size=UndefinedInt()),
            axis("w", size=UndefinedInt()),
        ),
        dtype=input_data_type,
        quantization=input_quantization,
    )

    conv = OpType("conv1d", input, weight, stride=stride_range)

    accumulator_data_type = int_t(signed=True, bits=accumulator_bits)
    accumulator_quantization = quantization(scale=UndefinedFloat(), zero_point=0)
    quant_conv = requantize(
        conv, dtype=accumulator_data_type, quantization=accumulator_quantization
    )

    bias_data_type = int_t(signed=True, bits=bias_bits)
    bias_quantization = quantization(scale=UndefinedFloat(), zero_point=0)
    bias = tensor(
        (axis("c", size=UndefinedInt()),),
        dtype=bias_data_type,
        quantization=bias_quantization,
    )

    bias_add = optional(
        add(quant_conv, broadcast(bias, axis=((axis("n"))))), quant_conv
    )  # FIXME: define broadcasting
    res_add = optional(add(bias_add, res_input), bias_add)
    pool = optional(avg_pool(res_add), res_add)
    activation = optional(relu(pool), pool)
    requantization = requantize(
        activation, dtype=input_data_type, quantization=input_quantization
    )

    return requantization


@parametrize
class Ultratrail(Device):
    def __init__(
        self,
        weight_bits: int = 6,
        bias_bits: int = 8,
        activation_bits: int = 8,
        accumulator_bits: int = 20,
        max_weight_bits: int = 8,
        rows: int = 8,
        cols: int = 8,
        ifmap_bits: int = 7,
        ic_block_bits: int = 4,
        oc_block_bits: int = 4,
        kernel_size_bits: int = 4,
        stride_bits: int = 3,
    ) -> None:
        super().__init__()

        self.weight_bits = weight_bits
        self.bias_bits = bias_bits
        self.activation_bits = activation_bits
        self.accumulator_bits = accumulator_bits
        self.max_weight_bits = max_weight_bits
        self.rows = rows
        self.cols = cols
        self.ifmap_bits = ifmap_bits
        self.ic_block_bits = ic_block_bits
        self.oc_block_bits = oc_block_bits
        self.kernel_size_bits = kernel_size_bits
        self.stride_bits = stride_bits

        max_kernel_size = 2**self.kernel_size_bits
        max_input_length = 2**self.ifmap_bits
        max_input_channel_block = self.rows * 2**self.ic_block_bits
        max_output_channel_block = self.cols * 2**self.oc_block_bits

        stride_range = IntRange(1, 2**2**stride_bits)

        # self.cond(stride <= 2**2**S_BIT and is_power_of_2(stride))

        op = ut_op(
            weight_bits=self.weight_bits,
            bias_bits=self.bias_bits,
            activation_bits=self.activation_bits,
            accumulator_bits=self.accumulator_bits,
            max_weight_bits=self.max_weight_bits,
            max_kernel_size=max_kernel_size,
            max_input_length=max_input_length,
            max_input_channel_block=max_input_channel_block,
            max_output_channel_block=max_output_channel_block,
            stride_range=stride_range,
        )
        self._ops.append(op)
