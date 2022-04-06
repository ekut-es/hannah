from .memory_type import MemoryType
from .quantization_type import QuantizationType
from .data_type import DataType
from .axis_type import AxisType
from typing import Optional, Tuple


class TensorType:
    def __init__(
        self,
        axis: Tuple[AxisType, ...],
        dtype: DataType,
        quantization: Optional[QuantizationType] = None,
        memory: Optional[MemoryType] = None,
    ):
        self.axis = axis
        self.dtype = dtype
        self.quantization = quantization
        self.memory = memory

    def dim(self) -> int:
        return len(self.axis)

    def shape(self) -> Tuple[int, ...]:
        return tuple((ax.size for ax in self.axis))
