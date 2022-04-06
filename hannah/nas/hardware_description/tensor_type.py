from hannah.nas.hardware_description.memory_type import MemoryType
from hannah.nas.hardware_description.quantization_type import QuantizationType
from .data_type import DataType
from typing import Optional, Tuple


class CompressionType:
    def __init__(self) -> None:
        pass


class SparsityType:
    def __init__(self) -> None:
        pass


class AxisType:
    def __init__(self,
                 name : str,
                 size : Optional[int] = None,
                 compression : Optional[CompressionType] = None,
                 sparsity : Optional[SparsityType] = None):

        self.name = name
        self.size = size
        self.compression = compression

        self.sparsity = sparsity


class TensorType:
    def __init__(self, axis: AxisType, dtype: DataType, quantization: Optional[QuantizationType] = None, memory: Optional[MemoryType] = None):
        self.axis = axis
        self.dtype = dtype
        self.quantization = quantization
        self.memory = memory

    def dim(self) -> int:
        return len(self.axis)

    def shape(self) -> Tuple[int, ...]:
        return tuple((ax.size for ax in self.axis))
