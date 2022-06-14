from hannah.nas.expressions.placeholder import UndefinedInt
from .compression_type import CompressionType
from typing import Optional


class AxisType:
    def __init__(
        self,
        name: str,
        size: Optional[int] = None,
        compression: Optional[CompressionType] = None,
    ):

        self.name = name
        if size is None:
            self.size = UndefinedInt()
        else:
            self.size = size
        self.compression = compression
