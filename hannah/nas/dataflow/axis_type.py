from hannah.nas.expressions.placeholder import UndefinedInt
from .compression_type import CompressionType
from typing import Optional
from copy import deepcopy


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

    def new(self, new_name=None):
        new_axis = deepcopy(self)
        if new_name:
            new_axis.name = new_name
        return new_axis
