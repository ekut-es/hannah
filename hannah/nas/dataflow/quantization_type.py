from typing import Optional

from .axis_type import AxisType


class QuantizationType:
    def __init__(
        self,
        axis: Optional[AxisType] = None,
        scale: Optional[float] = None,
        zero_point: Optional[float] = None,
    ) -> None:
        self.axis = axis
        self.scale = scale
        self.zero_point = zero_point
