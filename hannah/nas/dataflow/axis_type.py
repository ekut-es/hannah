from hannah.nas.dataflow.scoping_utils import get_id
from hannah.nas.expressions.placeholder import UndefinedInt
from .compression_type import CompressionType
from typing import Optional
from copy import deepcopy
from hannah.nas.parameters.parametrize import parametrize
from ..core.parametrized import is_parametrized


@parametrize
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

    def set_scope(self, current_scope, counters, visited):
        scope_id = get_id(current_scope, counters)
        self.id = f'{scope_id}.axis.{self.name}'
        self.set_param_scopes()


@parametrize
class AxisTuple:
    """Used to have the axis dict as a parametrized object
    """
    def __init__(self, *axis) -> None:
        self.axis = {}
        # reset parameters to improve naming
        self._PARAMETERS = {}
        for ax in axis:
            self.axis[ax.name] = ax
            if is_parametrized(ax):
                self._PARAMETERS[ax.name] = ax

    def set_scope(self, current_scope, counters, visited):
        scope_id = get_id(current_scope, counters)
        self.id = f'{scope_id}.axis'
        for _, ax in self.axis.items():
            ax.set_scope(current_scope, counters, visited)

    def __getitem__(self, key):
        return self.axis[key]

    def __repr__(self) -> str:
        return str(self.axis)
