from ..hardware_description.memory_type import MemoryType
from .quantization_type import QuantizationType
from .data_type import DataType
from .axis_type import AxisType
from typing import Optional, Tuple
from hannah.nas.dataflow.dataflow_utils import reset_nested_counters


class TensorType:
    def __init__(
        self,
        axis: Tuple[AxisType, ...],
        dtype: DataType,
        quantization: Optional[QuantizationType] = None,
        memory: Optional[MemoryType] = None,
        name: str = "",
    ):
        self.axis = {}
        for ax in axis:
            self.axis[ax.name] = ax
        self.dtype = dtype
        self.quantization = quantization
        self.memory = memory
        self.name = name
        self.id = name

    def dim(self) -> int:
        return len(self.axis)

    def output_tensor(self):
        return self

    def get_hierarchical_dict(self, hierarchy_dict, current_scope, inputs, scopes, input_names, tensors):
        if self in inputs:
            for i in inputs[self]:
                current_scope.remove(scopes[i])

    def insert_scope_to_id(self, inputs, scopes, current_scope, scope_counters, nested_scopes):
        self.id = ".".join(current_scope) + ".{}".format(self.name)
        if self in inputs:
            for i in inputs[self]:
                current_scope.remove(scopes[i])
                reset_nested_counters(scopes[i], nested_scopes, scope_counters)

    def shape(self) -> Tuple[int, ...]:
        return tuple((ax.size for ax in self.axis.values))

    def __repr__(self) -> str:
        return 'Tensor(name=' + self.name + ", axis=(" + ' '.join(['{}, '.format(a) for a in self.axis.keys()]) + '))'
