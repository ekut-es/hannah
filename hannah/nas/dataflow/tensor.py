from copy import deepcopy
from typing import List
from hannah.nas.dataflow.scoping_utils import get_id_and_update_counters, update_scope
from hannah.nas.dataflow.tensor_expression import TensorExpression
from hannah.nas.dataflow.tensor_type import TensorType


class Tensor(TensorExpression):
    def __init__(self, *operands, tensor_type : TensorType = None, name : str = "") -> None:
        super().__init__(*operands, tensor_type=tensor_type, name=name)

    def set_scope(self, current_scope, counters, visited):
        current_scope = update_scope(self, current_scope)
        scope_id = get_id_and_update_counters(current_scope, counters)
        self.id = scope_id

    def output_tensor(self):
        return self

    def parameters(self):
        return self.tensor_type.parameters()

    def new(self):
        new_tensor = deepcopy(self)
        return new_tensor

    @property
    def dim(self):
        return self.tensor_type.dim()

    def __repr__(self) -> str:
        return "Tensor({})".format(self.id)

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, key):
        return self.tensor_type.axis[key]


class TensorTuple(TensorExpression):
    def __init__(self, tensors : List[Tensor], name: str = ""):
        super().__init__(name=name)
        self.tensors = tensors
        self.name = name
