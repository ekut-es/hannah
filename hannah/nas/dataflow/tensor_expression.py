class TensorExpression:
    def __init__(self, *operands, tensor_type=None, name="") -> None:
        self.operands = operands
        self._tensor_type = tensor_type
        self.users = []
        self.name = name
        self.id = name

    def set_id(self, id):
        self.id = id

    def tensor_type(self):
        assert self._tensor_type is not None, "Tensor Type has not been set, please run shape inference"
        return self._tensor_type
