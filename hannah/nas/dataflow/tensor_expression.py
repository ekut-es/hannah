

class TensorExpression:
    def __init__(self, *operands, tensor_type=None, name="") -> None:
        self.operands = operands
        self.tensor_type = tensor_type
        self.users = []
        self.name = name
        self.id = name

    def set_id(self, id):
        self.id = id
