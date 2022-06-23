from hannah.nas.dataflow.tensor_expression import TensorExpression


class Tensor(TensorExpression):
    def __init__(self, *operands, tensor_type=None, name="") -> None:
        super().__init__(*operands, tensor_type=tensor_type, name=name)

    def output_tensor(self):
        return self

    def __repr__(self) -> str:
        return "Tensor({})".format(self.id)

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, key):
        return self.tensor_type.axis[key]
