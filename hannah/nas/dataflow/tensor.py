from hannah.nas.dataflow.tensor_expression import TensorExpression


class Tensor(TensorExpression):
    def __init__(self, *operands, tensor_type=None, name="") -> None:
        super().__init__(*operands, tensor_type=tensor_type, name=name)

    def __repr__(self) -> str:
        return "Tensor({})".format(self.id)

    def __str__(self) -> str:
        return self.__repr__()
