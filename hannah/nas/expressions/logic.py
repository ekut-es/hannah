from .op import BinaryOp


class And(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
        self._symbol = "and"

    def concrete_impl(self, lhs, rhs):
        # Fixme this does not follow python semantacics exactly
        return lhs and rhs


class Or(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
        self._symbol = "or"

    def concrete_impl(self, lhs, rhs):
        # Fixme this does not follow python semantacics exactly
        return lhs or rhs
