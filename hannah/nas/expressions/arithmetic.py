from .op import BinaryOp, UnaryOp
from math import floor


class Add(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
        self._symbol = "+"

    def concrete_impl(self, lhs, rhs):
        return lhs + rhs


class Sub(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
        self._symbol = "-"

    def concrete_impl(self, lhs, rhs):
        return lhs - rhs


class Mul(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
        self._symbol = "*"

    def concrete_impl(self, lhs, rhs):
        return lhs * rhs


class Truediv(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
        self._symbol = "/"

    def concrete_impl(self, lhs, rhs):
        return lhs / rhs


class Floordiv(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
        self._symbol = "//"

    def concrete_impl(self, lhs, rhs):
        return lhs // rhs


class Mod(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)
        self._symbol = "mod"

    def concrete_impl(self, lhs, rhs):
        return lhs % rhs


class Floor(UnaryOp):
    def __init__(self, operand) -> None:
        super().__init__(operand)

    def concrete_impl(self, operand):
        return floor(operand)
