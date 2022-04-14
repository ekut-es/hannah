import textwrap
from abc import ABC, abstractmethod
from typing import Any

from hannah.nas.parameters.protocol import is_parametrized
from .op import BinaryOp


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
