import textwrap
from abc import ABC, abstractmethod
from typing import Any

from ..core.expression import Expression
from ..core.parametrized import is_parametrized


class Op(Expression):
    @abstractmethod
    def evaluate(self):
        ...

    @abstractmethod
    def format(self, indent=2, length=80) -> str:
        ...

    def _evaluate_operand(self, current_lhs):
        if is_parametrized(current_lhs):
            current_lhs = current_lhs.current_value
        elif isinstance(current_lhs, Op):
            current_lhs = current_lhs.evaluate()
        return current_lhs

    def _format_operand(self, other: Any, indent: int, length: int) -> str:
        if isinstance(other, Op):
            return other.format(indent, length)
        return str(other)


class BinaryOp(Op):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
        self.symbol = type(self).__name__

    def format(self, indent=2, length=80):
        ret = "(" + self.symbol + "\n"
        ret += (
            textwrap.indent(
                self._format_operand(self.lhs, indent, length), " " * indent
            )
            + "\n"
        )
        ret += (
            textwrap.indent(
                self._format_operand(self.rhs, indent, length), " " * indent
            )
            + "\n"
        )
        ret += ")"
        return ret

    def __repr__(self):
        return type(self).__name__ + "(" + repr(self.lhs) + "," + repr(self.rhs) + ")"

    def evaluate(self):
        current_lhs = self.lhs
        current_rhs = self.rhs

        current_lhs = self._evaluate_operand(current_lhs)
        current_rhs = self._evaluate_operand(current_rhs)

        return self.concrete_impl(current_lhs, current_rhs)

    @abstractmethod
    def concrete_impl(self, lhs, rhs):
        ...


class UnaryOp(Op):
    def __init__(self, operand) -> None:
        super().__init__()
        self.operand = operand
        self.symbol = type(self).__name__

    def evaluate(self):
        current_operand = self._evaluate_operand(self.operand)
        return self.concrete_impl(current_operand)

    @abstractmethod
    def concrete_impl(self, operand):
        ...

    def format(self, indent=2, length=80):
        ret = "(" + self.symbol
        ret += " " + self._format_operand(self.operand) + ")"
        return ret
