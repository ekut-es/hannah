from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

T = TypeVar('T')

class Expression(ABC, Generic[T]):
    @abstractmethod
    def evaluate(self):
        ...

    @abstractmethod
    def format(self, indent=2, length=80) -> str:
        ...

    def __str__(self) -> str:
        return self.format()

    def __lt__(self, other):
        from ..expressions.conditions import LTCondition

        return LTCondition(self, other)

    def __le__(self, other):
        from ..expressions.conditions import LECondition

        return LECondition(self, other)

    def __gt__(self, other):
        from ..expressions.conditions import GTCondition

        return GTCondition(self, other)

    def __ge__(self, other):
        from ..expressions.conditions import GECondition

        return GECondition(self, other)

    def __eq__(self, other):
        from ..expressions.conditions import EQCondition

        return EQCondition(self, other)

    def __ne__(self, other):
        from ..expressions.conditions import NECondition

        return NECondition(self, other)

    def __add__(self, other):
        from ..expressions.arithmetic import Add

        return Add(self, other)

    def __sub__(self, other):
        from ..expressions.arithmetic import Sub

        return Sub(self, other)

    def __mul__(self, other):
        from ..expressions.arithmetic import Mul

        return Mul(self, other)

    def __truediv__(self, other):
        from ..expressions.arithmetic import Truediv

        return Truediv(self, other)

    def __floordiv__(self, other):
        from ..expressions.arithmetic import Floordiv

        return Floordiv(self, other)

    def __mod__(self, other):
        from ..expressions.arithmetic import Mod

        return Mod(self, other)

    def __divmod__(self, other):
        raise NotImplementedError()

    def __pow__(self, other):
        raise NotImplementedError()

    def __lshift__(self, other):
        raise NotImplementedError()

    def __rshift__(self, other):
        raise NotImplementedError()

    def __and__(self, other):
        from ..expressions.logic import And

        return And(self, other)

    def __xor__(self, other):
        raise NotImplementedError()

    def __or__(self, other):
        from ..expressions.logic import Or

        return Or(self, other)
