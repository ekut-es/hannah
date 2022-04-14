from typing import Any
from abc import ABC, abstractmethod


class Op(ABC):
    @abstractmethod
    def evaluate(self):
        ...

    @abstractmethod
    def format(self, indent=2, length=80) -> str:
        ...

    def _get_formatted(self, other: Any, indent: int, length: int) -> str:
        if isinstance(other, Expression):
            return other.format(indent, length)
        return str(other)

    def __str__(self) -> str:
        return self.format()
