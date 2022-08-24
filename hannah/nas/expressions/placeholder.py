from typing import Optional

from ..core.expression import Expression


class Placeholder(Expression):
    def __init__(self, id: Optional[str] = None):
        self.id = id
        self._conditions = []

    def evaluate(self):
        raise NotImplementedError()

    def format(self, indent=2, length=80) -> str:
        return self.__class__.__name__ + "()"


class UndefinedInt(Placeholder):
    def __init__(self, id: Optional[str] = None) -> None:
        super().__init__(id)


# TODO:
class UndefinedFloat(Placeholder):
    def __init__(self, id: Optional[str] = None) -> None:
        super().__init__(id)


class DefaultInt(Placeholder):
    def __init__(self, value: int, id: Optional[str] = None) -> None:
        super().__init__(id)
        self.value = value

    def evaluate(self):
        return self.value

    def format(self, indent=2, length=80) -> str:
        return self.__class__.__name__ + "({})".format(self.value)

    def __repr__(self) -> str:
        return str(self.value)


class DefaultFloat(Placeholder):
    def __init__(self, value: float, id: Optional[str] = None) -> None:
        super().__init__(id)
        self.value = value


class IntRange(Placeholder):
    def __init__(self, min: int, max: int, id: Optional[str] = None) -> None:
        super().__init__(id)
        self.min = min
        self.max = max
        # TODO: self.value = ??


class FloatRange(Placeholder):
    def __init__(self, lower: int, upper: int, id: Optional[str] = None):
        super().__init__(id)
        self.lower = lower
        self.upper = upper


class Categorical(Placeholder):
    def __init__(self, id: Optional[str] = None):
        super().__init__(id)
