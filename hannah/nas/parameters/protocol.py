from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class Parametrized(Protocol[T]):
    def sample(self) -> None:
        ...

    def instantiate(self) -> T:
        ...

    def set_current(self, value) -> None:
        ...

    def check(self, value) -> bool:
        ...


def is_parametrized(obj):
    return isinstance(obj, Parametrized)
