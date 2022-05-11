from typing import Protocol, TypeVar, runtime_checkable, Union, TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from ..parameters.parameters import Parameter

T = TypeVar("T")

ParameterTree =  Union['Parameter', Dict[str, 'ParameterTree']]

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

    def parameters(self) -> ParameterTree:
        ...


def is_parametrized(obj):
    return isinstance(obj, Parametrized)
