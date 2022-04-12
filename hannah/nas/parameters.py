from typing import Optional, Union
import numpy as np
from abc import ABC, abstractmethod
from .conditions import (
    LTCondition,
    LECondition,
    GTCondition,
    GECondition,
    EQCondition,
    NECondition,
)
from .abstract_arithmetic import (
    AbstractAdd,
    AbstractSub,
    AbstractMul,
    AbstractTruediv,
    AbstractFloordiv,
    AbstractMod,
    AbstractAnd,
    AbstractOr,
)
import inspect


class Parameter(ABC):
    def __init__(self, rng: Optional[Union[np.random.Generator, int]] = None) -> None:
        super().__init__()
        if rng is None:
            self.rng = np.random.default_rng(seed=None)
        elif isinstance(rng, int):
            self.rng = np.random.default_rng(seed=rng)
        elif isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            raise Exception("rng should be either np.random.Generator or int (or None)")

    @abstractmethod
    def sample(self):
        ...

    @abstractmethod
    def current(self):
        ...

    def __lt__(self, other):
        return LTCondition(self, other)

    def __le__(self, other):
        return LECondition(self, other)

    def __gt__(self, other):
        return GTCondition(self, other)

    def __ge__(self, other):
        return GECondition(self, other)

    def __eq__(self, other):
        return EQCondition(self, other)

    def __ne__(self, other):
        return NECondition(self, other)

    def __add__(self, other):
        return AbstractAdd(self, other)

    def __sub__(self, other):
        return AbstractSub(self, other)

    def __mul__(self, other):
        return AbstractMul(self, other)

    def __truediv__(self, other):
        return AbstractTruediv(self, other)

    def __floordiv__(self, other):
        return AbstractFloordiv(self, other)

    def __mod__(self, other):
        return AbstractMod(self, other)

    def __divmod__(self, other):
        raise NotImplementedError()

    def __pow__(self, other):
        raise NotImplementedError()

    def __lshift__(self, other):
        raise NotImplementedError()

    def __rshift__(self, other):
        raise NotImplementedError()

    def __and__(self, other):
        return AbstractAnd(self, other)

    def __xor__(self, other):
        raise NotImplementedError()

    def __or__(self, other):
        return AbstractOr(self, other)


class IntScalarParameter(Parameter):
    def __init__(
        self, min, max, rng: Optional[Union[np.random.Generator, int]] = None
    ) -> None:
        super().__init__(rng)
        self.min = min
        self.max = max
        self.current_value = min

    def sample(self):
        self.current_value = self.rng.randint(self.min, self.max)
        return self.current_value

    def current(self):
        return self.current_value


class FloatScalarParameter(Parameter):
    def __init__(
        self, min, max, rng: Optional[Union[np.random.Generator, int]] = None
    ) -> None:
        super().__init__(rng)
        self.min = min
        self.max = max
        self.current_value = min

    def sample(self):
        self.current_value = self.rng.uniform(self.min, self.max)
        return self.current_value

    def current(self):
        return self.current_value


class CategoricalParameter(Parameter):
    def __init__(
        self, choices, rng: Optional[Union[np.random.Generator, int]] = None
    ) -> None:
        super().__init__(rng)
        self.choices = choices

    def sample(self):
        self.current_value = self.rng.choice(self.choices)
        if isinstance(self.current_value, Parameter):
            self.current_value = self.current_value.sample()
        return self.current_value

    def current(self):
        return self.current_value


class SubsetParameter(Parameter):
    def __init__(
        self, choices, min, max, rng: Optional[Union[np.random.Generator, int]] = None
    ) -> None:
        super().__init__(rng)
        self.choices = choices
        self.min = min
        self.max = max

        self.current_value = None
        self.sample()

    def sample(self):
        size = self.rng.randint(min, max)
        chosen_set = self.rng.choice(self.choices, size=size)
        result = []
        for element in chosen_set:
            if isinstance(element, Parameter):
                element = element.sample()
            result.append(element)

        self.current_value = result
        return result

    def current(self):
        return self.current_value


def _create_parametrize_wrapper(parameters, cls):
    class _ComplexParameter(Parameter):
        def __init__(self, parameters, constants, cls):

            start_values = {(k, v.current()) for k, v in parameters.items()}

            self.current = cls(**start_values, **constants)

    parameter_list = list(parameters.values())

    old_init_fn = cls.__init__

    def init_fn(self, *args, **kwargs):
        self._PARAMETERS = {}

        for num, arg in enumerate(args):
            if isinstance(arg, Parameter):
                # breakpoint()
                name = parameter_list[num + 1].name
                self._PARAMETERS[name] = arg
        for name, arg in kwargs.items():
            if isinstance(arg, Parameter):
                self._PARAMETERS[name] = arg

        old_init_fn(self, *args, **kwargs)

    return init_fn


def parametrize(cls=None):
    def parametrize_function(cls):
        init_fn = cls.__init__
        init_sig = inspect.signature(init_fn)
        # for _key, param in init_sig.parameters.items():
        #    param_name = param.name
        #    param_type = param.annotation
        #    param_default = param.default
        #    print(_key, param)

        new_init_fn = _create_parametrize_wrapper(init_sig.parameters, cls)
        cls.__init__ = new_init_fn

        return cls

    if cls:
        return parametrize_function(cls)

    return parametrize_function
