from __future__ import annotations
from abc import ABC, abstractmethod
from ast import Param
from typing import Optional, Type, Union, get_type_hints, get_args

import numpy as np

from ..core.parametrized import is_parametrized
from ..expressions.arithmetic import Add, Floordiv, Mod, Mul, Sub, Truediv
from ..expressions.conditions import (
    EQCondition,
    GECondition,
    GTCondition,
    LECondition,
    LTCondition,
    NECondition,
)
from ..expressions.logic import And, Or


class Parameter(ABC):
    def __init__(self, scope: Optional[str] = None, rng: Optional[Union[np.random.Generator, int]] = None) -> None:
        super().__init__()
        if rng is None:
            self.rng = np.random.default_rng(seed=None)
        elif isinstance(rng, int):
            self.rng = np.random.default_rng(seed=rng)
        elif isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            raise Exception("rng should be either np.random.Generator or int (or None)")
        self.scope = scope

    @abstractmethod
    def sample(self):
        ...

    @abstractmethod
    def instantiate(self):
        ...

    @abstractmethod
    def set_current(self):
        ...

    @abstractmethod
    def check(self, value):
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
        return Add(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __truediv__(self, other):
        return Truediv(self, other)

    def __floordiv__(self, other):
        return Floordiv(self, other)

    def __mod__(self, other):
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
        return And(self, other)

    def __xor__(self, other):
        raise NotImplementedError()

    def __or__(self, other):
        return Or(self, other)

    def __repr__(self):
        return (
            type(self).__name__
            + "("
            + ", ".join((f"{k} = {v}" for k, v in self.__dict__.items()))
            + ")"
        )


class IntScalarParameter(Parameter):
    def __init__(
        self,
        min: Union[int, IntScalarParameter],
        max: Union[int, IntScalarParameter],
        scope: Optional[str] = None,
        rng: Optional[Union[np.random.Generator, int]] = None
    ) -> None:
        super().__init__(scope, rng)
        self.min = min
        self.max = max
        self.current_value = self.evaluate_field('min')

    def evaluate_field(self, field_str):
        field = getattr(self, field_str)
        if isinstance(field, Parameter):
            return self.min.instantiate()
        elif isinstance(field, int):
            return field
        else:
            raise TypeError("{} has an unsupported type for evaluation({})".format(field_str, type(field)))

    def get_bounds(self):
        return (self.evaluate_field('min'), self.evaluate_field('max'))

    def sample(self):
        self.current_value = self.rng.integers(*self.get_bounds())
        return self.current_value

    def instantiate(self):
        return self.current_value

    def check(self, value):
        if not isinstance(value, int):
            raise ValueError("Value {} must be of type int but is type {}".format(value, type(value)))
        elif value > self.max or value < self.min:
            raise ValueError("Value {} must be in range [{}, {}], ".format(value, self.min, self.max))

    def set_current(self, value):
        self.check(value)
        self.current_value = value


class FloatScalarParameter(Parameter):
    def __init__(
        self, min, max, scope: Optional[str] = None, rng: Optional[Union[np.random.Generator, int]] = None
    ) -> None:
        super().__init__(scope, rng)
        self.min = min
        self.max = max
        self.current_value = min

    def sample(self):
        self.current_value = self.rng.uniform(self.min, self.max)
        return self.current_value

    def instantiate(self):
        return self.current_value

    def check(self, value):
        if not isinstance(value, float):
            ValueError("Value {} must be of type float but is of type {}".format(value, type(value)))
        elif value > self.max or value < self.min:
            raise ValueError("Value {} must be in range [{}, {}], ".format(value, self.min, self.max))

    def set_current(self, value):
        self.check(value)
        self.current_value = value


class CategoricalParameter(Parameter):
    def __init__(
        self, choices, scope: Optional[str] = None, rng: Optional[Union[np.random.Generator, int]] = None
    ) -> None:
        super().__init__(scope, rng)
        self.choices = choices
        self.sample()

    def sample(self):
        self.current_value = self.rng.choice(self.choices)
        if is_parametrized(self.current_value):
            self.current_value = self.current_value.sample()
        return self.current_value

    def instantiate(self):
        return self.current_value

    def check(self, value):
        if is_parametrized(value):
            if value not in self.choices:
                raise ValueError("Desired value {} not a valid choice".format(value))
        else:
            for choice in self.choices:
                if choice.check(value):
                    return
        raise ValueError("Desired value {} not realizable with the given choices".format(value))

    def set_current(self, value):
        self.check(value)
        self.current_value = value


class SubsetParameter(Parameter):
    def __init__(
        self, choices, min, max, scope: Optional[str] = None, rng: Optional[Union[np.random.Generator, int]] = None
    ) -> None:
        super().__init__(scope, rng)
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
            if is_parametrized(element):
                element = element.sample()
            result.append(element)

        self.current_value = result
        return result

    def instantiate(self):
        return self.current_value

    def check(self, value):
        if not isinstance(value, list):
            raise ValueError("Value for SubsetParameter must be list")
        elif len(value) > self.max or len(value) < self.min:
            raise ValueError(
                "Size of subset ({}) not in supported range of [{},{}]".format(len(value), self.min, self.max))
        else:
            for v in value:
                if is_parametrized(v):
                    if v not in self.choices:
                        raise ValueError("Value {} not in choices".format(v))
                else:
                    for choice in self.choices:
                        choice.check(value)

    def set_current(self, value):
        self.check(value)
        self.current_value = value
