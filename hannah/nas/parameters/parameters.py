from typing import Optional, Union
import numpy as np
from abc import ABC, abstractmethod
from ..expressions.conditions import (
    LTCondition,
    LECondition,
    GTCondition,
    GECondition,
    EQCondition,
    NECondition,
)
from ..expressions.arithmetic import Add, Sub, Mul, Truediv, Floordiv, Mod
from ..expressions.logic import And, Or
from ..core.parametrized import is_parametrized
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
        self, min, max, rng: Optional[Union[np.random.Generator, int]] = None
    ) -> None:
        super().__init__(rng)
        self.min = min
        self.max = max
        self.current_value = min

    def sample(self):
        self.current_value = self.rng.integers(self.min, self.max)
        return self.current_value

    def instantiate(self):
        return self.current_value

    def check(self, value):
        if isinstance(value, int):
            return True
        else:
            print(
                "Value {} must be of type int but is type {}".format(value, type(value))
            )
            return False

    def set_current(self, value):
        if self.check(value):
            self.current_value = value


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

    def instantiate(self):
        return self.current_value

    def check(self, value):
        if not isinstance(value, float):
            print(
                "Value {} must be of type float but is of type {}".format(
                    value, type(value)
                )
            )
            return False
        elif value > self.max and value < self.min:
            print(
                "Value {} must be in range [{}, {}], ".format(value, self.min, self.max)
            )
            return False
        else:
            return True

    def set_current(self, value):
        if self.check(value):
            self.current_value = value


class CategoricalParameter(Parameter):
    def __init__(
        self, choices, rng: Optional[Union[np.random.Generator, int]] = None
    ) -> None:
        super().__init__(rng)
        self.choices = choices

    def sample(self):
        self.current_value = self.rng.choice(self.choices)
        if is_parametrized(self.current_value):
            self.current_value = self.current_value.sample()
        return self.current_value

    def instantiate(self):
        return self.current_value

    def check(self, value):
        if is_parametrized(value):
            if value in self.choices:
                return True
            else:
                print("Desired value {} not a valid choice".format(value))
        else:
            for choice in self.choices:
                if choice.check(value):
                    return True
        print("Desired value {} not realizable with the given choices".format(value))
        return False

    def set_current(self, value):
        if self.check(value):
            self.current_value = value


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
            if is_parametrized(element):
                element = element.sample()
            result.append(element)

        self.current_value = result
        return result

    def instantiate(self):
        return self.current_value

    def check(self, value):
        if not isinstance(value, list):
            print("Value for SubsetParameter must be list")
            return False
        elif len(value) > self.max or len(value) < self.min:
            print(
                "Size of subset ({}) not in supported range of [{},{}]".format(
                    len(value), self.min, self.max
                )
            )
            return False
        else:
            for v in value:
                if is_parametrized(v):
                    if v in self.choices:
                        return True
                    else:
                        print("Value {} not in choices".format(v))
                        return False
                else:
                    for choice in self.choices:
                        if choice.check(value):
                            return True
        return False

    def set_current(self, value):
        assert value in self.choices, "Desired value {} not a valid choice".format(
            value
        )
        self.current_value = value


def _create_parametrize_wrapper(parameters, cls):
    parameter_list = list(parameters.values())
    old_init_fn = cls.__init__

    def init_fn(self, *args, **kwargs):
        self._PARAMETERS = {}

        for num, arg in enumerate(args):
            if is_parametrized(arg):
                # breakpoint()
                name = parameter_list[num + 1].name
                self._PARAMETERS[name] = arg
        for name, arg in kwargs.items():
            if is_parametrized(arg):
                self._PARAMETERS[name] = arg

        # TODO:
        cls.sample = sample
        cls.set_params = set_params
        # cls.instantiate = instantiate
        self._parametrized = True
        old_init_fn(self, *args, **kwargs)

    return init_fn


def parametrize(cls=None):
    def parametrize_function(cls):
        init_fn = cls.__init__
        init_sig = inspect.signature(init_fn)

        new_init_fn = _create_parametrize_wrapper(init_sig.parameters, cls)
        cls.__init__ = new_init_fn

        return cls

    if cls:
        return parametrize_function(cls)

    return parametrize_function


def sample(self):
    for _key, param in self._PARAMETERS.items():
        param.sample()


def set_params(self, **kwargs):
    for key, value in kwargs.items():
        assert key in self._PARAMETERS, "{} has no parameter {}".format(self, key)
        self._PARAMETERS[key].set_current(value)


def instantiate(self):
    self._parametrized = False
