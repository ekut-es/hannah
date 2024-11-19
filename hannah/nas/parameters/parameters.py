#
# Copyright (c) 2024 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import Any, Optional, Sequence, Union

import numpy as np

from ..core.expression import Expression
from ..core.parametrized import is_parametrized


class Parameter(Expression):
    def __init__(
        self,
        name: Optional[str] = "",
        rng: Optional[Union[np.random.Generator, int]] = None,
    ) -> None:
        super().__init__()
        self.setup_rng(rng)
        self.name = name
        self.id = None
        self._registered = False

    def get_children(self):
        return []

    def is_registered(self):
        return self._registered

    def register(self):
        self._registered = True

    @abstractmethod
    def sample(self): ...

    @abstractmethod
    def instantiate(self): ...

    @abstractmethod
    def set_current(self): ...

    @abstractmethod
    def check(self, value): ...

    @abstractmethod
    def from_float(self, value): ...

    # FIXME: evaluate and instantiate?
    def evaluate(self):
        return self.instantiate()

    def parametrization(self):
        return self

    def new(self, rng=None):
        new_param = deepcopy(self)
        new_param.setup_rng(rng)
        return new_param

    def setup_rng(self, rng):
        if rng is None:
            self.rng = np.random.default_rng(seed=None)
        elif isinstance(rng, int):
            self.rng = np.random.default_rng(seed=rng)
        elif isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            raise Exception("rng should be either np.random.Generator or int (or None)")

    def format(self, indent=2, length=80) -> str:
        return repr(self)

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
        step_size: int = 1,
        name: Optional[str] = "",
        rng: Optional[Union[np.random.Generator, int]] = None,
    ) -> None:
        super().__init__(name, rng)
        self.min = min
        self.max = max
        self.step_size = step_size
        self.current_value = self.evaluate_field("min")

    def evaluate_field(self, field_str):
        field = getattr(self, field_str)
        if isinstance(field, Parameter):
            return int(field.instantiate())
        elif isinstance(field, int):
            return field
        else:
            raise TypeError(
                "{} has an unsupported type for evaluation({})".format(
                    field_str, type(field)
                )
            )

    def get_bounds(self):
        return (self.evaluate_field("min"), self.evaluate_field("max"))

    def sample(self):
        min, max = self.get_bounds()
        values = np.arange(min, max + 1, self.step_size)
        # self.current_value = self.rng.integers(min, max+1)
        self.current_value = int(self.rng.choice(values))
        return self.current_value

    def instantiate(self):
        return self.current_value

    def check(self, value):
        if not isinstance(value, (int, np.int64)):
            raise ValueError(
                "Value {} must be of type int but is type {}".format(value, type(value))
            )
        elif value > self.max or value < self.min:
            raise ValueError(
                "Value {} is not in range [{}, {}], ".format(value, self.min, self.max)
            )

    def set_current(self, value):
        self.check(value)
        self.current_value = value

    def from_float(self, val):
        return int(val * (self.max - self.min) + self.min)


class FloatScalarParameter(Parameter):
    def __init__(
        self,
        min,
        max,
        name: Optional[str] = "",
        rng: Optional[Union[np.random.Generator, int]] = None,
    ) -> None:
        super().__init__(name, rng)
        self.min = float(min)
        self.max = float(max)
        self.current_value = self.min

    def sample(self):
        self.current_value = float(self.rng.uniform(self.min, self.max))
        return self.current_value

    def instantiate(self):
        return self.current_value

    def check(self, value):
        if not isinstance(value, float):
            ValueError(
                "Value {} must be of type float but is of type {}".format(
                    value, type(value)
                )
            )
        elif value > self.max or value < self.min:
            raise ValueError(
                "Value {} must be in range [{}, {}], ".format(value, self.min, self.max)
            )

    def set_current(self, value):
        self.check(value)
        self.current_value = value

    def from_float(self, val):
        return val * (self.max - self.min) + self.min


class CategoricalParameter(Parameter):
    def __init__(
        self,
        choices: Sequence[Any],
        name: Optional[str] = "",
        rng: Optional[Union[np.random.Generator, int]] = None,
    ) -> None:
        super().__init__(name, rng)
        self.choices = tuple(choices)
        self.sample()

    def sample(self):
        idx = int(self.rng.choice(range(len(self.choices))))
        self.current_value = self.choices[idx]
        if is_parametrized(self.current_value):
            self.current_value = self.current_value.sample()  # FIXME: This doesnt seem right? It should probably just call sample but not assign to SELF current value
        return self.current_value

    def instantiate(self):
        return self.current_value

    def check(self, value):
        if not is_parametrized(value):
            if value not in self.choices:
                raise ValueError("Desired value {} not a valid choice".format(value))
            else:
                return
        else:
            for choice in self.choices:
                if choice.check(value):
                    return
        raise ValueError(
            "Desired value {} not realizable with the given choices".format(value)
        )

    def set_current(self, value):
        self.check(value)
        self.current_value = value

    def __iter__(self):
        yield from iter(self.choices)

    def from_float(self, val):
        return self.choices[int(val * len(self.choices))]


class SubsetParameter(Parameter):
    def __init__(
        self,
        choices,
        min,
        max,
        name: Optional[str] = "",
        rng: Optional[Union[np.random.Generator, int]] = None,
    ) -> None:
        super().__init__(name, rng)
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
                "Size of subset ({}) not in supported range of [{},{}]".format(
                    len(value), self.min, self.max
                )
            )
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

    def from_float(self, val):
        raise NotImplementedError("SubsetParameter does not support from_float")
