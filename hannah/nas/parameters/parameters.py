#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
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
from typing import Optional, Union

import numpy as np

from ..core.expression import Expression
from ..core.parametrized import is_parametrized


class Parameter(Expression):
    def __init__(
        self,
        name: Optional[str] = None,
        rng: Optional[Union[np.random.Generator, int]] = None,
    ) -> None:
        super().__init__()
        if rng is None:
            self.rng = np.random.default_rng(seed=None)
        elif isinstance(rng, int):
            self.rng = np.random.default_rng(seed=rng)
        elif isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            raise Exception("rng should be either np.random.Generator or int (or None)")
        self.name = name
        self.id = None

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

    # FIXME: evaluate and instantiate?
    def evaluate(self):
        return self.instantiate()

    def parametrization(self):
        return self

    def new(self):
        return deepcopy(self)

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
        name: Optional[str] = None,
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
            return self.min.instantiate()
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
        values = np.arange(min, max + 1 , self.step_size)
        # self.current_value = self.rng.integers(min, max+1)
        self.current_value = self.rng.choice(values)
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


class FloatScalarParameter(Parameter):
    def __init__(
        self,
        min,
        max,
        name: Optional[str] = None,
        rng: Optional[Union[np.random.Generator, int]] = None,
    ) -> None:
        super().__init__(name, rng)
        self.min = float(min)
        self.max = float(max)
        self.current_value = self.min

    def sample(self):
        self.current_value = self.rng.uniform(self.min, self.max)
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


class CategoricalParameter(Parameter):
    def __init__(
        self,
        choices,
        name: Optional[str] = None,
        rng: Optional[Union[np.random.Generator, int]] = None,
    ) -> None:
        super().__init__(name, rng)
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


class SubsetParameter(Parameter):
    def __init__(
        self,
        choices,
        min,
        max,
        name: Optional[str] = None,
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
