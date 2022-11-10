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
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, MutableMapping, MutableSequence

from .config import ChoiceList, Partition, Scalar, Subset


def nested_set(d, index, val):
    for k in index[:-1]:
        d = d[k]
    d[index[-1]] = val


def nested_get(d, index):
    for k in index:
        d = d[k]
    return d


@dataclass
class ParameterState:
    value: Any

    def flatten(self):
        return self.value


class Parameter:
    def _recurse(self, config, random_state):
        if isinstance(config, MutableSequence):
            return ChoiceParameter(config, random_state=random_state)
        elif isinstance(config, MutableMapping):
            for config_class in [Scalar, ChoiceList, Partition, Subset]:
                try:
                    config = config_class(**config)
                except TypeError:
                    continue
                break

            if isinstance(config, Scalar):
                res = ScalarParameter(config, random_state)
            elif isinstance(config, ChoiceList):
                res = ChoiceListParameter(config, random_state)
            elif isinstance(config, Subset):
                res = SubsetParameter(config, random_state)
            elif isinstance(config, Partition):
                res = PartitionParameter(config, random_state)
            else:
                res = SearchSpace(config, random_state)

            return res
        else:
            return config

    def get_random(self):
        return ParameterState(None)

    def mutations(self, state):
        # FIXME: here we would need to add child mutations
        def mutate_random():
            print("Warning: using mutate random")
            value = self.get_random().value
            state.value = value

        return [mutate_random]


@dataclass
class SubsetParameterState(ParameterState):
    selection: MutableSequence[int]

    def flatten(self):
        return [v.flatten() for v in self.value]


class SubsetParameter(Parameter):
    def __init__(self, config, random_state):
        self.choices = config.choices
        self.size = config.size
        self.random_state = random_state

    def get_random(self):
        subset = self.random_state.choice(len(self.choices), size=self.size)
        subset_choices = []
        subset_selection = []
        for choice_idx in subset:
            choice = self.choices[int(choice_idx)]
            if isinstance(choice, Parameter):
                choice = choice.get_random()
            else:
                choice = ParameterState(choice)
            subset_choices.append(choice)
            subset_selection.append(choice_idx)

        return SubsetParameterState(subset_choices, subset_selection)


@dataclass
class PartitionParameterState(ParameterState):
    pass

    def flatten(self):
        res = []
        for v in self.value:
            res.append([e.flatten() for e in v])
        return res


class PartitionParameter(Parameter):
    def __init__(self, config, random_state):
        self.choices = config.choices
        self.partitions = config.partitions
        self.random_state = random_state

    def get_random(self):
        partitions = []
        for p in range(self.partitions):
            partitions.append([])
        for choice in self.choices:
            partition = self.random_state.randint(0, len(partitions))

            if isinstance(choice, Parameter):
                choice = choice.get_random()
            else:
                choice = ParameterState(choice)
            partitions[partition].append(choice)

        return PartitionParameterState(partitions)


@dataclass
class ChoiceParameterState(ParameterState):
    choice_num: int

    def flatten(self):
        return self.value.flatten()


class ChoiceParameter(Parameter):
    def __init__(self, config, random_state):
        self.choices = [self._recurse(c, random_state) for c in config]
        self.random_state = random_state

    def get_random(self):
        num_choices = len(self.choices)
        choice_num = self.random_state.randint(0, num_choices)
        choice = self.choices[choice_num]

        if isinstance(choice, Parameter):
            choice = choice.get_random()
        else:
            choice = ParameterState(choice)

        return ChoiceParameterState(choice, choice_num)

    def mutations(self, state):
        choice_num = state.choice_num

        def increase_choice():
            choice_num = state.choice_num + 1
            choice = self.choices[choice_num]

            if isinstance(choice, Parameter):
                choice = choice.get_random()
            else:
                choice = ParameterState(choice)

            state.value = choice
            state.choice_num = choice_num

        def decrease_choice():
            choice_num = state.choice_num - 1
            choice = self.choices[choice_num]

            if isinstance(choice, Parameter):
                choice = choice.get_random()
            else:
                choice = ParameterState(choice)

            state.value = choice
            state.choice_num = choice_num

        mutations = []
        if choice_num < len(self.choices) - 1:
            mutations.append(increase_choice)
        if choice_num > 0:
            mutations.append(decrease_choice)

        choice = self.choices[choice_num]
        if isinstance(choice, Parameter):
            mutations.extend(choice.mutations(state.value))

        return mutations


@dataclass
class ChoiceListParameterState(ParameterState):
    choices: MutableSequence[int]
    length: int

    def flatten(self):
        return [v.flatten() for v in self.value]


class ChoiceListParameter(Parameter):
    def __init__(self, config, random_state):
        self.min = config.min
        self.max = config.max
        self.choices = [
            self._recurse(choice, random_state) for choice in config.choices
        ]
        self.random_state = random_state

    def _random_child(self):
        num_choices = len(self.choices)
        choice_num = self.random_state.randint(0, num_choices)
        choice = self.choices[choice_num]
        if isinstance(choice, Parameter):
            value = choice.get_random()
        else:
            value = ParameterState(choice)

        return value, choice_num

    def get_random(self):
        length = self.random_state.randint(self.min, self.max)
        result = []
        choices = []
        for _ in range(length):
            value, choice = self._random_child()
            result.append(value)
            choices.append(choice)

        return ChoiceListParameterState(result, choices, length)

    def mutations(self, state: ChoiceListParameterState):

        length = state.length

        def drop_random():
            idx = self.random_state.randint(low=0, high=length)
            state.value.pop(idx)
            state.choices.pop(idx)
            state.length -= 1

        def add_random():
            num = self.random_state.randint(low=0, high=length + 1)
            value, choice = self._random_child()
            state.value.insert(num, value)
            state.choices.insert(num, choice)
            state.length += 1

        mutations = []
        if length < self.max:
            mutations.append(add_random)
        if length > self.min:
            mutations.append(drop_random)

        # Create mutations for all children
        for num, child in enumerate(state.value):
            choice_num = state.choices[num]
            choice = self.choices[choice_num]
            if isinstance(choice, Parameter):
                mutations.extend(choice.mutations(child))

        return mutations


@dataclass
class ScalarParameterState(ParameterState):
    sigma: float  # variance of sampling parameter

    def flatten(self):
        return self.value


class ScalarParameter(Parameter):
    def __init__(self, config, random_state):
        self.config = config
        self.random_state = random_state

    def get_random(self):
        if self.config.lower and self.config.upper:
            lower = self.config.lower
            upper = self.config.upper
            if self.config.log:
                lower = math.log2(lower)
                upper = math.log2(upper)
            if self.config.integer:
                value = int(self.random_state.random_integers(int(lower), int(upper)))
                if self.config.log:
                    value = 2**value
                return ScalarParameterState(value, (upper - lower) / 6.0)
            else:
                value = self.random_state.uniform(lower, upper)
                if self.config.log:
                    value = 2**value
                return ScalarParameterState(value, (upper - lower) / 6.0)
        else:
            res = self.random_state.random()
            if self.config.integer:
                res = int(res)
            if self.config.log:
                res = 2**res
            return ScalarParameterState(res, 1.0)

    def mutations(self, state):
        def mutate_scalar():
            value = state.value
            print(state.sigma)
            if self.config.log:
                value = math.log2(value)
            new_value = self.random_state.normal(value, state.sigma)
            if self.config.log:
                value = 2**new_value
            if self.config.lower is not None:
                if new_value < self.config.lower:
                    new_value = self.config.lower
            if self.config.upper is not None:
                if new_value > self.config.upper:
                    new_value = self.config.upper

            state.value = new_value

        return [mutate_scalar]


@dataclass
class SearchSpaceState(ParameterState):
    def flatten(self):
        res = {}
        for k, v in self.value.items():
            res[k] = v.flatten()

        return res


class SearchSpace(Parameter):
    def __init__(self, config, random_state):
        self.config = config
        self.random_state = random_state
        self.space = {k: self._recurse(v, self.random_state) for k, v in config.items()}

    def get_random(self):
        config = {}

        for k, v in self.space.items():
            if isinstance(v, Parameter):
                config[k] = v.get_random()
            else:
                config[k] = ParameterState(v)

        return SearchSpaceState(config)

    def mutations(self, state):
        mutations = []
        for k, v in state.value.items():
            if isinstance(self.space[k], Parameter):
                mutations.extend(self.space[k].mutations(v))

        return mutations

    def mutate(self, config):
        config = deepcopy(config)

        mutations = self.mutations(config)

        mutation = self.random_state.choice(mutations)
        mutation()

        return config

    def __str__(self):
        res = ""

        for k, v in self.space.items():
            res += str(k) + ":" + str(v) + " "

        return res
