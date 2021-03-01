from typing import Dict, Any, MutableMapping, MutableSequence
from dataclasses import dataclass
from copy import deepcopy

from .config import Scalar, ChoiceList, Partition, Subset
from .utils import get_pareto_points


import numpy as np


def nested_set(d, index, val):
    for k in index[:-1]:
        d = d[k]
    d[index[-1]] = val


def nested_get(d, index):
    for k in index:
        d = d[k]
    return d


@dataclass
class EvolutionResult:
    parameters: Dict[str, Any]
    result: Dict[str, float]


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
                res = IntervalParameter(config, random_state)
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

        print("Partitions:", partitions)

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
            print("Dropping random element")
            idx = self.random_state.randint(low=0, high=length)
            print(idx, length)
            print(state.choices)
            print(state.value)
            state.value.pop(idx)
            state.choices.pop(idx)
            state.length -= 1

        def add_random():
            print("adding random element")
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
class IntervalParameterState(ParameterState):
    def flatten(self):
        return self.value


class IntervalParameter(Parameter):
    def __init__(self, config, random_state):
        self.config = config
        self.random_state = random_state

    def get_random(self):
        print(self.config)
        if self.config.lower and self.config.upper:
            if self.config.integer:
                return IntervalParameterState(
                    self.random_state.random_integers(
                        self.config.lower, self.config.upper
                    )
                )
            else:
                return IntervalParameterState(
                    self.random_state.uniform(self.config.lower, self.config.upper)
                )
        else:
            res = self.random_state.random()
            if self.config.integer:
                res = int(res)
            return IntervalParameterState(res)


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


class FitnessFunction:
    def __init__(self, bounds, random_state):
        self.bounds = bounds
        self.lambdas = random_state.uniform(low=0.0, high=1.0, size=len(self.bounds))

    def __call__(self, values):

        result = 0.0
        for num, key in enumerate(self.bounds.keys()):
            if key in values:
                result += np.power(
                    self.lambdas[num] * (values[key] / self.bounds[key]), 2
                )
        return np.sqrt(result)


class AgingEvolution:
    """Aging Evolution based multi objective optimization"""

    def __init__(
        self, population_size, sample_size, eps, bounds, parametrization, random_state
    ):
        self.population_size = population_size
        self.sample_size = sample_size
        self.eps = eps
        self.parametrization = SearchSpace(parametrization, random_state)

        self.random_state = random_state

        self.history = []
        self.population = []
        self.bounds = bounds
        self.visited_configs = set()

    def get_fitness_function(self):
        ff = FitnessFunction(self.bounds, self.random_state)

        return ff

    def next_parameters(self):
        "Returns a list of current tasks"

        parametrization = {}

        while (
            hash(repr(parametrization)) in self.visited_configs or not parametrization
        ):
            if len(self.history) < self.population_size:
                parametrization = self.parametrization.get_random()
            elif self.random_state.uniform() < self.eps:
                parametrization = self.parametrization.get_random()
            else:
                sample = self.random_state.choice(
                    self.population, size=self.sample_size
                )
                fitness_function = self.get_fitness_function()

                fitness = [fitness_function(x.result) for x in sample]

                parent = sample[np.argmin(fitness)]

                parametrization = self.parametrization.mutate(parent.parameters)

        self.visited_configs.add(hash(repr(parametrization)))

        return parametrization

    def tell_result(self, parameters, metrics):
        "Tell the result of a task"

        result = EvolutionResult(parameters, metrics)

        self.history.append(result)
        self.population.append(result)
        if len(self.population) > self.population_size:
            self.population.pop(0)

        return None

    def pareto_points(self):
        """ Get pareto optimal points discovered during search """

        # Build cost matrix
        costs = []
        for point in self.history:
            result = point.result
            costs.append(np.array(list(result.values())))

        costs = np.array(costs)

        pareto_indices = get_pareto_points(costs)

        result = []
        for num, index in enumerate(pareto_indices):
            if index:
                result.append(self.history[num])

        return result
