from typing import Dict, Any, MutableMapping, MutableSequence
from dataclasses import dataclass
from copy import deepcopy

from omegaconf import DictConfig

from .config import OptimConf, ScalarConfigSpec, ChoiceList

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


class Parameter:
    def _recurse(self, config, random_state):
        if isinstance(config, MutableSequence):
            return ChoiceParameter(config, random_state=random_state)
        elif isinstance(config, MutableMapping):
            try:
                config = ScalarConfigSpec(**config)
            except:
                try:
                    config = ChoiceList(**config)
                except:
                    config = config

            if isinstance(config, ScalarConfigSpec):
                res = IntervalParameter(config, random_state)
            elif isinstance(config, ChoiceList):
                res = ChoiceListParameter(config, random_state)
            else:
                res = SearchSpace(config, random_state)

            return res
        else:
            return config

    def mutations(self, config, index):
        # FIXME: here we would need to add child mutations
        def mutate_random(d):
            print("Warning: using mutate random for ", index)
            return nested_set(d, index, self.get_random())

        return [mutate_random]


class ChoiceParameter(Parameter):
    def __init__(self, config, random_state):
        self.choices = [self._recurse(c, random_state) for c in config]
        self.random_state = random_state

    def get_random(self):
        choice = self.random_state.choice(self.choices)

        if isinstance(choice, Parameter):
            return choice.get_random()
        elif isinstance(choice, MutableMapping):
            ret = {}
            for k, v in choice.items:
                if isinstance(v, Parameter):
                    ret[k] = v.get_random()
                else:
                    ret[k] = v
            return ret

        return choice

    # TODO: Add child mutations


class ChoiceListParameter(Parameter):
    def __init__(self, config, random_state):
        self.min = config.min
        self.max = config.max
        self.choices = [
            self._recurse(choice, random_state) for choice in config.choices
        ]
        self.random_state = random_state

    def _random_child(self):
        choice = self.random_state.choice(self.choices)
        if isinstance(choice, Parameter):
            choice = choice.get_random()
        elif isinstance(choice, MutableMapping):
            ret = {}
            for k, v in choice.items():
                if isinstance(v, Parameter):
                    ret[k] = v.get_random()
                else:
                    ret[k] = v
            choice = ret
        return choice

    def get_random(self):
        length = self.random_state.randint(self.min, self.max)
        result = []
        for _ in range(length):
            choice = self._random_child()
            result.append(choice)

        return result

    def mutations(self, config, index):

        length = len(config)

        def drop_random(d):
            print("Dropping random element", index)
            idx = self.random_state.randint(low=0, high=length)
            l = nested_get(d, index)
            l.pop(idx)

        def add_random(d):
            print("adding random element", index)
            num = self.random_state.randint(low=0, high=length + 1)
            choice = self._random_child()
            l = nested_get(d, index)
            l.insert(num, choice)

        mutations = []
        if length < self.max:
            mutations.append(add_random)
        if length > self.min:
            mutations.append(drop_random)

        # Create mutations for all children
        for num, child in enumerate(config):
            print("Child", child)
            child_keys = child.keys()
            for choice in self.choices:
                choice_keys = choice.space.keys()

                # FIXME: also compare values
                if choice_keys == child_keys:
                    print("Get child mutations")
                    child_index = index + (num,)
                    if isinstance(choice, Parameter):
                        mutations.extend(choice.mutations(child, child_index))

        return mutations


class IntervalParameter(Parameter):
    def __init__(self, config, random_state):
        self.config = config
        self.random_state = random_state

    def get_random(self):
        return self.random_state.random()


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
                config[k] = v

        return config

    def mutations(self, config, index):
        mutations = []
        for k, v in config.items():
            child_index = index + (k,)
            if isinstance(self.space[k], Parameter):
                mutations.extend(self.space[k].mutations(v, child_index))

        return mutations

    def mutate(self, config):
        print("mutate")
        config = deepcopy(config)

        mutations = self.mutations(config, tuple())

        mutation = self.random_state.choice(mutations)
        mutation(config)

        return config

    def __str__(self):
        res = ""

        for k, v in self.space.items():
            res += str(k) + ":" + str(v) + " "

        return res


class FitnessFunction:
    def __init__(self, bounds, random_state):
        self.bounds = bounds
        self.random_state = random_state

    def __call__(self, values):
        lambdas = self.random_state.uniform(low=0.0, high=1.0, size=len(bounds))

        result = 0.0
        for num, key in enumerate(self.bounds.keys()):
            if key in values:
                result += np.power(lambdas[num] * (values[key] / self.bounds[key]), 2)
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

        while hash(repr(parametrization)) in self.visited_configs:
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
