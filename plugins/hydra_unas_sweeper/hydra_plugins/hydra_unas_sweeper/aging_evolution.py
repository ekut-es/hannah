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
    def _recurse(self, config):
        print(config)
        # breakpoint()
        if isinstance(config, MutableSequence):
            return ChoiceParameter(config)
        elif isinstance(config, MutableMapping):
            try:
                config = ScalarConfigSpec(**config)
            except:
                try:
                    config = ChoiceList(**config)
                except:
                    pass

            if isinstance(config, ScalarConfigSpec):
                res = IntervalParameter(config)
            elif isinstance(config, ChoiceList):
                res = ChoiceListParameter(config)
            else:
                res = {}
                for k, v in config.items():
                    res[k] = self._recurse(v)

            return res
        else:
            return config

    def mutations(self, config, index):
        return []


class ChoiceParameter(Parameter):
    def __init__(self, config):
        self.choices = [self._recurse(c) for c in config]

    def get_random(self):
        choice = np.random.choice(self.choices)

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

    def mutations(self, config, index):
        # FIXME: here we would need to add child mutations
        def mutate_random(d):
            print("mutating: ", index)
            return nested_set(d, index, self.get_random())

        return [mutate_random]


class ChoiceListParameter(Parameter):
    def __init__(self, config):
        self.min = config.min
        self.max = config.max
        self.choices = [self._recurse(choice) for choice in config.choices]

    def _random_child(self):
        choice = np.random.choice(self.choices)
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

    def _child_mutations(self, choice, config, index):
        mutations = []
        if isinstance(choice, Parameter):
            choice.mutations(config, index)
        elif isinstance(choice, MutableMapping):
            for k, v in choice.items():
                param_index = index + (k,)
                mutations.extend(choice[k].mutations(config[k], param_index))

        return mutations

    def get_random(self):
        length = np.random.randint(self.min, self.max)
        result = []
        for _ in range(length):
            choice = self._random_child()
            result.append(choice)

        return result

    def mutations(self, config, index):

        length = len(config)

        def drop_random(d):
            print("Dropping random element", index)
            idx = np.random.randint(low=0, high=length)
            l = nested_get(d, index)
            l.pop(idx)

        def add_random(d):
            print("adding random element", index)
            num = np.random.randint(low=0, high=length + 1)
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
                choice_keys = choice.keys()

                # FIXME: also compare values
                if choice_keys == child_keys:
                    print("Get child mutations")
                    child_index = index + (num,)
                    mutations.extend(self._child_mutations(choice, child, child_index))

        return mutations


class IntervalParameter(Parameter):
    def __init__(self, config):
        self.config = config

    def get_random(self):
        return np.random.random()

    def mutations(self, config, index):

        print("mutations", index)

        def mutation(d):
            print("mutating: ", index)
            return nested_set(d, index, self.get_random())

        return [mutation]


class SearchSpace(Parameter):
    def __init__(self, config):
        self.config = config
        self.space = self._recurse(config)

    def get_random(self):
        config = {}

        for k, v in self.space.items():
            if isinstance(v, Parameter):
                config[k] = v.get_random()
            else:
                config[k] = v

        return config

    def mutate(self, config):
        print("mutate")
        config = deepcopy(config)

        mutations = []

        for k, v in config.items():
            index = (k,)
            print(self.space[k])
            mutations.extend(self.space[k].mutations(v, index))

        mutation = np.random.choice(mutations)
        mutation(config)
        return config

    def __str__(self):
        res = ""

        for k, v in self.space.items():
            res += str(k) + ":" + str(v) + " "

        return res


class FitnessFunction:
    def __init__(self, bounds):
        self.bounds = bounds
        self.lambdas = np.random.uniform(low=0.0, high=1.0, size=len(bounds))

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

    def __init__(self, population_size, sample_size, eps, bounds, parametrization):
        self.population_size = population_size
        self.sample_size = sample_size
        self.eps = eps

        print("parametrization:", parametrization)

        self.parametrization = SearchSpace(parametrization)

        print("SearchSpace:", self.parametrization)

        self.history = []
        self.population = []
        self.bounds = bounds

    def get_fitness_function(self):
        ff = FitnessFunction(self.bounds)

        return ff

    def next_parameters(self):
        "Returns a list of current tasks"

        if len(self.history) < self.population_size:
            return self.parametrization.get_random()

        if np.random.uniform() < self.eps:
            return self.parametrization.get_random()

        sample = np.random.choice(self.population, size=self.sample_size)
        fitness_function = self.get_fitness_function()

        fitness = [fitness_function(x.result) for x in sample]

        parent = sample[np.argmax(fitness)]

        child = self.parametrization.mutate(parent.parameters)

        return child

    def tell(self, parameters, metrics):
        "Tell the result of a task"

        result = EvolutionResult(parameters, metrics)

        self.history.append(result)
        self.population.append(result)
        if len(self.population) > self.population_size:
            self.population.pop(0)

        return None
