from typing import Dict, Any
from dataclasses import dataclass

from .config import OptimConf

import numpy as np


@dataclass
class EvolutionResult:
    overrides: Dict[str, Any]
    result: float


class Parameter:
    def _recurse(self, config):
        if isinstance(config, list):
            return ChoiceParameter(config)
        elif isinstance(config, dict):
            res = {}
            for k, v in config.dict():
                res[k] = self._recurse(v)

            return res
        else:
            return config


class ChoiceParameter(Parameter):
    def __init__(self, config):
        self.choices = [self._recurse(c) for c in config]


class SearchSpace(Parameter):
    def __init__(config):
        self.config = config
        self.space = self._recurse(config)


class AgingEvolution:
    """Aging Evolution based multi objective optimization"""

    def __init__(self, population_size, sample_size, eps, bounds, parametrization):
        self.population_size = population_size
        self.sample_size = sample_size
        self.eps = eps

        print("parametrization:", parametrization)

        self.parametrization = parametrization

        self.history = []
        self.population = []
        self.bounds = bounds

    def get_fitness_funtion():
        return lambda x: x.result

    def next_parameters(self):
        "Returns a list of current tasks"

        if len(self.history) < self.population_size:
            return self.parametrization.get_random()

        if np.random.uniform() < self.eps:
            return self.parametrization.get_random()

        sample = np.random.choice(self.population, size=self.sample_size)
        parent = self.max(sample, self.get_fitness_function())

        child = self.parametrization.evolve(parent.parameters)

        return child

    def tell(parameters, metrics):
        "Tell the result of a task"

        result = EvolutionResult(task, parameters)

        self.history.append(result)
        self.population.append(result)
        if len(self.population) > self.population_size:
            self.population.pop(0)

        return None
