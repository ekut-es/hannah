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
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import yaml

from .parametrization import SearchSpace
from .utils import is_pareto


@dataclass()
class EvolutionResult:
    index: int
    parameters: Dict[str, Any]
    result: Dict[str, float]

    def costs(self):
        return np.asarray(
            [float(self.result[k]) for k in sorted(self.result.keys())],
            dtype=np.float32,
        )


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
        self,
        parametrization,
        bounds,
        population_size=100,
        sample_size=10,
        eps=0.1,
        random_state=None,
        output_folder=".",
    ):
        self.parametrization = SearchSpace(parametrization, random_state)
        self.bounds = bounds

        self.population_size = population_size
        self.sample_size = sample_size
        self.eps = eps

        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        if random_state is None:
            self.random_state = np.random.RandomState()

        self.history = []
        self.population = []
        self._pareto_points = []
        self.output_folder = Path(output_folder)
        if (self.output_folder / "history.yml").exists():
            self.load()

    def get_fitness_function(self):
        ff = FitnessFunction(self.bounds, self.random_state)

        return ff

    def ask(self):
        return self.next_parameters()

    def next_parameters(self):
        "Returns a list of current tasks"

        parametrization = {}

        if len(self.history) < self.population_size:
            parametrization = self.parametrization.get_random()
        elif self.random_state.uniform() < self.eps:
            parametrization = self.parametrization.get_random()
        else:
            sample = self.random_state.choice(self.population, size=self.sample_size)
            fitness_function = self.get_fitness_function()

            fitness = [fitness_function(x.result) for x in sample]

            parent = sample[np.argmin(fitness)]

            parametrization = self.parametrization.mutate(parent.parameters)

        return parametrization

    def tell(self, parameters, metrics):
        return self.tell_result(parameters, metrics)

    def tell_result(self, parameters, metrics):
        "Tell the result of a task"

        result = EvolutionResult(len(self.history), parameters, metrics)

        self.history.append(result)
        self.population.append(result)
        if len(self.population) > self.population_size:
            self.population.pop(0)

        self.save()

        return None

    @property
    def pareto_points(self):
        self._update_pareto_front(self.history)

        return self._pareto_points

    def _update_pareto_front(
        self, result: Union[EvolutionResult, List[EvolutionResult]]
    ) -> None:
        if isinstance(result, list):
            self._pareto_points.extend(result)
        else:
            self._pareto_points.append(result)

        if len(self._pareto_points) == 1:
            return

        costs = np.stack([x.costs() for x in self._pareto_points])
        is_efficient = is_pareto(costs, maximise=False)

        new_points = []
        for point, efficient in zip(self._pareto_points, is_efficient):
            if efficient:
                new_points.append(point)

        self._pareto_points = new_points

    def save(self):
        history_file = self.output_folder / "history.yml"
        history_file_tmp = history_file.with_suffix(".tmp")

        with history_file_tmp.open("w") as history_data:
            yaml.dump(self.history, history_data)
        shutil.move(history_file_tmp, history_file)

    def load(self):
        history_file = self.output_folder / "history.yml"

        self.history = []
        self.population = []

        with history_file.open("r") as history_data:
            self.history = yaml.unsafe_load(history_data)

        if len(self.history) > self.population_size:
            self.population = self.history[len(self.history) - self.population_size :]
        else:
            self.population = self.history

        logging.info("Loaded %d points from history", len(self.history))
