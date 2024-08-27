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
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
from typing import Any, Dict

import numpy as np
import pymoo.algorithms
import pymoo.algorithms.moo
import pymoo.algorithms.moo.nsga3
import yaml

from hannah.nas.parameters.parametrize import set_parametrization
from .base_sampler import Sampler, SearchResult

from hannah.nas.search.utils import np_to_primitive

msglogger = logging.getLogger(__name__)

import pymoo

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3

from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem

_ALGORITHMS = {
    "nsga2": NSGA2,
    "nsga3": NSGA3,
}


def _transform_parameters(parameters, pop):
    results = []
    for ind in pop:
        result = {}
        for i, (name, param) in enumerate(parameters.items()):
            result[name] = param.from_float(ind.X[i])
        results.append(result)

    return results


class PyMOOSampler(Sampler):
    def __init__(
        self,
        parent_config,
        parametrization,
        output_folder=".",
        algorithm="nsga2",
        **kwargs,
    ) -> None:
        super().__init__(parent_config=parent_config, output_folder=output_folder)

        self.parametrization = parametrization

        if algorithm not in _ALGORITHMS:
            raise ValueError(
                f"Search algorithm {algorithm} not supported. Supported algorithms: {list(_ALGORITHMS.keys())}"
            )

        self._n_var = len(parametrization)
        self._n_obj = 1

        problem: Problem = Problem(
            n_var=self._n_var,
            n_obj=1,
            n_constr=0,
            xl=np.zeros(self._n_var),
            xu=np.ones(self._n_var),
        )
        termination = NoTermination()

        self.algorithm = _ALGORITHMS[algorithm](**kwargs)
        self.algorithm.setup(problem, termination=NoTermination())

    def next_parameters(self):
        pop = self.algorithm.ask()

        parameters = _transform_parameters(self.parametrization, pop)

        return parameters, []

    def tell(self, parameters, metrics):
        return self.tell_result(parameters, metrics)

    def tell_result(self, parameters, metrics):
        "Tell the result of a task"

        parameters = np_to_primitive(parameters)
        result = SearchResult(len(self.history), parameters, metrics)
        self.history.append(result)
        self.save()
        return None

    def save(self):
        history_file = self.output_folder / "history.yml"
        history_file_tmp = history_file.with_suffix(".tmp")

        with history_file_tmp.open("w") as history_data:
            yaml.dump(self.history, history_data)
        shutil.move(history_file_tmp, history_file)
        msglogger.info(f"Updated {history_file.name}")

    def load(self):
        history_file = self.output_folder / "history.yml"
        self.history = []
        with history_file.open("r") as history_data:
            self.history = yaml.unsafe_load(history_data)

        msglogger.info("Loaded %d points from history", len(self.history))
