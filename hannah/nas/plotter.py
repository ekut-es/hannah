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
import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import yaml
from tabulate import tabulate

from .aging_evolution import EvolutionResult
from .plot import plot_history, plot_pareto_front
from .utils import is_pareto

logger = logging.getLogger()


def update_pareto_front(
    result: Union[EvolutionResult, List[EvolutionResult]]
) -> List[EvolutionResult]:

    if isinstance(result, list):
        pareto_points = result
    else:
        pareto_points = list(result)

    if len(pareto_points) == 1:
        return pareto_points

    costs = np.stack([x.costs() for x in pareto_points])
    is_efficient = is_pareto(costs, maximise=False)

    new_points = []
    for point, efficient in zip(pareto_points, is_efficient):
        if efficient:
            new_points.append(point)

    return new_points


def plot(history_file):
    bounds = {"val_error": 0.07}

    logger.info("Loading history file (%s)", str(history_file))
    history_file = Path(history_file)

    history = []
    with history_file.open("r") as history_data:
        history = yaml.unsafe_load(history_data)

    pruned_history = []
    for result in history:
        skip = False
        for name, bound in bounds.items():
            if name not in result.result:
                skip = True
            if result.result[name] > bound:
                skip = True

        if not skip:
            pruned_history.append(result)

    logger.info("Plotting history")
    plot_history(history, Path("."))

    logger.info("Plotting pareto points")
    plot_pareto_front(history, Path("."))

    pareto_points = update_pareto_front(history)

    print(f"Found {len(pareto_points)}  pareto-optimal solutions:")
    table = []
    headers = ["Num"] + list(list(pareto_points[0].result.keys()))
    for pareto_point in pareto_points:
        result = pareto_point.result
        table.append([pareto_point.index] + list(result.values()))

    print(tabulate(table, headers=headers))


def main():
    history_file = sys.argv[1]

    plot(history_file)


if __name__ == "__main__":
    main()
