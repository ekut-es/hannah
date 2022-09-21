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
from dataclasses import dataclass
from typing import Any, Callable, List, MutableMapping, Tuple

import libsvm.svmutil as svmutil
import matplotlib.pyplot as plt
import numpy as np
from numpy import random


@dataclass
class ParameterRange:
    start: float
    end: float

    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


@dataclass
class Parameter:
    range: ParameterRange
    uuid: str
    catuuid: str
    pos: int

    def __init__(self, range: ParameterRange, uuid: str, catuuid: str, pos: int):
        self.range = range
        self.uuid = uuid
        self.catuuid = catuuid
        self.pos = pos


@dataclass
class Opts:
    runs: int
    waterlevel: float
    samples: int
    svm_params: str
    parameters: List[Any]
    active: bool
    augmentation_conf: MutableMapping[str, Any]
    best_path: str
    sample_fun: Callable[[Any, int], List[List[float]]]
    dut_fun: Callable[[List[float]], float]

    def __init__(self, parameters, runs, augmentation_conf, best_path):
        self.parameters = parameters
        self.runs = runs
        self.augmentation_conf = augmentation_conf
        self.best_path = best_path
        self.waterlevel = 1
        self.samples = 1000
        self.svm_params = "-s 4"
        self.active = False


class Bordersearch:
    def __int__(self):
        return

    def get_random_value(self, range: ParameterRange) -> float:
        return (
            range.start
            if range.start == range.end
            else random.uniform(range.start, range.end)
        )

    def get_random_param(self, opts: Opts) -> List[float]:
        dims = len(opts.parameters)
        ret = list()

        for j in range(dims):
            ret.append(self.get_random_value(opts.parameters[j].range))

        return ret

    def normalize(self, opts: Opts, values: List[float]) -> List[List[float]]:
        result = list()

        for i in range(len(values)):
            param = list()
            for j in range(len(opts.parameters)):
                min = opts.parameters[j].range.start
                max = opts.parameters[j].range.end
                col = values[i][j]
                param.append((col - min) / (max - min))
            result.append(param)

        return result

    def svm_train(
        self,
        opts: Opts,
        labels: List[int],
        data: List[Tuple[List[float], float]],
        parameter: Parameter,
    ):
        problem = svmutil.svm_problem(labels, [i[0] for i in data])

        return svmutil.svm_train(problem, parameter)

    def calc_above_wl(self, data, waterlevel) -> List[float]:
        return [1 if i[1] > waterlevel else 0 for i in data]

    def find_waterlevel(
        self, opts: Opts, waterlevel: float
    ) -> Tuple[List[float], float]:
        n_candidates_target = opts.samples
        svm_cost = 100
        weight_lambda = 20
        refinement_level = 0
        idx = 0

        conf = list()

        for i in range(opts.runs):
            progress = (float)(i + 1) / opts.runs
            size = 50
            out = "\r["
            out += str(["#" for _ in range(int(progress * size))])
            out += str([" " for _ in range(size)])
            out += str("]" + str((i + 1)) + ("/") + str(opts.runs) + (" "))
            print(out.join(""))

            if i == 0:
                # first run, generate some bootsrap data
                point = self.get_random_param(opts)
                z = opts.dut_fun(opts, point)
                conf.append((point, z))
                continue

            known = conf
            above_wl = self.calc_above_wl(known, waterlevel)
            parameter = svmutil.svm_parameter(
                "-s 0 -t 2 -d 3 -g 0.5 -r 0 -c " + str(svm_cost) + " -b 1 -m 20 -e 0.1"
            )

            # train model using previous sampled data as ground truth
            normalized_known = self.normalize(opts, [i[0] for i in known])
            normalized_known = [
                (param, z) for param, z in zip(normalized_known, [i[1] for i in known])
            ]
            model = self.svm_train(opts, above_wl, normalized_known, parameter)

            # generate new sampels
            n_candidates = 0
            candidates = list()

            while n_candidates < n_candidates_target:
                candidates = opts.sample_fun(opts, n_candidates_target)
                n_candidates = len(candidates)

                if n_candidates < n_candidates_target:
                    refinement_level = refinement_level + 1
                    print(
                        "No more samples found by bordersearch ("
                        + str(n_candidates)
                        + "/"
                        + str(n_candidates_target)
                        + "), "
                        + "increasing refinement level to "
                        + str(refinement_level)
                        + ". "
                    )
            if model.get_nr_class() < 2:
                idx = np.random.randint(0, n_candidates_target - 1)
            else:
                interval = range(n_candidates)
                weights = list()

                normalized_candidates = self.normalize(opts, candidates)

                _, _, probs = svmutil.svm_predict(list(), normalized_candidates, model)
                weights = [
                    math.exp(-weight_lambda * pow((prob[0] - 0.5), 2)) for prob in probs
                ]

                weights_sum = sum(weights)
                weights = [i / weights_sum for i in weights]
                idx = np.random.choice(interval, p=weights)

            point = candidates[idx]
            z = opts.dut_fun(opts, point)
            conf.append((point, z))

        return conf


def test_dut_fun(opts, params):
    return math.pow(params[0], 2) + math.pow(params[1], 2)


def dut_fun(opts, params):
    augmentation = [
        {key: value}
        for key, value in zip(
            opts.augmentation_conf.keys(), opts.augmentation_conf.values()
        )
    ]
    conf = {
        "checkpoints": opts.best_path,
        "augmentation": augmentation,
        "methods": "bordersearch",
    }
    result = eval(conf)
    return result[0]["bordersearch"][0]["val_ap"]


def random_sample(opts: Opts, n_candidates_target):
    def get_random_value(range: ParameterRange) -> float:
        return (
            range.start
            if range.start == range.end
            else random.uniform(range.start, range.end)
        )

    def get_random_param(opts: Opts) -> List[float]:
        dims = len(opts.parameters)
        ret = list()

        for j in range(dims):
            ret.append(get_random_value(opts.parameters[j].range))

        return ret

    candidates = list()
    for m in range(n_candidates_target):
        candidates.append(get_random_param(opts))
    return candidates


def main():
    b = Bordersearch()
    param = list()
    param.append(Parameter(ParameterRange(-1, 1), "", 0))
    param.append(Parameter(ParameterRange(-1, 1), "", 1))

    opts = Opts(param, 100)
    opts.dut_fun = test_dut_fun
    opts.sample_fun = random_sample
    conf = b.find_waterlevel(opts, 1)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for i in range(len(conf)):
        ax.scatter(conf[i][0][0], conf[i][0][1], conf[i][1])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.view_init(-60, 180)
    plt.savefig("border.png")


if __name__ == "__main__":
    main()
