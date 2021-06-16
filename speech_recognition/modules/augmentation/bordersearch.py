from dataclasses import dataclass

import numpy as np
from numpy import random
import libsvm.svmutil as svmutil

import math
from numpy.core.arrayprint import ComplexFloatingFormat

import matplotlib.pyplot as plt


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
    pos: int
    # TODO add solver

    def __init__(self, range: ParameterRange, uuid: str, pos: int):
        self.range = range
        self.uuid = uuid
        self.pos = pos


@dataclass
class Opts:
    ground_truth: np
    runs: int
    waterlevel: float
    samples: int
    svm_params: str
    parameters: list()
    dummy_tsr_matrix: np
    active: bool

    def __init__(self, parameters: list, ground_truth: np, runs: int):
        self.parameters = parameters
        self.ground_truth = ground_truth
        self.runs = runs
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

    def get_random_np(self, opts: Opts) -> np:
        dims = len(opts.parameters)
        ret = np.zeros(dims)

        for j in range(dims):
            ret[j] = self.get_random_value(opts.parameters[j][0])

        return ret

    def normalize(self, opts: Opts, values: np) -> np:
        result = np.zeros(len(values[0]))

        for i in range(len(opts.parameters)):
            min = opts.parameters[i][0].start
            max = opts.parameters[i][0].end
            col = values[0][i]
            result[i] = (col - min) / (max - min)

        return result

    def mat_to_svm_nodes(self, mat, i):
        return (i, mat) if mat != 0 else (-1, 0)

    def mat_to_svm_nodes_array(self, mat, number_of_params):
        nodes = list()
        for i in range(len(mat)):
            nodes.append(self.mat_to_svm_nodes(mat[i], i))
        return nodes

    def svm_train_matrix(self, opts: Opts, labels, data, parameter):
        number_of_params = len(opts.parameters)
        problem = svmutil.svm_problem(
            labels.tolist(), self.mat_to_svm_nodes_array(data, number_of_params)
        )

        return svmutil.svm_train(problem, parameter)

    def calc_above_wl(self, data, waterlevel):
        above_wl = np.zeros(len(data[0]))

        for i in range(len(data[0])):
            above_wl[i] = 0 if data[0][i] < waterlevel else 1

        return above_wl

    def test_dut_fun(self, opts, params):
        return math.pow(params[0], 2) + math.pow(params[1], 2)

    def find_waterlevel(self, opts: Opts, waterlevel: float) -> np:
        n_candidates_target = opts.samples
        svm_cost = 100
        weight_lambda = 20
        number_of_params = len(opts.parameters)
        refinement_level = 0

        conf = np.zeros((opts.runs, number_of_params + 1))

        for i in range(opts.runs):
            progress = (float)(i + 1) / opts.runs
            size = 50
            out = "\r["
            out += str(["#" for _ in range(int(progress * size))])
            out += str([" " for _ in range(size)])
            out += str("]" + str((i + 1)) + ("/") + str(opts.runs) + (" "))
            print(out)

            if i == 0:
                # first run, generate some bootsrap data
                point = self.get_random_np(opts)
                z = self.test_dut_fun(opts, point)
                conf[i, :-1] = point
                conf[i, -1] = z
                continue

            known = conf[:i, :]
            above_wl = self.calc_above_wl(known[:, :number_of_params], waterlevel)
            parameter = svmutil.svm_parameter(
                "-s 0 -t 2 -d 3 -g 0.5 -r 0 -c " + str(svm_cost) + " -b 1 -m 20 -e 0.1"
            )

            # train model using previous sampled data as ground truth
            normalized_known = self.normalize(opts, known[:, :number_of_params])
            model = self.svm_train_matrix(opts, above_wl, normalized_known, parameter)

            # generate new sampels
            n_candidates = 0
            candidates = np.zeros(0)

            while n_candidates < n_candidates_target:
                candidates = np.append(candidates, self.get_random_np(opts))
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
                interval = np.zeros(n_candidates + 1)
                weights = np.zeros(n_candidates)

                normalized_candidates = self.normalize(opts, candidates)

                for j in range(n_candidates):
                    predict_nodes = self.mat_to_svm_nodes(
                        normalized_candidates[j], number_of_params
                    )
                    prob = svmutil.svm_predict(model, predict_nodes)

                    interval[j] = j
                    weights[j] = math.exp(-weight_lambda * pow((prob - 0.5), 2))

                interval[n_candidates] = n_candidates

                idx = np.random.randint(0, n_candidates_target - 1)
                idx = min(n_candidates - 1, idx)

            point = candidates[idx]
            z = point  # opts.dut_fun(opts, point)
            conf[i, :-1] = point
            conf[i, -1] = z

        return conf


def main():
    b = Bordersearch()
    param = list()
    param.append((ParameterRange(-1, 1), "", 0))
    param.append((ParameterRange(-1, 1), "", 1))

    opts = Opts(param, None, 100)
    conf = b.find_waterlevel(opts, 1)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for i in range(len(conf)):
        ax.scatter(conf[i, 0], conf[i, 1], conf[i, 2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(-60, 180)
    plt.savefig("border.png")


if __name__ == "__main__":
    main()
