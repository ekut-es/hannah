import os
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
import torch
import time

import yaml

import pandas as pd

from hannah.models.capsule_net.models import SearchSpace
from hannah.nas.parameters.parameters import CategoricalParameter
from hannah.nas.constraints.constraint_model import ConstraintModel
from z3 import Z3_reset_memory


def diff(search_space, solver):
    for solv in solver.solver:
        solv.check()
        mod = solv.model()
        par = {}
        sol = {}
        for name, p in search_space.parametrization(flatten=True).items():
            if name in solver.vars[solv]:
                par[name] = int(p.current_value)
                sol[name] = mod[solver.vars[solv][name]].as_long()

                print(f"{name}: {par[name]} - {sol[name]}")
    par = np.array(list(par.values()))
    sol = np.array(list(sol.values()))
    return np.linalg.norm(par-sol)


def test_capsule_net():
    cwd = os.getcwd()
    config_path = Path(cwd + "/hannah/conf/model/capsule_net.yaml")
    input_shape = [3, 3, 336, 336]
    with config_path.open("r") as config_file:
        config = yaml.unsafe_load(config_file)
        config = OmegaConf.create(config)
    search_space = SearchSpace("resnet", params=config.params, input_shape=input_shape, labels=config.labels)
    search_space.sample()


    solver = ConstraintModel(method='naive')
    solver.build_model(search_space._conditions)

    df = pd.DataFrame(columns=["run", "method", "time", "diff"])
    # for i in range(1):
    #     search_space.sample()
        # t0 = time.perf_counter()
        # solver.soft_constrain_current_parametrization(search_space, key="stride", method='linear')
        # t1 = time.perf_counter()
        # td = t1 - t0
        # data = {"run": i, "method": 'linear', 'time': td, 'diff': diff(search_space, solver)}
        # df = df.append(data, ignore_index=True)
        # print("Run {} Method: {} Time: {:.4f}".format(i, 'linear', td))

        # t0 = time.perf_counter()
        # solver.soft_constrain_current_parametrization(search_space)
        # t1 = time.perf_counter()
        # td = t1 - t0

        # data = {"run": i, "method": 'naive', 'time': td, 'diff': diff(search_space, solver)}
        # df = df.append(data, ignore_index=True)
        # print("Run {} Method: {} Time: {:.4f}".format(i, 'naive', td))

    solver.soft_constrain_current_parametrization(search_space)
    print(diff(search_space, solver))
    solver.insert_model_values_to_module(search_space)
    # search_space.check()

    x = torch.randn(input_shape)
    search_space.initialize()
    out = search_space(x)
    assert out.shape == (3, 10)

    print()

if __name__ == '__main__':
    test_capsule_net()


