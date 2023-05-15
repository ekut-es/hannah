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
from hannah.callbacks.summaries import walk_model

# from z3 import Z3_global_param_set
# Z3_global_param_set("combined_solver.solver2_timeout", '5000')


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


def test_symbolic_macs():
    cwd = os.getcwd()
    config_path = Path(cwd + "/hannah/conf/model/capsule_net.yaml")
    input_shape = [3, 3, 32, 32]
    with config_path.open("r") as config_file:
        config = yaml.unsafe_load(config_file)
        config = OmegaConf.create(config)
    search_space = SearchSpace("resnet", params=config.params, input_shape=input_shape, labels=config.labels)
    search_space.sample()
    search_space.initialize()
    input_tensor = torch.ones(input_shape)


    t0 = time.perf_counter()
    summary = walk_model(search_space, dummy_input=input_tensor)
    t1 = time.perf_counter()
    print("walk_model: {:.4f}s.".format(t1 - t0))

    solver = ConstraintModel(method='linear')
    # solver.mac_constraint(search_space)
    solver.build_model(search_space._conditions)
    fixed_vars = {n: var for n, var in search_space.parametrization(flatten=True).items()}
    # for n, v in fixed_vars.items():
    #     solver.fix_var(solver.solver[0], v)
    t0 = time.perf_counter()
    print("Start check")
    # solver.solver[0].check()
    solver.soft_constrain_current_parametrization(search_space)
    search_space = solver.insert_model_values_to_module(search_space)
    t1 = time.perf_counter()
    print("Model constrained in {:.4f}s".format(t1 - t0))
    assert search_space.downsampling.evaluate() <= input_shape[2]
    assert search_space.downsampling.evaluate() >= input_shape[2] / 4
    # new_macs = search_space.macs.evaluate()
    # new_macs = search_space.easy_macs.evaluate()
    # print("MACs after constraining: {:.4E}".format(new_macs))
    # assert new_macs <= 1000000000
    print()


if __name__ == '__main__':
    test_symbolic_macs()