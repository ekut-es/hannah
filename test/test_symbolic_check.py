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

    summary = walk_model(search_space, dummy_input=input_tensor)
    macs = search_space.macs.evaluate()
    # print("MACs before constraining: {:.4E}".format(macs))

    # assert summary['MACs'].sum() == macs
    solver = ConstraintModel(method='naive')
    t0 = time.perf_counter()
    solver.soft_constrain_current_parametrization(search_space)
    solver.insert_model_values_to_module(search_space)
    t1 = time.perf_counter()
    # print("Model constrained in {:.4f}s".format(t1 - t0))
    new_macs = search_space.macs.evaluate()
    # print("MACs after constraining: {:.4E}".format(new_macs))

    times = []
    macs = []
    tries = []
    # print("MACs before constraining: {:.4E}".format(macs))
    budget = 100

    # for i in range(100):
    #     t0 = time.perf_counter()
    #     ct = 0
    #     while True:
    #         search_space.sample()
    #         try:
    #             search_space.check()
    #             break
    #         except Exception:
    #             ct += 1
    #     t1 = time.perf_counter()
    #     times.append(t1 - t0)
    #     tries.append(ct)
    #     print("Model constrained in {:.4f}s".format(t1 - t0))
    #     new_macs = search_space.macs.evaluate()
    #     print("MACs after constraining: {:.4E}".format(new_macs))
    #     macs.append(new_macs)
    # df = pd.DataFrame({'times': times, 'tries': tries, 'macs': macs})
    # df.to_csv("./random_pre_constrainting.csv")
    # print()


if __name__ == '__main__':
    test_symbolic_macs()
