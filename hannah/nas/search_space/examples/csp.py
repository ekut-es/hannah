import constraint as c
import hydra
from omegaconf import DictConfig
import torch
from hannah.nas.search_space.symbolic_operator import Context

from hannah.nas.search_space.tcresnet.tcresnet_space import TCResNetSpace
from hannah.nas.search_space.utils import flatten_config, get_random_cfg
from torch import nn as nn
from hannah.nas.search_space.symbolic_constraint_solver import SymbolicConstrainer
import numpy as np


def conv_output(out, inp, stride):
    return out == int(np.ceil(inp / stride))


def similarity(input_1, input_2):
    return input_1 == input_2


def halves(x):
    halves = []

    while x > 0:
        halves.append(x)
        x /= 2
        x = int(np.ceil(x))
    return halves


@hydra.main(config_name="config", config_path="../../../conf")
def main(config: DictConfig):

    x = torch.ones([1, 40, 101])
    fm_values = halves(x.shape[2])

    space = TCResNetSpace(config, parameterization=True)
    problem = c.Problem()
    constraints = {'similarity': [],
                   'conv_output': []}
    for node in space.nodes:
        fm_name = '{}_{}'.format(node.name, 'fm_size')
        problem.addVariable(fm_name, fm_values)

        in_edges = space.in_edges(node)
        input_fm_names = []
        if len(in_edges) == 0:
            input_fm_names = ['input']
            problem.addVariable('input', [x.shape[2]])
        else:
            for u, v in in_edges:
                input_fm_names.append('{}_{}'.format(u.name, 'fm_size'))
        if 'add' in node.name:
            constraints['similarity'].append((input_fm_names[0], input_fm_names[1]))
            constraints['similarity'].append((input_fm_names[0], fm_name))
        elif node.target_cls != nn.Conv1d:
            constraints['similarity'].append((fm_name, input_fm_names[0]))
        elif node.target_cls == nn.Conv1d:
            stride_name = '{}_{}'.format(node.name, 'stride')
            problem.addVariable(stride_name, [1, 2])

            constraints['conv_output'].append((fm_name, input_fm_names[0], stride_name))
            last_conv = [fm_name]

    for var in constraints['similarity']:
        problem.addConstraint(similarity, var)

    for var in constraints['conv_output']:
        problem.addConstraint(conv_output, var)

    problem.addConstraint(lambda out: out > 2, last_conv)

    sol = problem.getSolutions()
    cfg = get_random_cfg(space.get_config_dims())

    for node, params in cfg.items():
        for k, v in params.items():
            if k == 'stride':
                cfg[node][k] = 0 if sol[0]['{}_{}'.format(node, 'stride')] == 1 else 1
    values = list(flatten_config(cfg).values())
    print(values)
    channel_constrainer = SymbolicConstrainer(space)

    cfg = channel_constrainer.constrain_output_channels(cfg)
    ctx = Context(cfg)

    instance, out = space.infer_parameters(x, ctx, verbose=True)
    print(out.shape)


if __name__ == '__main__':
    main()
