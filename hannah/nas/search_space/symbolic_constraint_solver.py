import torch
from sympy import solve, symbols
from hannah.nas.search_space.tcresnet.tcresnet_space import TCResNetSpace
import hydra
from omegaconf import DictConfig
from hannah.nas.search_space.symbolic_operator import Context
import numpy as np
import torch.nn as nn


@hydra.main(config_name="config", config_path="../../conf")
def main(config: DictConfig):
    constraints = []
    channel_symbols = {}
    space = TCResNetSpace(config, parameterization=True)

    for node in space.nodes:
        channel_name = '{}_{}'.format(node.name, 'out_channels')
        channel_symbols[channel_name] = symbols(channel_name, integer=True)
        in_edges = space.in_edges(node)
        channel_names = []
        if 'add' in node.name:
            for u, v in in_edges:
                channel_names.append('{}_{}'.format(u.name, 'out_channels'))
            constraints.append(channel_symbols[channel_names[0]] - channel_symbols[channel_names[1]])  # channels of inputs have to be equal
            constraints.append(channel_symbols[channel_names[0]] - channel_symbols[channel_name])  # out_channels of add itself are equal to first input
        elif node.target_cls != nn.Conv1d:
            for u, v in in_edges:
                channel_names.append('{}_{}'.format(u.name, 'out_channels'))
            constraints.append(channel_symbols[channel_name] - channel_symbols[channel_names[0]])  # outputchannels are unchanged
        elif node.target_cls == nn.Conv1d:
            pass

    cfg_dims = space.get_config_dims()
    sol = solve(constraints)
    cfg = {}
    for node, params in cfg_dims.items():
        cfg[node] = {}
        for k, v in params.items():
            if k == 'out_channels':
                channel_name = '{}_{}'.format(node, 'out_channels')
                if channel_symbols[channel_name] in sol and not sol[channel_symbols[channel_name]].is_symbol:
                    val = int(sol[channel_symbols[channel_name]])
                    constraints.append(channel_symbols[channel_name] - val)
                else:
                    val = np.random.choice(v)
                    constraints.append(channel_symbols[channel_name] - val)
                cfg[node][k] = val
            else:
                cfg[node][k] = np.random.choice(v)
        sol = solve(constraints)
    print(cfg)
    ctx = Context(cfg)
    x = torch.ones([1, 40, 101])
    instance, out = space.infer_parameters(x, ctx, verbose=True)
    print(out.shape)


if __name__ == '__main__':
    main()
