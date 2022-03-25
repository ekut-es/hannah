from copy import deepcopy
import hydra
from omegaconf import DictConfig
from hannah.nas.search_space.symbolic_operator import Context
import torch

from hannah.nas.search_space.tcresnet.tcresnet_space import TCResNetSpace
from hannah.nas.search_space.pruner import Pruner
from hannah.nas.search_space.symbolic_constraint_solver import SymbolicConstrainer
from hannah.nas.search_space.utils import flatten_config, get_random_cfg


@hydra.main(config_name="config", config_path="../../../conf")
def main(config: DictConfig):

    space = TCResNetSpace(config, parameterization=True)
    pruner = Pruner(space)
    channel_constrainer = SymbolicConstrainer(space)

    cfg = get_random_cfg(space.get_config_dims())
    values = list(flatten_config(cfg).values())
    print("Before channel constrainer")
    print(values)
    cfg = channel_constrainer.constrain_output_channels(cfg)
    old_cfg = deepcopy(cfg)
    values = list(flatten_config(cfg).values())
    print("After channel constrainer")
    print(values)
    x = torch.ones([1, 40, 101])
    cfg = pruner.find_next_valid_config(x, cfg, exclude_keys=['out_channels', 'kernel_size', 'dilation'])
    values = list(flatten_config(cfg).values())
    print("After pruner")
    print(values)
    ctx = Context(cfg)
    instance, out = space.infer_parameters(x, ctx, verbose=True)
    print(out.shape)

    for node, param in space.get_config_dims().items():
        for key, value in param.items():
            if key == 'stride':
                print('{:25s}: {}-{}'.format(node, old_cfg[node][key], cfg[node][key]))


if __name__ == '__main__':
    main()
