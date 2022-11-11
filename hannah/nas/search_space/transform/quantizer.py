import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import hannah.models.factory.qat as qat

from hannah.nas.search_space.tcresnet.tcresnet_space import TCResNetSpace
from hannah.nas.search_space.pruner import Pruner
from hannah.nas.search_space.symbolic_constraint_solver import SymbolicConstrainer
from hannah.nas.search_space.utils import get_random_cfg


class Quantizer:
    def __init__(self) -> None:
        self.transform_map = {nn.Conv1d: qat.Conv1d}







@hydra.main(config_name="config", config_path="../../conf")
def main(config: DictConfig):
    space = TCResNetSpace(config, parameterization=True)
    pruner = Pruner(space)
    channel_constrainer = SymbolicConstrainer(space)
    cfg = get_random_cfg(space.get_config_dims())
    cfg = channel_constrainer.constrain_output_channels(cfg)
    x = torch.ones([1, 40, 101])
    cfg = pruner.find_next_valid_config(x, cfg, exclude_keys=['out_channels', 'kernel_size', 'dilation'])
