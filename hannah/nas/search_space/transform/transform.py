import hydra
from omegaconf import DictConfig
import torch
# import torch.nn as nn
from hannah.nas.search_space.symbolic_operator import Context
from hannah.nas.search_space.torch_converter import FunctionWrapper
from hannah.nas.search_space.modules.primitive_operators import Add

from hannah.nas.search_space.tcresnet.tcresnet_space import TCResNetSpace
from hannah.nas.search_space.pruner import Pruner
from hannah.nas.search_space.symbolic_constraint_solver import SymbolicConstrainer
from hannah.nas.search_space.utils import get_random_cfg


class Transformer:
    def __init__(self, space) -> None:
        self.space = space

    def transform_nodes(self, node_map, rules=[], attr_map={}, **kwargs):
        for node in self.space.nodes:
            if node.target_cls in node_map and self.check_rules(node, rules):
                node.target_cls = node_map[node.target_cls]
                new_params = {}
                for key, value in node.params.items():
                    if key in attr_map:
                        new_params[attr_map[key]] = node.params[key]
                for key, value in kwargs.items():
                    new_params[key] = value
                node.params = new_params

    # def merge_nodes(self, source_sequence, target_sequence)

    def check_rules(self, node, rules):
        marker = True
        for rule in rules[node.target_cls]:
            marker = rule(node)
            if not marker:
                return marker
        return marker


@hydra.main(config_name="config", config_path="../../../conf")
def main(config: DictConfig):
    space = TCResNetSpace(config, parameterization=True)
    transformer = Transformer(space)

    def is_add(node):
        return True if 'function' in node.params and node.params['function'] == 'add' else False

    node_map = {FunctionWrapper: Add}
    rules = {}
    rules[FunctionWrapper] = [is_add]

    transformer.transform_nodes(node_map, rules)

    pruner = Pruner(space)
    channel_constrainer = SymbolicConstrainer(space)
    cfg = get_random_cfg(space.get_config_dims())
    cfg = channel_constrainer.constrain_output_channels(cfg)
    x = torch.ones([1, 40, 101])
    cfg = pruner.find_next_valid_config(x, cfg, exclude_keys=['out_channels', 'kernel_size', 'dilation'])
    ctx = Context(cfg)
    instance, out = space.infer_parameters(x, ctx, verbose=True)
    print(out.shape)


if __name__ == '__main__':
    main()
