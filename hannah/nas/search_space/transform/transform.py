import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from hannah.models.factory import qat as qat
from hannah.nas.search_space.symbolic_operator import Context, SymbolicOperator
from hannah.nas.search_space.torch_converter import FunctionWrapper
from hannah.nas.search_space.modules.primitive_operators import Add

from hannah.nas.search_space.tcresnet.tcresnet_space import TCResNetSpace
from hannah.nas.search_space.pruner import Pruner
from hannah.nas.search_space.symbolic_constraint_solver import SymbolicConstrainer
from hannah.nas.search_space.utils import get_random_cfg
import networkx as nx
# from hannah.models.factory.qconfig import get_trax_qat_qconfig
from torch.quantization import default_qconfig


class Transformer:
    def __init__(self, space) -> None:
        self.space = space

    def transform_nodes(self, node_map, rules={}, attr_map={}, **kwargs):
        for node in self.space.nodes:
            if node.target_cls in node_map and self.check_rules(node, rules):
                node.target_cls = node_map[node.target_cls]
                new_params = {}
                for key, value in node.params.items():
                    if node.target_cls in attr_map and key in attr_map[node.target_cls]:
                        new_params[attr_map[node.target_cls][key]] = node.params[key]
                for key, value in kwargs.items():
                    new_params[key] = value
                node.params = new_params

    def transform_node_sequence(self, source, target, rules={}, attr_map={}, additional_attrs={}):
        new_edges = []
        to_delete = []
        ct = 0
        for node in nx.topological_sort(self.space):
            found_sequence = False
            if node.target_cls == source[0] and self.check_rules(node, rules):
                sequence = []
                found_sequence = self.check_path(node, source, rules, sequence)
            if found_sequence:
                attrs = {}
                for target_index, mapping in attr_map.items():
                    attrs[target_index] = {}
                    for target_key, source_value in mapping.items():
                        attrs[target_index][target_key] = sequence[source_value[0]].params[source_value[1]]

                for target_index, mapping in additional_attrs.items():
                    for key, value in mapping.items():
                        attrs[target_index][key] = value

                new_node = SymbolicOperator(name=str(target[0]).split('.')[-1].split('\'')[0] + '_{}'.format(ct),
                                            target_cls=target[0],
                                            **attrs[target_index])
                ct += 1

                new_edges.append((list(self.space.in_edges(node))[0][0], new_node))
                new_edges.append((new_node, list(self.space.out_edges(sequence[-1]))[0][1]))
                to_delete.extend(sequence)

        self.space.remove_nodes_from(to_delete)
        self.space.add_edges_from(new_edges)

    def check_rules(self, node, rules):
        marker = True
        if node.target_cls not in rules:
            return marker
        for rule in rules[node.target_cls]:
            marker = rule(node)
            if not marker:
                return marker
        return marker

    def check_path(self, start_node, path, rules, sequence):
        if len(path) > 1 and start_node.target_cls == path[0] and self.check_rules(start_node, rules):
            out_edges = list(self.space.out_edges(start_node))
            if len(out_edges) > 1:
                return False  # matching with path forking not supported
            v = out_edges[0][1]
            sequence.append(start_node)
            return self.check_path(v, path[1:], rules, sequence)
        elif len(path) == 1 and start_node.target_cls == path[0] and self.check_rules(start_node, rules):
            sequence.append(start_node)
            return True
        else:
            return False


@hydra.main(config_name="config", config_path="../../../conf")
def main(config: DictConfig):
    space = TCResNetSpace(config, parameterization=True)
    transformer = Transformer(space)

    def is_add(node):
        return True if 'function' in node.params and node.params['function'] == 'add' else False

    node_map = {FunctionWrapper: Add}
    rules = {}
    rules[FunctionWrapper] = [is_add]

    pruner = Pruner(space)
    channel_constrainer = SymbolicConstrainer(space)
    cfg = get_random_cfg(space.get_config_dims())
    cfg = channel_constrainer.constrain_output_channels(cfg)
    x = torch.ones([1, 40, 101])
    cfg = pruner.find_next_valid_config(x, cfg, exclude_keys=['out_channels', 'kernel_size', 'dilation'])
    ctx = Context(cfg)
    instance_1, out_1 = space.infer_parameters(x, ctx, verbose=True)
    instance_1.eval()
    out_1 = instance_1(x)

    transformer.transform_nodes(node_map, rules)
    ctx = Context(cfg)
    instance_2, out_2 = space.infer_parameters(x, ctx, verbose=True)
    state_dict = instance_1.state_dict()
    instance_2.load_state_dict(state_dict)
    instance_2.eval()
    out_2 = instance_2(x)

    torch.testing.assert_allclose(out_2, out_1)
    print(out_2.shape)

    source_sequence = [nn.Conv1d, nn.BatchNorm1d, nn.ReLU]
    target_sequence = [qat.ConvBnReLU1d]

    # Format:
    # {index of node in target_sequence: {key in target node: (index of node in source_sequence, key in source_sequence)}}
    attr_map = {0: {'in_channels':  (0, 'in_channels'),
                    'out_channels': (0, 'out_channels'),
                    'kernel_size':  (0, 'kernel_size'),
                    'stride':       (0, 'stride'),
                    'padding':      (0, 'padding'),
                    'dilation':     (0, 'dilation'),
                    'eps':          (1, 'eps'),
                    'momentum':     (1, 'momentum')}
                }
    additional_attrs = {0: {'qconfig': default_qconfig,
                            'out_quant': False}
                        }

    transformer.transform_node_sequence(source_sequence,
                                        target_sequence,
                                        attr_map=attr_map,
                                        additional_attrs=additional_attrs)

    pruner = Pruner(space)
    channel_constrainer = SymbolicConstrainer(space)
    cfg = get_random_cfg(space.get_config_dims())
    cfg = channel_constrainer.constrain_output_channels(cfg)
    x = torch.ones([1, 40, 101])
    cfg = pruner.find_next_valid_config(x, cfg, exclude_keys=['out_channels', 'kernel_size', 'dilation'])
    ctx = Context(cfg)
    instance, out = space.infer_parameters(x, ctx, verbose=True)
    print(out.shape)
    print("Passed all tests")


if __name__ == '__main__':
    main()
