import hydra
from omegaconf import DictConfig
from hannah.nas.search_space.symbolic_operator import Context
import torch
import networkx as nx
import numpy as np

from hannah.nas.search_space.tcresnet.tcresnet_space import TCResNetSpace
from hannah.nas.search_space.utils import flatten_config, get_first_cfg, get_random_cfg


class Pruner:
    def __init__(self, space):
        self.space = space
        self.config_dims = space.get_config_dims()

        # compute the order in that nodes have to be executed to have
        # have the correct inputs at the correct times (e.g. executed all branches
        # before the add)
        self.node_queue = self.create_node_queue()
        self.current_config = get_first_cfg(self.config_dims)

        # sort parameters by node order to have a theoretical correct traversal tree
        self.node_order_in_config = {v.name: i for i, v in enumerate(self.node_queue) if v.name in self.config_dims}
        self.node_order_full = {v.name: i for i, v in enumerate(self.node_queue)}
        self.config_dims = {k: v for k, v in sorted(self.config_dims.items(), key=lambda x: self.node_order_in_config[x[0]])}
        self.flatten_config_dims = flatten_config(self.config_dims)
        self.config_indices = {n : i for i, (n, v) in enumerate(self.flatten_config_dims.items())}
        self.invalid_paths = np.zeros((1, len(self.flatten_config_dims.keys())))
        self.valid = 0
        self.invalid = 0

    def prune(self, x):
        self.outputs = {'input': x}
        self.ctx = Context(config=self.current_config)

        def _prune(node):
            # print(' {} - {}'.format(self.valid, self.invalid), end='\r', flush=True)
            print(' {} - {}'.format(self.valid, self.invalid), list(flatten_config(self.current_config).values()), end='\r', flush=True)
            if node.name in self.config_dims:
                for key, values in self.config_dims[node.name].items():
                    for val in values:
                        self.current_config[node.name][key] = val
                        self.ctx.set_cfg(cfg=self.current_config)

                        passed = self.process_node(node)
                        if passed and self.node_order_full[node.name] < len(self.node_queue) - 1:
                            _prune(self.node_queue[self.node_order_full[node.name] + 1])
                        elif passed and self.node_order_full[node.name] == len(self.node_queue) - 1:
                            self.valid += 1
                        else:
                            self.invalid += 1
            else:
                passed = self.process_node(node)
                if passed and self.node_order_full[node.name] < len(self.node_queue) - 1:
                    _prune(self.node_queue[self.node_order_full[node.name] + 1])
                elif passed and self.node_order_full[node.name] == len(self.node_queue) - 1:
                    self.valid += 1
                else:
                    self.invalid += 1

        _prune(self.node_queue[0])
        # print('{} - {}'.format(valid, invalid), end='\r', flush=True)

    def find_next_valid_config(self, x, config):
        self.outputs = {'input': x}
        self.current_config = config
        self.ctx = Context(config=self.current_config)

        def _search(node):
            print(' {} - {}'.format(self.valid, self.invalid), end='\r', flush=True)
            if node.name in self.config_dims:
                for key, values in self.config_dims[node.name].items():
                    for val in values:
                        self.current_config[node.name][key] = val
                        self.ctx.set_cfg(cfg=self.current_config)

                        passed = self.process_node(node)
                        if passed and self.node_order_full[node.name] < len(self.node_queue) - 1:
                            _search(self.node_queue[self.node_order_full[node.name] + 1])
                        elif passed and self.node_order_full[node.name] == len(self.node_queue) - 1:
                            self.valid += 1
                            raise Exception("Catch this exception to end search")
                        else:
                            self.invalid += 1

            else:
                passed = self.process_node(node)
                if passed and self.node_order_full[node.name] < len(self.node_queue) - 1:
                    return _search(self.node_queue[self.node_order_full[node.name] + 1])
                elif passed and self.node_order_full[node.name] == len(self.node_queue) - 1:
                    self.valid += 1
                    raise Exception("Catch this exception to end search")
                else:
                    self.invalid += 1

        try:
            _search(self.node_queue[0])
        except Exception:
            print("Found config.")
        print(' {} - {}'.format(self.valid, self.invalid))
        return self.current_config

    def process_node(self, node):
        passed = True
        try:
            args = []
            in_edges = self.space.in_edges(node)
            if len(in_edges) > 0:
                for u, v in in_edges:
                    args.append(self.outputs[u.name])
                if len(args) == 1:
                    args = args[0]
            else:
                args = self.outputs['input']
            self.ctx.set_input(args)
            mod = node.instantiate(self.ctx)
            out = mod(args)
            self.outputs[node.name] = out
        except Exception:
            passed = False

        return passed

    def create_node_queue(self):
        visited = []

        def _traverse(node, queue):
            # print("Traverse node", node)
            in_edges = self.space.in_edges(node)

            for u, v in in_edges:
                _traverse(u, queue)
            if node not in visited:
                queue.append(node)
                visited.append(node)

        nodes = list(nx.topological_sort(self.space))
        last = nodes[-1]
        queue = []
        _traverse(last, queue)
        return queue


@hydra.main(config_name="config", config_path="../../conf")
def main(config: DictConfig):
    # test_basic_model(config)

    space = TCResNetSpace(config, parameterization=True)
    pruner = Pruner(space)
    cfg = get_random_cfg(space.get_config_dims())
    x = torch.ones([1, 40, 101])
    pruner.prune(x)

    # print(cfg)
    # try:
    #     ctx = Context(cfg)
    #     instance, out = space.infer_parameters(x, ctx, verbose=True)
    # except Exception as e:
    #     print("Instantiation failed: ", str(e))
    # cfg = pruner.find_next_valid_config(x, cfg)
    # print(cfg)
    # ctx = Context(cfg)
    # instance, out = space.infer_parameters(x, ctx, verbose=True)
    # print(out.shape)


if __name__ == '__main__':
    main()
