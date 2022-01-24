import networkx as nx
import torch.nn as nn
import traceback


class Space(nx.DiGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    def infer_parameters(self, x, ctx):
        def _traverse(node, input):
            # print(" Traverse node", node)
            in_edges = self.in_edges(node)
            # print(len(in_edges))
            if len(in_edges) > 0:
                args = []
                for u, v in in_edges:
                    # print("in_edge ", u)
                    if u not in ctx.outputs:
                        args.append(_traverse(u, input))
                    else:
                        args.append(ctx.outputs[u])
                if len(args) == 1:
                    args = args[0]
                try:
                    ctx.set_input(args)
                    mod = node.instantiate(ctx)
                    ctx.relabel_dict[node] = mod
                    out = mod(args)
                    ctx.outputs[node] = out
                except Exception as e:
                    print(node)
                    print(len(args))
                    print(str(e))
                    print(traceback.format_exc())
            else:
                ctx.set_input(input)
                mod = node.instantiate(ctx)
                ctx.relabel_dict[node] = mod
                out = mod(input)
                ctx.outputs[node] = out
            return out

        nodes = list(nx.topological_sort(self))
        last = nodes[-1]
        out = _traverse(last, x)
        instance = Instance(nx.relabel_nodes(self, ctx.relabel_dict))
        return instance, out

    def get_instance(self, ctx):
        graph = self.get_graph(ctx)
        relabel_dict = {}
        for node in graph.nodes:
            mod = node.instantiate(ctx)
            relabel_dict[node] = mod

        instance = Instance(nx.relabel_nodes(graph, relabel_dict))
        return instance

    def get_config_dims(self):
        cfg = {}
        for node in self.nodes:
            cfg_dims = node.get_config_dims()
            if len(cfg_dims[node.name]) > 0:
                cfg.update(cfg_dims)
        return cfg


class Instance(nx.DiGraph, nn.Module):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)

    def forward(self, x):
        outputs = {}

        def _compute(node, input):
            in_edges = self.in_edges(node)
            if len(in_edges) > 0:
                args = []
                for u, v in in_edges:
                    if u not in outputs:
                        args.append(_compute(u, input))
                    else:
                        args.append(outputs[u])
                if len(args) == 1:
                    args = args[0]
                try:
                    out = node(args)
                    outputs[node] = out
                except Exception as e:
                    print(str(e))
            else:
                out = node(input)
                outputs[node] = out
            return out

        last = list(nx.topological_sort(self))[-1]
        out = _compute(last, x)
        return out
