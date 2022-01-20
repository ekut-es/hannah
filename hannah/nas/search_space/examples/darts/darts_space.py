from hannah.nas.search_space.symbolic_operator import (SymbolicOperator,
                                                       Choice,
                                                       Variable,
                                                       Context)
from hannah.nas.search_space.examples.darts.darts_parameter_functions import (
                                                       infer_in_channel,
                                                       keep_channels,
                                                       reduce_channels_by_edge_number,
                                                       multiply_by_stem,
                                                       reduce_and_double)
import torch
import networkx as nx
import numpy as np

from hannah.nas.search_space.symbolic_space import Space
from hannah.nas.search_space.connectivity_constrainer import DARTSCell
from hannah.nas.search_space.modules import Add, Concat
from hannah.nas.search_space.examples.darts.darts_modules import MixedOp, Classifier, Stem, Input
from copy import deepcopy


class DARTSSpace(Space):
    def __init__(self, num_cells=3, reduction_cells=[1]):
        super().__init__()

        in_channels = Variable('in_channels', func=infer_in_channel)
        stem_channels = Variable('in_channels_stem', func=multiply_by_stem)
        out_channels = Variable('out_channels', func=keep_channels)
        out_channels_adaptive = Variable('out_channels_adaptive', func=reduce_channels_by_edge_number)
        double_out_channels_adaptive = Variable('out_channels_adaptive', func=reduce_and_double)
        stride1 = Choice('stride', 1)
        stride2 = Choice('stride', 2)
        choice = Choice('choice', 0, 1, 2, 3, 4, 5, 6, 7)

        normal_cell = DARTSCell()
        normal_cell = normal_cell.add_operator_nodes()
        mapping = {}
        cfg = {'in_edges': 4, 'stem_multiplier': 4}

        for n in normal_cell.nodes:
            if n == 0:
                mapping[n] = SymbolicOperator('input_0', Input, in_channels=in_channels, out_channels=out_channels_adaptive, stride=stride1)
                cfg.update({'input_{}'.format(n): {'stride': 0}})
            elif n == 1:
                mapping[n] = SymbolicOperator('input_1', Input, in_channels=in_channels, out_channels=out_channels_adaptive, stride=stride1)
                cfg.update({'input_{}'.format(n): {'stride': 0}})
            elif n in range(2, 6):
                mapping[n] = SymbolicOperator('add_{}'.format(n), Add)
            elif n == 6:
                mapping[n] = SymbolicOperator('out', Concat)
            else:
                mapping[n] = SymbolicOperator('mixed_op_{}'.format(n), MixedOp, choice=choice, in_channels=in_channels, out_channels=out_channels, stride=stride1)
                cfg.update({'mixed_op_{}'.format(n): {'stride': 0, 'choice': np.random.randint(8)}})

        nx.relabel_nodes(normal_cell, mapping, copy=False)

        reduction_cell = DARTSCell()
        reduction_cell = reduction_cell.add_operator_nodes()
        mapping = {}
        for n in reduction_cell.nodes:
            if n == 0:
                mapping[n] = SymbolicOperator('input_0', Input, in_channels=in_channels, out_channels=double_out_channels_adaptive, stride=stride1)
                cfg.update({'input_{}'.format(n): {'stride': 0}})
            elif n == 1:
                mapping[n] = SymbolicOperator('input_1', Input, in_channels=in_channels, out_channels=double_out_channels_adaptive, stride=stride1)
                cfg.update({'input_{}'.format(n): {'stride': 0}})

            elif n in range(2, 6):
                mapping[n] = SymbolicOperator('add_{}'.format(n), Add)
            elif n == 6:
                mapping[n] = SymbolicOperator('out', Concat)
            elif isinstance(n, tuple) and n[0] in [0, 1]:
                mapping[n] = SymbolicOperator('mixed_op_{}'.format(n), MixedOp, choice=choice, in_channels=in_channels, out_channels=out_channels, stride=stride2)
                cfg.update({'mixed_op_{}'.format(n): {'stride': 0, 'choice': np.random.randint(8)}})
            else:
                mapping[n] = SymbolicOperator('mixed_op_{}'.format(n), MixedOp, choice=choice, in_channels=in_channels, out_channels=out_channels, stride=stride1)
                cfg.update({'mixed_op_{}'.format(n): {'stride': 0, 'choice': np.random.randint(8)}})

        nx.relabel_nodes(reduction_cell, mapping, copy=False)
        out_idx = 6
        input_0_idx = 0
        input_1_idx = 1

        stem = SymbolicOperator('stem0', Stem, C_out=stem_channels)

        cells = [deepcopy(normal_cell) for i in range(num_cells)]
        for idx in reduction_cells:
            cells[idx] = deepcopy(reduction_cell)
            if idx < len(cells) - 1:
                list(cells[idx+1].nodes)[input_0_idx].params['stride'] = stride2
                list(cells[idx+1].nodes)[input_0_idx].params['out_channels'] = double_out_channels_adaptive

        list(cells[0].nodes)[input_0_idx].params['out_channels'] = out_channels
        list(cells[0].nodes)[input_1_idx].params['out_channels'] = out_channels
        list(cells[1].nodes)[input_0_idx].params['out_channels'] = out_channels

        for i in range(len(cells)):
            self.add_nodes_from([n for n in cells[i].nodes])
            self.add_edges_from([e for e in cells[i].edges])

        self.add_edge(stem, list(cells[0].nodes)[input_0_idx])
        self.add_edge(stem, list(cells[0].nodes)[input_1_idx])
        self.add_edge(stem, list(cells[1].nodes)[input_0_idx])

        for i in range(len(cells)):
            if i < len(cells) - 2:
                self.add_edge(list(cells[i].nodes)[out_idx], list(cells[i+2].nodes)[input_0_idx])
            if i < len(cells) - 1:
                self.add_edge(list(cells[i].nodes)[out_idx],   list(cells[i+1].nodes)[input_1_idx])

        post = SymbolicOperator('post', Classifier, C=in_channels, num_classes=Choice('classes', 10))
        cfg.update({'post': {'classes': 0}})
        self.add_node(post)
        self.add_edge(list(cells[-1].nodes)[out_idx], post)

        self.ctx = Context(config=cfg)

    def get_ctx(self):
        return self.ctx


if __name__ == "__main__":
    num_cells = 9
    # reduction_cells = [i for i in range(num_cells) if i in [num_cells // 3, 2 * num_cells // 3]]
    reduction_cells = [2, 4, 6]
    print(reduction_cells)
    space = DARTSSpace(num_cells=num_cells, reduction_cells=reduction_cells)
    ctx = space.get_ctx()
    input = torch.ones([1, 3, 32, 32])
    instance, out1 = space.infer_parameters(input, ctx)
    print(out1.shape)

    out2 = instance.forward(input)
    print(out2.shape)
    print(out1 == out2)
