from itertools import repeat
import numpy as np
import networkx as nx
import torch

from hannah.nas.search_space.space import Space, Cell
from hannah.nas.search_space.operator import DepthwiseSeparableConvolution, Pooling, FactorizedReduce, Operator, Zero, Combine, Activation, Choice
from hannah.nas.search_space.utils import get_random_cfg_vec, vec_to_knob
from hannah.nas.search_space.model import Model
from hannah.nas.search_space.connectivity_constrainer import DARTSCell, DARTSGraph

class DARTSSpace(Space):

    def __init__(self, num_cells=3) -> None:
        num_nodes = 4
        num_input = 2
        num_output = 1

        redux_cells = [1, 2]

        self.IN_EDGES = 4

        def double_channel(x):
            return 2*x

        def double_channel_mult(x):
            return 2*x*self.IN_EDGES

        def same_channel(x):
            return x

        def same_channel_mult(x):
            return x*self.IN_EDGES

        preprocess_cell = DARTSCell(num_inputs=0, num_nodes=0, num_outputs=1)
        cells = list(repeat(DARTSCell(num_nodes=num_nodes, num_inputs=num_input, num_outputs=num_output), num_cells))
        darts_graph = DARTSGraph([preprocess_cell] + cells)

        dw_conv = DepthwiseSeparableConvolution(same_channel, kernel_size=[3, 5],  dilation=[1], padding='same')
        dw_conv_redux = DepthwiseSeparableConvolution(double_channel, kernel_size=[3, 5], dilation=[2], padding='half', stride=2)

        identity = Operator()


        factorized_reduce_normal = FactorizedReduce(stride=1, out_channels=same_channel)
        factorized_reduce_redux = FactorizedReduce(stride=2, out_channels=double_channel)


        pool = Pooling(mode=['max', 'avg'], kernel_size=3, stride=1, padding='same')
        pool_reduce = Pooling(mode=['max', 'avg'], kernel_size=3, stride=2, padding='half')
        zero = Zero()
        act = Activation('relu')
        add = Combine('add')
        add_1x1 = Combine('add')
        concat = Combine('concat')
        choice = Choice([dw_conv, factorized_reduce_normal])
        choice_redux = Choice([dw_conv_redux, factorized_reduce_redux])

        factorized_reduce_redux_skip = FactorizedReduce(stride=2, out_channels=double_channel_mult)
        factorized_reduce_normal_skip = FactorizedReduce(stride=1, out_channels=same_channel_mult)

        operator_node = Cell([choice])
        input_node = Cell([identity])
        add_node = Cell([add_1x1])
        output_node = Cell([concat])
        fact_red_redux = Cell([factorized_reduce_redux_skip])
        fact_red_normal = Cell([factorized_reduce_normal_skip])

        operator_redux = Cell([choice_redux])
        cell_dict = {}
        types = nx.get_node_attributes(darts_graph.m_arch, 'type')
        cells = nx.get_node_attributes(darts_graph.m_arch, 'cell')
        label_dict = {}
        for node in darts_graph.m_arch:

            if types[node] == 'op' and isinstance(cells[node], tuple) and cells[node][0] == cells[node][1] and types[node[0]] == 'input' and cells[node][0] in redux_cells:
                cell_dict[node] = operator_redux.new()
                label_dict[node] = 'red'
            elif types[node] == 'op' and cells[node][0] != cells[node][1]:
                if cells[node[0] + 1] in redux_cells and cells[node][0] + 1 != cells[node][1]:
                    cell_dict[node] = fact_red_redux.new()
                    label_dict[node] = 'fr r'
                elif cells[node[0] + 1] not in redux_cells and cells[node][0] + 1 != cells[node][1]:
                    cell_dict[node] = fact_red_normal.new()
                    label_dict[node] = 'fr n'
                else:
                    cell_dict[node] = input_node.new()
                    label_dict[node] = 'id'
            elif types[node] == 'op':
                cell_dict[node] = operator_node.new()
                label_dict[node] = "norm"
            elif types[node] == 'input':
                cell_dict[node] = input_node.new()
            elif types[node] == 'sum':
                cell_dict[node] = add_node.new()
            elif types[node] == 'cat':
                cell_dict[node] = output_node.new()

        super().__init__(cell_dict, darts_graph)


if __name__ == "__main__":
    space = DARTSSpace(num_cells=3)

    knobs = space.get_knobs()
    vec = get_random_cfg_vec(knobs=knobs)
    cfg = vec_to_knob(vec, knobs)
    instance = space.get(cfg)

    input_shape = np.array([1, 3, 32, 32])
    model = Model(instance, input_shape)

    input = torch.ones(tuple(input_shape))
    output = model(input)
    print(output.shape)
