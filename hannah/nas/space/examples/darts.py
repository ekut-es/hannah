from itertools import repeat
import numpy as np
import networkx as nx
import torch

from hannah.nas.space.space import Space, Cell
from hannah.nas.space.operator import *
from hannah.nas.space.utils import get_random_cfg_vec, vec_to_knob
from hannah.nas.space.model import Model
from hannah.nas.space.connectivity_constrainer import DARTSCell, DARTSGraph

num_cells = 3

num_nodes = 4
num_input = 2
num_output = 1
cells = repeat(DARTSCell(), num_cells)
darts_graph = DARTSGraph(cells, redux_cell_indices=[1])

dw_conv = DepthwiseSeparableConvolution(12, kernel_size=[3, 5],  dilation=[1, 2], padding='same')
dw_conv_redux = DepthwiseSeparableConvolution(12, kernel_size=[3, 5], dilation=[1, 2], padding='half', stride=2)

identity = Operator()
factorized_reduce_normal = FactorizedReduce(stride=1, out_channels=12)
factorized_reduce_redux = FactorizedReduce(stride=2, out_channels=12)

pool = Pooling(stride=2, padding=1)
zero = Zero()
act = Activation('relu')
add = Combine('add')
add_1x1 = Combine('add', one_by_one_conv=[True])
concat = Combine('concat')
choice = Choice([dw_conv, factorized_reduce_normal])
choice_redux = Choice([dw_conv_redux, factorized_reduce_redux])


def double_channel(x):
    return 2*x


factorized_reduce = FactorizedReduce(stride=2, out_channels=double_channel)

operator_node = Cell([choice, add])
input_node = Cell([identity])
add_node = Cell([add_1x1])
output_node = Cell([concat])
fact_red_cell = Cell([concat, factorized_reduce])

operator_redux = Cell([choice_redux, add])

ct = 0
cell_dict = {}
types = nx.get_node_attributes(darts_graph.m_arch, 'type')
cells = nx.get_node_attributes(darts_graph.m_arch, 'cell')
for node in darts_graph.m_arch:
    if types[node] == 'op':
        cell_dict[node] = operator_node.new()
    elif types[node] == 'redux':
        cell_dict[node] = operator_redux.new()
    elif types[node] == 'factorize_reduce':
        cell_dict[node] = fact_red_cell.new()
    elif types[node] == 'identity':
        cell_dict[node] = input_node.new()
    elif types[node] == 'add':
        cell_dict[node] = add_node.new()
    elif types[node] == 'concat':
        cell_dict[node] = output_node.new()

space = Space(cell_dict, darts_graph)



knobs = space.get_knobs()
vec = get_random_cfg_vec(knobs=knobs)
cfg = vec_to_knob(vec, knobs)
instance = space.get(cfg)

input_shape = np.array([1, 3, 32, 32])
model = Model(instance, input_shape)

input = torch.ones(tuple(input_shape))
# print(input.shape)
output = model(input)
print(output.shape)
