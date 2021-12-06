from itertools import repeat
import numpy as np
import networkx as nx
import torch

from hannah.nas.space.space import Space, Cell
from hannah.nas.space.operator import Convolution, Combine, Choice, Linear, Operator, Pooling, DepthwiseSeparableConvolution, Activation
from hannah.nas.space.utils import get_random_cfg_vec, vec_to_knob
from hannah.nas.space.model import Model
from hannah.nas.space.connectivity_constrainer import DARTSCell, DARTSGraph

num_cells = 2

num_nodes = 4
num_input = 2
num_output = 1
cells = repeat(DARTSCell(), num_cells)
darts_graph = DARTSGraph(cells)

dws_conv = DepthwiseSeparableConvolution(12, 3, 5, padding='same')
# dws_conv = Convolution(12, 3, padding='same')
identity = Operator()
identity.instantiate()
dws_conv.instantiate()
pool = Linear(10)
pool.instantiate()
act = Activation('relu')
act.instantiate()
add = Combine('add')
concat = Combine('concat')
choice = Choice([dws_conv, act])
operator_node = Cell()
input_node = Cell()
add_node = Cell()
output_node = Cell()
operator_node.add_sequential_operators([choice, add])
input_node.add_sequential_operators([identity])
add_node.add_sequential_operators([add])
output_node.add_sequential_operators([concat])

ct = 0
cell_dict = {}
types = nx.get_node_attributes(darts_graph.m_arch, 'type')
for node in darts_graph.m_arch:
    if types[node] == 'op':
        cell_dict[node] = operator_node.new()
    elif types[node] == 'identity':
        cell_dict[node] = input_node.new()
    elif types[node] == 'add':
        cell_dict[node] = add_node.new()
    elif types[node] == 'concat':
        cell_dict[node] = output_node.new()


print(len(cell_dict))
print(len(darts_graph.m_arch.nodes))

# cell_dict = {i: cell.new() for i in range(len(darts_graph.get_default_graph().nodes))}
space = Space(cell_dict, darts_graph)


knobs = space.get_knobs()
vec = get_random_cfg_vec(knobs=knobs)
cfg = vec_to_knob(vec, knobs)


instance = space.get(cfg)

input_shape = np.array([1, 12, 32, 32])
model = Model(instance, input_shape)

input = torch.ones(tuple(input_shape))
# print(input.shape)
output = model(input)
print(output.shape)
