import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from hannah.nas.space.connectivity_constrainer import PathConstrainer
from hannah.nas.space.operator import Convolution, Activation, Combine, Quantize, Linear, Pooling
import hannah.nas.space.space as space
from hannah.models.factory.qconfig import QConfig, STEQuantize


def get_node_coord(g, first_node=0):
    positions = {first_node: [0, 0]}
    prev = 0
    d = 0
    for e in nx.edge_dfs(g):
        x = e[1] if isinstance(e[1], int) else int(e[1].split('_')[0])
        if x < prev:
            d += 1
        # else:
        #     d = min(0, d-1)
        prev = x
        positions[e[1]] = [x, d]

    cur_x = -1
    ct = 1
    step = 1
    for key, pos in positions.items():
        if pos[0] == cur_x:
            pos[0] = cur_x + ct
            ct += step
        else:
            cur_x = pos[0]
            ct = step

    return positions


def get_example_cell_dict(num_nodes=5):
    qc = QConfig(
            STEQuantize.with_args(
                bits=8
            ),
            STEQuantize.with_args(
                bits=8
            ),
            STEQuantize.with_args(
                bits=8
            ),
        )

    quant_args = {'bits': 8}
    # define basic operators
    conv = Convolution(kernel_size=[1, 3, 5], out_channels=[12, 24, 48], qconfig=qc, batch_norm=True)
    act = Activation('relu')
    add = Combine(mode='concat')
    act_quant = Quantize(quant=[quant_args])
    linear = Linear(out_features=12)
    pool = Pooling(kernel_size=3, stride=2, padding=0)

    # initiate cells, i.e. microstructures
    first_cell = space.Cell()
    standard_cell = space.Cell()
    final_cell = space.Cell()

    # add operators to cells
    first_cell.add_sequential_operators([act_quant, conv, act, act_quant])
    standard_cell.add_sequential_operators([conv, act_quant, add, act])
    final_cell.add_sequential_operators([add, pool, linear])

    # create cell dict that maps a connectivity graph node to a cell
    cell_dict = {i: standard_cell.new() for i in range(num_nodes)}
    cell_dict[0] = first_cell
    cell_dict[num_nodes-1] = final_cell

    return cell_dict


def get_example_space():
    cell_dict = get_example_cell_dict()
    search_space = space.Space(cell_dict)
    return search_space


def get_example_subgraph(max_parallel_paths=2, num_nodes=5):
    cc = PathConstrainer(max_parallel_paths, num_nodes, None)

    random_c_graph = cc.get_random_dag()

    cell_dict = get_example_cell_dict(num_nodes)

    subgraph = space.Subgraph(cells=cell_dict, connectivity_graph=random_c_graph)
    return subgraph


def get_random_cfg(subgraph, random_state=None):
    cfg_dims = []
    for node in subgraph.nodes():
        for key, attr in node.attrs.items():
            # cfg_dims.append(len(feature_options[key]))
            cfg_dims.append(len(attr))
    cfg = []
    for d in cfg_dims:
        if random_state:
            cfg.append(random_state.choice(d))
        else:
            cfg.append(np.random.choice(d))

    return cfg


def get_example_instance():
    subgraph = get_example_subgraph()
    cfg = get_random_cfg(subgraph)
    instance = subgraph.instantiate(cfg)
    return instance


def flatten_knobs(knobs):
    flattened_knobs = {}
    for k, v in knobs.items():
        for i, j in v.items():
            flattened_knobs['{}_{}'.format(str(k), str(i))] = j
    return flattened_knobs


def get_random_cfg_vec(knobs, random_state=None):
    flattened_knobs = flatten_knobs(knobs)
    vec = []
    for k, v in flattened_knobs.items():
        if random_state is None:
            vec.append(np.random.choice(range(len(v))))
        else:
            vec.append(random_state.choice(range(len(v))))
    return vec


def knob_to_vec(cfg):
    cfg_vec = []
    for k, v in cfg.items():
        for i, j in v.items():
            cfg.append(j)
    return cfg_vec


def vec_to_knob(vec, knobs):
    cfg = {}
    ct = 0
    for k, v in knobs.items():
        cfg[k] = {}
        for x, y in v.items():
            cfg[k][x] = y[vec[ct]]
            ct += 1
    return cfg


def draw_pretty(graph, labels, figsize=(20, 8), box=True, enum=False, vertical=False, label_color='white', save_as=None):
    if enum:
        if vertical:
            pos = {node: (0, i) for i, node in enumerate(nx.topological_sort(graph))}
        else:
            pos = {node: (i, 0) for i, node in enumerate(nx.topological_sort(graph))}
    else:
        if vertical:
            pos = {node: (0, node) for node in graph.nodes()}
        else:
            pos = {node: (node, 0) for node in graph.nodes()}
    int_nodes = {n: i for i, n in enumerate(nx.topological_sort(graph))}

    plt.figure(figsize=figsize)
    ax = plt.gca()
    for edge in graph.edges:
        source, target = edge
        rad = 0.8
        rad = rad if int_nodes[source] % 2 else -rad
        ax.annotate("",
                    xy=pos[source],
                    xytext=pos[target],
                    arrowprops=dict(arrowstyle="-", color="black",
                                    connectionstyle=f"arc3,rad={rad}",
                                    alpha=0.6,
                                    linewidth=1.5))
    nx.draw_networkx_nodes(graph, pos=pos, node_size=500, node_color='black')
    nx.draw_networkx_labels(graph, labels=labels, pos=pos, font_color=label_color)
    # plt.margins(y=0.5)
    plt.box(box)
    if save_as:
        plt.savefig(save_as)
    plt.show()
