import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import hannah.nas.search_space.space as space
from hannah.models.factory.qconfig import QConfig, STEQuantize
from hannah.nas.search_space.connectivity_constrainer import PathConstrainer
from hannah.nas.search_space.operator import (Activation, Combine, Convolution,
                                              Linear, Pooling, Quantize)


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
            if isinstance(j, dict):
                for ki, vi in j.items():
                    flattened_knobs['{}_{}_{}'.format(str(k), str(i), str(ki))] = vi
            else:
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
            if isinstance(y, dict):
                cfg[k][x] = {}
                for ki, vi in y.items():
                    cfg[k][x][ki] = vi[vec[ct]]
                    ct += 1
            else:
                cfg[k][x] = y[vec[ct]]
                ct += 1
    return cfg


def draw_pretty(graph, labels=None, figsize=(20, 8), edge_labels=None, box=True, enum=False, vertical=False, label_color='white', save_as=None):
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

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    for edge in graph.edges:
        source, target = edge
        rad = 0.8
        rad = rad if int_nodes[source] % 2 else -rad

        arc = mpl.patches.ConnectionStyle.Arc3(rad)
        # arc = arc(pos[target], pos[source])

        ax.annotate("",
                    xy=pos[source],
                    xytext=pos[target],
                    arrowprops=dict(arrowstyle="-", color="black",
                                    connectionstyle=arc,  # f"arc3,rad={rad}",
                                    alpha=0.6,
                                    linewidth=1.5))
        if edge_labels:
            edge_pos = arc.connect(pos[source], pos[target]).vertices[1]
        #     edge_pos[0] *= 0# .95
            edge_pos[1] /= - (180 - (180/np.pi))
            ax.annotate(edge_labels[edge],
                        xy=edge_pos,
                        xycoords='data')

    nx.draw_networkx_nodes(graph, pos=pos, node_size=500, node_color='black')
    nx.draw_networkx_labels(graph, labels=labels, pos=pos, font_color=label_color)
    # if edge_labels:
    #     nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, font_size=6)
    # plt.margins(y=0.5)
    plt.box(box)

    if save_as:
        plt.savefig(save_as)
    # ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()
    return fig


def draw_pretty_nodes(graph, labels=None, figsize=(25, 7)):
    xpos = {}
    ypos = {}
    successors = nx.dfs_successors(graph)
    successors_ct = np.zeros(len(successors))
    even = 1
    for x, node in enumerate(nx.topological_sort(graph)):
        if isinstance(node, int):
            xpos[node] = node
            ypos[node] = 0
        elif isinstance(node, tuple):
            xpos[node] = ((node[1] - node[0]) / 2) + node[0]
            ypos[node] = successors_ct[node[0]] + 1
            if even == 1:
                ypos[node] *= -1
            even *= -1
            successors_ct[node[0]] += 1

    ymax = np.max(list(ypos.values()))
    ypos = {k: v + (v*ymax) if isinstance(k, tuple) else v for k, v in ypos.items()}
    pos = {n: (xpos[n], ypos[n]) for n in graph.nodes}

    fig = plt.figure(figsize=figsize)
    rad = 0.3
    nx.draw_networkx_nodes(graph, pos=pos, node_size=1000, node_color='black')
    for edge in graph.edges:
        if pos[edge[1]][1] > 0 and pos[edge[0]][1] == 0:
            r = -rad
        elif pos[edge[0]][1] > 0 and pos[edge[1]][1] == 0:
            r = -rad
        else:
            r = rad
        nx.draw_networkx_edges(graph, edgelist=[edge], pos=pos, connectionstyle="arc3,rad={}".format(r), alpha=0.6)
    labels = labels
    nx.draw_networkx_labels(graph, labels=labels, pos=pos, font_color='white')
    plt.show()

    return fig
