import os
import sys

import dgl
import networkx as nx
import numpy as np
import torch
from search_space import space

# import pandas as pd


def to_one_hot(val, options):
    vec = np.zeros(len(options))
    options = np.array(options)
    try:
        # print(val)
        # print(options)
        index = np.where(val == options)[0][0]
        vec[index] = 1
    except Exception as e:
        pass
    return vec


def get_feature_options(net):
    features = {"op": set()}
    for block in net.blocks:
        for layer in block.layers:
            features["op"].add(layer.type)
            for k, v in layer.entity_map.items():
                if k not in features:
                    features[k] = set()
                features[k].add(v)
        for layer in block.residual:
            features["op"].add(layer.type)
            for k, v in layer.entity_map.items():
                if k not in features:
                    features[k] = set()
                features[k].add(v)
        if block.residual:
            features["op"].add("add")
    features_ = {}
    for k, v in features.items():
        features_[k] = list(v)
    return features_


def get_global_feature_options(cfg_space):
    dims = cfg_space.collapsed_dims()
    global_options = {}

    # for each choice generate one cfg with this choice
    # sum(dims) cfgs will be generated
    for dim, choices in enumerate(cfg_space.collapsed_dims()):
        for choice in range(choices):
            cfg = np.zeros(len(dims), dtype=int)
            cfg[dim] = int(choice)
            # idx = list(idx)
            net = space.NetworkEntity(cfg_space, cfg_space.expand_config(cfg))
            new_options = get_feature_options(net)

            # fuse the options from the new net with the already established
            # options
            new_global_options = {}
            for k in global_options.keys():
                new_global_options[k] = global_options[k]
            for k in new_options.keys():
                if k in global_options:
                    new_global_options[k] = list(
                        set(global_options[k]).union(new_options[k])
                    )
                else:
                    new_global_options[k] = new_options[k]
            global_options = new_global_options
    return global_options


def get_feature_flatten(layer, options):
    features = []
    for key, opts in options.items():
        if key in layer.entity_map or key == "op":
            if key == "op":
                val = layer.type
            else:
                val = layer.entity_map[key]
            if isinstance(val, int):
                # features.append(to_one_hot(val, opts))
                features.append(val)
            else:
                features.append(to_one_hot(val, opts))
        else:
            if isinstance(opts[0], int):
                features.append(-1)
                # features.append(to_one_hot(0, opts))
            else:
                features.append(to_one_hot(0, opts))

    features = np.hstack(features)
    # features = np.pad(features, (0, 20 - len(features)), mode='constant', constant_values=-1)

    return features


def addition_features(options):
    features = []
    for key, opts in options.items():
        if key == "op":
            if key == "op":
                val = "add"
            else:
                val = -1
            if isinstance(val, int):
                # features.append(to_one_hot(val, opts))
                features.append(val)
            else:
                features.append(to_one_hot(val, opts))
        else:
            if isinstance(opts[0], int):
                # features.append(to_one_hot(0, opts))
                features.append(-1)
            else:
                features.append(to_one_hot(0, opts))
    features = np.hstack(features)
    # features = np.pad(features, (0, 20 - len(features)), mode='constant', constant_values=-1)
    return features


def to_networkx_graph(net, fopt=None):
    G = nx.Graph()
    if not fopt:
        fopt = get_feature_options(net)
    i = 0
    for block in net.blocks:
        block_i = i
        block_num = 0
        for layer in block.layers:
            G.add_node(i, op=layer.type, features=get_feature_flatten(layer, fopt))
            if i > 0:
                G.add_edge(i - 1, i)
            i += 1
            block_num += 1
        for layer in block.residual:
            G.add_node(i, op=layer.type, features=get_feature_flatten(layer, fopt))
            if i == block_i + block_num:
                G.add_edge(block_i - 1, i)
            else:
                G.add_edge(i - 1, i)
            i += 1
        if block.residual:
            G.add_node(i, op="add", features=addition_features(fopt))
            G.add_edge(block_i + block_num - 1, i)
            G.add_edge(i - 1, i)
            i += 1
    return G


def to_dgl_graph(net, fopt=None):
    g = to_networkx_graph(net, fopt)
    src = []
    dst = []
    for i, j in g.edges:
        src.append(i)
        dst.append(j)

    fea_tensor = []
    for n in g.nodes:
        fea_tensor.append(torch.tensor(g.nodes[n]["features"]))
    fea_tensor = torch.vstack(fea_tensor)
    g = dgl.graph(data=(src, dst))
    g.ndata["features"] = fea_tensor
    g = dgl.add_self_loop(g)

    return g
