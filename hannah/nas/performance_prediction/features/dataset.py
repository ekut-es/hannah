#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import collections
import json
from pathlib import Path

import dgl
import hydra
import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from dgl.data import DGLDataset
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import MinMaxScaler


# FIXME: Find better way
COLUMNS = ['attrs_dilation', 'attrs_groups', 'attrs_in_channels',
           'attrs_in_features', 'attrs_kernel_size', 'attrs_out_channels',
           'attrs_out_features', 'attrs_padding', 'attrs_stride', 'bias_nan',
           'output_quant_bits', 'output_quant_dtype_float',
           'output_quant_dtype_nan', 'output_quant_method_nan',
           'output_quant_method_none', 'output_quant_nan', 'output_shape_0',
           'output_shape_1', 'output_shape_2', 'output_shape_3', 'type_add',
           'type_batch_norm', 'type_conv', 'type_flatten', 'type_linear',
           'type_nan', 'type_placeholder', 'type_pooling', 'type_relu',
           'type_tensor', 'weight_quant_nan', 'weight_shape_0', 'weight_shape_1',
           'weight_shape_2', 'weight_shape_3']

class NASGraphDataset(DGLDataset):
    def __init__(self, result_folder: str):
        self.result_folder = Path(result_folder)
        super().__init__(name="nasgraph")

    def process(self):
        self.nx_graphs = []
        self.graphs = []
        self.labels = []
        assert self.result_folder.exists()


        for i, data_path in enumerate(self.result_folder.glob("model_*.json")):
            if i % 500 == 0:
                print("Processing graph {}".format(i))

            d = json.load(data_path.open())

            graph = nx.json_graph.node_link_graph(d["graph"])
            self.nx_graphs.append(graph)
            fea = get_features(graph)
            for i, n in enumerate(graph.nodes):
                graph.nodes[n]['features'] = fea.iloc[i].to_numpy()
            dgl_graph = to_dgl_graph(graph)

            metrics = d["metrics"]
            if metrics.get("val_error", None) is not None:
                if metrics['val_error'] == 0:
                    label = float('inf')
                else:
                    label = metrics["val_error"]
            else:
                label = 1 / metrics["latency"]

            self.labels.append(label)
            self.graphs.append(dgl_graph)

        self.labels = torch.FloatTensor(self.labels)

    def normalize_labels(self):
        std = self.labels.std()
        mean = self.labels.mean()
        self.labels = (self.labels - mean) / std

    def normalize_features(self, max_feature=None):
        max_feature = np.zeros(self.graphs[0].ndata['features'].shape[1])
        min_feature = np.zeros(self.graphs[0].ndata['features'].shape[1])
        for g in self.graphs:
            maximums = g.ndata['features'].max(axis=0)[0]
            minimums = g.ndata['features'].min(axis=0)[0]
            for row in range(len(maximums)):
                max_feature[row] = max(max_feature[row], maximums[row])
                min_feature[row] = min(min_feature[row], minimums[row])

        for g in self.graphs:
            new_features = g.ndata["features"].clone()
            for row in range(len(new_features)):
                for col in range(len(new_features[row])):
                    # max_feature = max(new_features[row])
                    new_features[row][col] = (new_features[row][col] - min_feature[col]) / ((max_feature[col] - min_feature[col]) + 1e-6)
            g.ndata["features"] = new_features

    def to_class_labels(self):
        new_labels = []
        for label in self.labels:
            if label > 0:
                new_labels.append(1)
            else:
                new_labels.append(0)
        self.float_labels = self.labels.clone()
        self.labels = torch.LongTensor(new_labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


class OnlineNASGraphDataset(DGLDataset):
    def __init__(self, dgl_graphs, labels):
        super().__init__(name="OnlineNASGraphDataset")
        self.graphs = dgl_graphs
        self.labels = torch.FloatTensor(labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


def get_features(nx_graph):
    dataframes = []
    for n in nx_graph.nodes:
        df = pd.json_normalize(nx_graph.nodes[n], sep='_')
        if 'inputs' in df:
            df.drop(columns="inputs", inplace=True)
        if 'output_name' in df:
            df.drop(columns="output_name", inplace=True)
        df = unfold_columns(df, columns=get_list_columns(df))
        dataframes.append(df)
    df = pd.concat(dataframes)
    # df.dropna(axis = 0, how = 'all', inplace = True)
    df = pd.get_dummies(df, dummy_na=True)
    df = df.fillna(0)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = 0
    df = df.reindex(sorted(df.columns), axis=1)  # Sort to have consistency
    return df.astype(np.float32)

def get_list_columns(df):
    list_cols = []
    for col in df.keys():
        if isinstance(df[col][0], (list, tuple, torch.Size)):
            list_cols.append(col)
    return list_cols

# modified from
# https://stackoverflow.com/questions/35491274/split-a-pandas-column-of-lists-into-multiple-columns
def unfold_columns(df, columns=[], strict=False):
    assert isinstance(columns, list), "Columns should be a list of column names"
    if len(columns) == 0:
        columns = [
            column for column in df.columns
            if df.applymap(lambda x: isinstance(x, (list, tuple, torch.Size))).all()[column]
        ]
    else:
        assert(all([(column in df.columns) for column in columns])), \
            "Not all given columns are found in df"
    columns_order = df.columns
    for column_name in columns:
        if df[column_name].apply(lambda x: isinstance(x, (list, tuple, torch.Size))).all():
            if strict:
                assert len(set(df[column_name].apply(lambda x: len(x)))) == 1, \
                    f"Lists in df['{column_name}'] are not of equal length"
            unfolded = pd.DataFrame(df[column_name].tolist())
            unfolded.columns = [f'{column_name}_{x}' for x in unfolded.columns]
            columns_order = [
                *columns_order[:list(columns_order).index(column_name)],
                *unfolded.columns,
                *columns_order[list(columns_order).index(column_name)+1:]
            ]
            df = df.join(unfolded).drop([column_name], axis=1)
    return df[columns_order]



def to_dgl_graph(nx_graph):

    node_num = {}
    fea_tensor = []
    fea_len = 0
    for num, n in enumerate(nx_graph.nodes):
        feature_vec = torch.tensor(nx_graph.nodes[n]["features"])
        fea_len = max(len(feature_vec), fea_len)
        fea_tensor.append(feature_vec)
        node_num[n] = num

    padded_features = []
    for vec in fea_tensor:
        padded_vec = np.pad(
            vec, (0, fea_len - len(vec)), mode="constant", constant_values=0
        )
        padded_vec = torch.tensor(padded_vec)
        padded_features.append(padded_vec)

    src = []
    dst = []
    for i, j in nx_graph.edges:
        src.append(node_num[i])
        dst.append(node_num[j])

    fea_tensor = torch.vstack(padded_features)
    g = dgl.graph(data=(src, dst))
    g.ndata["features"] = fea_tensor
    g = dgl.add_self_loop(g)

    return g
