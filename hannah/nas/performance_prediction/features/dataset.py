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

import hannah.conf
from hannah.nas.graph_conversion import model_to_graph


class NASGraphDataset(DGLDataset):
    def __init__(self, result_folder: str):
        self.result_folder = Path(result_folder)
        super().__init__(name="nasgraph")

    def process(self):
        self.nx_graphs = []
        self.graphs = []
        self.labels = []
        assert self.result_folder.exists()

        data_dict = {}
        ct = 0
        for i, data_path in enumerate(self.result_folder.glob("model_*.json")):
            if i % 500 == 0:
                print("Processing graph {}".format(i))

            d = json.load(data_path.open())

            graph = nx.json_graph.node_link_graph(d["graph"])
            self.nx_graphs.append(graph)
            for j, n in enumerate(graph.nodes):
                node = graph.nodes[n]
                row_as_dict = dict({"graph": i, "node": j}, **flatten(node))
                data_dict[ct] = row_as_dict
                ct = ct + 1

            metrics = d["metrics"]
            if metrics.get("val_error", None):
                label = metrics["val_error"]
            else:
                label = 1 / metrics["latency"]
            self.labels.append(label)

        df = pd.DataFrame.from_dict(data_dict, orient="index")
        df = df.dropna(axis=1, how="all")

        columns_to_drop = [c for c in df.columns if "name" in c]
        df = df.drop(columns=columns_to_drop)

        cat_columns = [c for c in df.columns if "type" in c or "method" in c]
        df = pd.get_dummies(df, columns=cat_columns).fillna(0)

        array_columns = [
            c
            for c in df.columns
            if "shape" in c
            or "size" in c
            or "dilation" in c
            or "channels" in c
            or "stride" in c
            or "padding" in c
        ]
        df = unnest(df, array_columns, axis=0).fillna(0)
        # for col in df.columns:
        #     if col not in ['graph', 'node'] + cat_columns:
        #         div = (df[col].max()-df[col].min())
        #         if not div:
        #             div = 1
        #         df[col] = (df[col]-df[col].min())/ div
        for i, g in enumerate(self.nx_graphs):
            dgl_graph = to_dgl_graph(
                g, df[df["graph"] == i].drop(columns="node").to_numpy()
            )
            self.graphs.append(dgl_graph)

        self.labels = torch.FloatTensor(self.labels)

    def normalize_labels(self):
        std = self.labels.std()
        mean = self.labels.mean()
        self.labels = (self.labels - mean) / std

    def normalize_features(self, max_feature):
        for g in self.graphs:
            new_features = g.ndata["features"].clone()
            for row in range(len(new_features)):
                new_features[row] = new_features[row] / max_feature
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


def to_dgl_graph(nx_graph, features):
    node_num = {}
    for num, n in enumerate(nx_graph.nodes):
        node_num[n] = num
    src = []
    dst = []
    for i, j in nx_graph.edges:
        src.append(node_num[i])
        dst.append(node_num[j])

    g = dgl.graph(data=(src, dst))
    g.ndata["features"] = torch.Tensor(features)

    # Self-loops allow the network to include the node's own representation in the aggregation
    g = dgl.add_self_loop(g)

    return g


#  modified from https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        elif isinstance(v, list) and isinstance(v[0], dict):

            for i, item in enumerate(v):
                if i > 0:
                    new_key = new_key[:-2]  # remove last i
                new_key = new_key + sep + str(i)
                items.extend(flatten(item, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# modified from https://stackoverflow.com/questions/53218931/how-to-unnest-explode-a-column-in-a-pandas-dataframe
def unnest(df, explode, axis):
    if axis == 1:
        idx = df.index.repeat(df[explode[0]].str.len())
        df1 = pd.concat(
            [pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1
        )
        df1.index = idx

        return df1.join(df.drop(explode, 1), how="left")
    else:
        df1 = pd.concat(
            [
                pd.DataFrame(
                    [
                        item if isinstance(item, list) else [item]
                        for item in df[x].tolist()
                    ],
                    index=df.index,
                ).add_prefix(x)
                for x in explode
            ],
            axis=1,
        )
        return df1.join(df.drop(explode, 1), how="left")


@hydra.main(config_path="../../../conf", config_name="config")
def main(config):
    dataset = NASGraphDataset(
        "/home/moritz/projects/hannah/experiments/trained_models/debug_random_nas/performance_data"
    )
    for item in dataset:
        print(item)


if __name__ == "__main__":
    main()
