import dgl
from dgl.data import DGLDataset

from search_space import space
from .graph_conversion import to_dgl_graph, get_global_feature_options
import torch
import pandas as pd
import numpy as np


class NASGraphDataset(DGLDataset):
    def __init__(self, cfg_space, graph_properties_file, graph_edges_file=None):
        self.cfg_space = cfg_space
        self.graph_edges = graph_edges_file
        self.graph_properties = graph_properties_file
        super().__init__(name="nasgraph")

    def process(self):
        if self.graph_edges:
            edges = pd.read_csv(self.graph_edges)
        properties = pd.read_csv(self.graph_properties)

        # if graphs are duplicated in the dataset, average their cost
        properties = properties.groupby("graph_id", as_index=False).mean()
        self.graphs = []
        self.labels = []
        self.ids = []

        fopt = get_global_feature_options(self.cfg_space)

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            cost = row["label"]
            label = 0 if cost == 1e5 else 1 / cost
            # label = 1/cost

            label_dict[row["graph_id"]] = label
            num_nodes_dict[row["graph_id"]] = row["num_nodes"]

        if self.graph_edges:  # TODO: maybe delete this option altogether?
            # For the edges, first group the table by graph IDs.
            edges_group = edges.groupby("graph_id")
            # For each graph ID...
            for graph_id in edges_group.groups:
                if graph_id not in num_nodes_dict:
                    continue
                # Find the edges as well as the number of nodes and its label.
                edges_of_id = edges_group.get_group(graph_id)
                src = edges_of_id["src"].to_numpy()
                dst = edges_of_id["dst"].to_numpy()
                num_nodes = num_nodes_dict[graph_id]
                label = label_dict[graph_id]

                # Create a graph and add it to the list of graphs and labels.
                g = dgl.graph((src, dst), num_nodes=num_nodes)
                g = dgl.add_self_loop(g)
                cfg = space.point2knob(graph_id, self.cfg_space.collapsed_dims())
                net = space.NetworkEntity(
                    self.cfg_space, self.cfg_space.expand_config(cfg)
                )
                featured_graph = to_dgl_graph(net, fopt)

                features = featured_graph.ndata["features"]

                # append # of nodes to each feature row
                num_nodes_col = (
                    torch.Tensor([num_nodes]).repeat(features.shape[0]).unsqueeze(-1)
                )
                features = torch.hstack((num_nodes_col, features))

                g.ndata["features"] = features
                self.graphs.append(g)
                self.labels.append(label)
                self.ids.append(graph_id)
        else:
            for graph_id, label in label_dict.items():
                graph_id = int(graph_id)
                cfg = space.point2knob(graph_id, self.cfg_space.collapsed_dims())
                net = space.NetworkEntity(
                    self.cfg_space, self.cfg_space.expand_config(cfg)
                )
                g = to_dgl_graph(net, fopt)

                features = g.ndata["features"]
                # append # of nodes to each feature row
                num_nodes = num_nodes_dict[graph_id]
                num_nodes_col = (
                    torch.Tensor([num_nodes]).repeat(features.shape[0]).unsqueeze(-1)
                )
                features = torch.hstack((num_nodes_col, features))

                g.ndata["features"] = features

                self.graphs.append(g)
                self.labels.append(label)
                self.ids.append(graph_id)

        # Convert the label list to tensor for saving.
        self.labels = torch.FloatTensor(self.labels)

    def normalize_labels(self):
        max_label = self.labels.max()
        new_labels = []
        for l in self.labels:
            if l == -1:
                new_labels.append(-max_label)
            else:
                new_labels.append(l)
        self.labels = torch.FloatTensor(new_labels)

        std = self.labels.std()
        mean = self.labels.mean()
        self.labels = (self.labels - mean) / std

    def normalize_features(self, max_feature):
        for g in self.graphs:
            new_features = g.ndata["features"].clone()
            for row in range(len(new_features)):
                new_features[row] = new_features[row] / max_feature
            g.ndata["features"] = new_features

    def close_label_gap(self, gap=0.2):
        # ignore the first one (outlier in current dataset)
        non_zero = np.sort([l for l in self.labels if l > 0])[1:]
        min_non_zero = np.min(non_zero)
        new_labels = []

        for l in self.labels:
            if l > 1.5:
                new_labels.append(l - 1.5)
            else:
                new_labels.append(l)
        self.labels = torch.FloatTensor(new_labels)

    def to_class_labels(self):
        new_labels = []
        for l in self.labels:
            if l > 0:
                new_labels.append(1)
            else:
                new_labels.append(0)
        self.float_labels = self.labels.clone()
        self.labels = torch.LongTensor(new_labels)

    def get_graph_id(self, i):
        return self.ids[i]

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)
