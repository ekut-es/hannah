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
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, readout="mean"):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats[0])
        self.convs = []
        for i in range(len(h_feats[0:]) - 1):
            self.convs.append(GraphConv(h_feats[i], h_feats[i + 1]))
        self.conv2 = GraphConv(h_feats[-1], num_classes)

        self.readout = readout

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        for conv in self.convs:
            h = conv(g, h)
            h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata["h"] = h
        readout = dgl.readout_nodes(g, feat="h", weight=None, op=self.readout)
        return readout


class GCNEmbedding(nn.Module):
    def __init__(self, in_feats, h_feats, embedding_size, readout="mean"):
        super(GCNEmbedding, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats[0])
        self.convs = []
        for i in range(len(h_feats[0:]) - 1):
            self.convs.append(GraphConv(h_feats[i], h_feats[i + 1]))
        self.conv2 = GraphConv(h_feats[-1], embedding_size)
        self.readout = readout
        self.fc = nn.Linear(embedding_size, 1)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        for conv in self.convs:
            h = conv(g, h)
            h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata["h"] = h
        readout = dgl.readout_nodes(g, feat="h", weight=None, op=self.readout)
        output = self.fc(readout)
        return output

    def get_embedding(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        for conv in self.convs:
            h = conv(g, h)
            h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata["h"] = h
        readout = dgl.readout_nodes(g, feat="h", weight=None, op=self.readout)
        return readout
