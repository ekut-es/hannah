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
