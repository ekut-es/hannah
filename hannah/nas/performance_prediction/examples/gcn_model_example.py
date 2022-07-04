import numpy as np
import torch
import torch.nn.functional as F
from dgl import data
from dgl.dataloading import GraphDataLoader
from features import dataset as ds
from gcn.model import GCN
from search_space.space import NetworkEntity, NetworkSpace
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
from tvm import relay

wd = "/home/moritz/Dokumente/Hiwi/code/nas/subgraph_generator"
name = "test_net"
inp = relay.var("input", shape=(1, 40, 101))

cfg_space = NetworkSpace()
cfg_space.from_yaml(wd + "/configs/" + name + ".yaml")
edge_file = wd + "/data/" + name + "/graph_defs/graph_edges.csv"
prop_file = wd + "/data/" + name + "/graph_defs/graph_properties.csv"

dataset = ds.NASGraphDataset(cfg_space, edge_file, prop_file)
# dataset.normalize_labels()
print("Feature Shape: ", dataset[0][0].ndata["features"].shape)


num_examples = np.arange(len(dataset), dtype=int)[1:]
num_train = int(len(num_examples) * 0.8)

indices = np.random.choice(num_examples, size=len(num_examples), replace=False)
train_indices = indices[:num_train]
test_indices = indices[num_train:]

train_indices = [i for i in train_indices if dataset[i][1] > 0]
test_indices = [i for i in test_indices if dataset[i][1] > 0]

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)


train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=5, drop_last=False
)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=5, drop_last=False
)

# Create the model with given dimensions
input_shape = dataset[0][0].ndata["features"].shape[1]
model = GCN(input_shape, [1024], 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

for epoch in range(200):
    # for batched_graph, labels in train_only_valid_dataloader:
    # print("Learning Rate:", optimizer.param_groups[0]['lr'])
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata["features"].float()).squeeze()
        # loss = F.mse_loss(pred, labels)
        loss = F.mse_loss(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # scheduler.step()

    total_loss = 0
    num_tests = 0
    # for batched_graph, labels in test_only_valid_dataloader:
    for batched_graph, labels in test_dataloader:
        pred = model(batched_graph, batched_graph.ndata["features"].float()).squeeze()
        loss = F.mse_loss(pred, labels).item()
        total_loss += loss
        num_tests += len(labels)
    print("Epoch {} MSE: {}".format(epoch, total_loss / num_tests))

test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=1, drop_last=False
)

for batched_graph, labels in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata["features"].float()).squeeze()
    print("Pred|Expected: {:>6.3f}|{:>6.3f}".format(pred, labels.item()))
    loss = F.mse_loss(pred, labels).item()
    total_loss += loss
    num_tests += len(labels)
