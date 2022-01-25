from gcn.predictor import GCNPredictor, prepare_dataloader, get_input_feature_size
from search_space.space import NetworkSpace
from features.dataset import NASGraphDataset

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

plt.style.use("seaborn")


wd = "/home/moritz/Dokumente/Hiwi/code/nas/subgraph_generator"

# name of the configuration space
net = "test_net"

# where to find the data
data_name = "/data/test_net_tuned_jetsontx2"

cfg_space = NetworkSpace()
cfg_space.from_yaml(wd + "/configs/" + net + ".yaml")

# Used to construct the graph structure ( unnecessary because graph also constructed
# during graph_conversion)
# edge_file = wd + data_name + '/graph_defs/graph_edges.csv'

# properties, i.e. cost/latency
prop_file = wd + data_name + "/graph_defs/graph_properties.csv"

# dataset = NASGraphDataset(cfg_space, edge_file, prop_file)
dataset = NASGraphDataset(cfg_space, prop_file)

train_dataloader, test_dataloader = prepare_dataloader(
    dataset, batch_size=250, train_test_split=0.7
)
in_feat = get_input_feature_size(dataset)
predictor = GCNPredictor(in_feat, [512], "mean")
predictor.train(
    train_dataloader,
    num_epochs=400,
    learning_rate=0.001,
    validation_dataloader=test_dataloader,
)

pred_eval = pd.DataFrame(columns=["pred", "real"])
total_loss = 0
num_tests = 0
for batched_graph, labels in test_dataloader:
    pred = predictor.predict(batched_graph).squeeze()
    for p, l in zip(pred, labels):
        pred_eval = pred_eval.append(
            {"pred": p.item(), "real": l.item()}, ignore_index=True
        )


fig = plt.figure(figsize=(20, 8))
d = pred_eval.sort_values(by="real")
x = np.arange(len(d))
plt.scatter(x, d["pred"], label="Prediction")
plt.scatter(x, d["real"], label="Real Value")
plt.legend()
plt.ylabel("1 / cost (time in ms)")
# plt.savefig('../experiments/performance_prediction/prediction_acc_h50_b250_e400')
plt.show()