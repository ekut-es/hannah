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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from features.dataset import NASGraphDataset
from gcn.predictor import XGBPredictor, get_input_feature_size, prepare_dataloader
from search_space.space import NetworkSpace

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
predictor = XGBPredictor(
    in_feat, hidden_units=[512], embedding_size=10, readout="mean", xgb_param="default"
)

predictor.train_and_fit(
    train_dataloader,
    num_epochs=400,
    num_round=2000,
    learning_rate=0.001,
    validation_dataloader=test_dataloader,
    verbose=25,
)

pred_eval = pd.DataFrame(columns=["pred", "real"])
total_loss = 0
num_tests = 0
for batched_graph, labels in test_dataloader:
    pred = predictor.predict(batched_graph)
    for p, l in zip(pred, labels):
        pred_eval = pred_eval.append({"pred": p, "real": l.item()}, ignore_index=True)

    loss = F.mse_loss(torch.tensor(pred), labels, reduction="sum").item()
    total_loss += loss
    num_tests += len(labels)
print("MSE (Test): {}".format(total_loss / num_tests))


fig = plt.figure(figsize=(20, 8))
d = pred_eval.sort_values(by="real")
x = np.arange(len(d))
plt.scatter(x, d["pred"], label="Prediction")
plt.scatter(x, d["real"], label="Real Value")
plt.legend()
plt.ylabel("1 / cost (time in ms)")
# plt.savefig('../experiments/performance_prediction/prediction_acc_h50_b250_e400')
plt.show()
