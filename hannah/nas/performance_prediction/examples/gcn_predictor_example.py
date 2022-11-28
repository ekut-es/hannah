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
from tqdm import tqdm
from hannah.nas.performance_prediction.features.dataset import NASGraphDataset
from hannah.nas.performance_prediction.gcn.predictor import GCNPredictor, get_input_feature_size, prepare_dataloader

plt.style.use("seaborn")



# dataset = NASGraphDataset(cfg_space, edge_file, prop_file)
#/home/elia/Desktop/MA/performance_data/trained_models/dsd22_kws_10uw/conv_net_trax/performance_data
#/home/elia/Desktop/MA/hannah/experiments/dsd22/trained_models/dsd22_kws_10uw/conv_net_trax/performance_data

path_to_json_models= "/home/elia/Desktop/MA/performance_data/trained_models/dsd22_kws_10uw/conv_net_trax/performance_data"
dataset = NASGraphDataset(path_to_json_models)

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
#for epoch in tqdm(range(int(epochs)),desc='Training Epochs'):
for batched_graph, labels in tqdm(test_dataloader,desc='Training Epochs'):
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
