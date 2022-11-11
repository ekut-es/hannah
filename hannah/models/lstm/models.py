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
import torch.nn as nn


class LSTMModel(nn.Module):
    """Simple LSTM model."""

    def __init__(self, config):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=config["height"],
            hidden_size=config["hidden_size"],
            num_layers=config["n_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )
        self.dropout = nn.Dropout(config["dropout"])
        self.fc = nn.Linear(config["hidden_size"], config["n_labels"])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        _, (ht, _) = self.lstm(x)
        x = self.dropout(ht[-1])
        x = self.fc(x)
        return x
