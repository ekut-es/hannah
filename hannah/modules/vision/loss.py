#
# Copyright (c) 2023 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
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
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemiSupervisedLoss(nn.Module):
    def __init__(self, kind="normal"):
        self.kind = kind
        if self.kind == "normal":
            self.exponent = 1
        elif self.kind == "anomaly":
            self.exponent = (
                -1
            )  # TODO: ADD temperature scaling to adapt loss accordingly
        else:
            raise ValueError("Kind: " + kind + " not supported.")
        super().__init__()

    def forward(
        self,
        true_data=None,
        labels=None,
        logits=None,
        decoded=None,
        boxes=None,
        weight=None,
        label_smoothing=0.0,
    ):
        loss = torch.tensor([0.0])

        if labels is not None and logits.numel() > 0:  # all labeled + classifier
            loss = F.cross_entropy(
                logits, labels, weight=weight, label_smoothing=label_smoothing
            )

        elif decoded is not None:  # Autoencoder
            loss = F.mse_loss(decoded, true_data) ** self.exponent

        return loss
