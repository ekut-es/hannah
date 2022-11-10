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
import torch


class BatchNormalizationThroughTime1D(torch.nn.Module):
    def __init__(
        self,
        channels,
        timesteps: int = 0,
        eps: float = 1e-4,
        momentum: float = 0.1,
        variant="BNTTv1",
    ):
        super(BatchNormalizationThroughTime1D, self).__init__()
        self.variant = variant
        self.bnttlayer = torch.nn.ModuleList()
        for _ in range(timesteps):
            self.bnttlayer.append(
                torch.nn.BatchNorm1d(channels, eps=eps, momentum=momentum)
            )

    def forward(self, x):
        timesteps = x.shape[2]
        new = x.clone()
        for t in range(timesteps):
            if self.variant == "BNTTv1" or x.shape[0] == 1:
                new[:, :, t] = self.bnttlayer[t](x)[:, :, t]
            elif self.variant == "BNTTv2":
                new[:, :, t] = self.bnttlayer[t](x[:, :, t])
        x = new
        return x
