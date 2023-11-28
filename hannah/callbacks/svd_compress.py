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
from pytorch_lightning.callbacks import Callback

""" Singular Value Decomposition of the linear layer of a neural network - tested for conv-net-trax and tc-res8"""


class SVD(Callback):
    """ """

    def __init__(self, rank_compression, compress_after):
        self.rank_compression = rank_compression
        self.compress_after = compress_after
        super().__init__()

    def on_train_epoch_start(self, trainer, pl_module):
        """

        Args:
          trainer:
          pl_module:

        Returns:

        """
        # Train - apply SVD - restructure - retrain
        if trainer.current_epoch == self.compress_after / 2:
            with torch.no_grad():
                for name, module in pl_module.named_modules():
                    # First case: conv-net-trax model with Sequential Layers
                    if name == "model.linear.0.0" and not isinstance(
                        pl_module.model.linear[0][0], nn.Sequential
                    ):
                        U, S, Vh = torch.linalg.svd(
                            module.weight.detach(), full_matrices=True
                        )  # apply SVD

                        # Slicing of matrices for rank r and reassembly
                        U = U[:, : self.rank_compression]
                        SVh = torch.matmul(torch.diag(S), Vh[: S.size()[0], :])
                        SVh = SVh[: self.rank_compression, :]

                        """Replace linear layer by sequential layer with two linear layers,
                        one containing SVh and the other U, approximating the original fully connected layer"""
                        original_fc = pl_module.model.linear[0][0]
                        new_fc = nn.Sequential(
                            nn.Linear(
                                original_fc.in_features,
                                self.rank_compression,
                                bias=original_fc.bias,
                            ),
                            nn.Linear(
                                self.rank_compression,
                                original_fc.out_features,
                                bias=original_fc.bias,
                            ),
                        )
                        pl_module.model.linear[0][0] = new_fc
                        pl_module.model.linear[0][0][0].weight = torch.nn.Parameter(
                            SVh, requires_grad=True
                        )
                        pl_module.model.linear[0][0][1].weight = torch.nn.Parameter(
                            U, requires_grad=True
                        )

                    # Second case: tc-res8 model
                    elif (
                        type(module) in [nn.Linear]
                        and name != "model.linear.0.0.0"
                        and name != "model.linear.0.0.1"
                        and not isinstance(pl_module.model.fc, nn.Sequential)
                    ):
                        U, S, Vh = torch.linalg.svd(
                            module.weight.detach(), full_matrices=True
                        )
                        U = U[:, : self.rank_compression]
                        SVh = torch.matmul(torch.diag(S), Vh[: S.size()[0], :])
                        SVh = SVh[: self.rank_compression, :]
                        original_fc = pl_module.model.fc
                        new_fc = nn.Sequential(
                            nn.Linear(
                                original_fc.in_features,
                                self.rank_compression,
                                bias=original_fc.bias,
                            ),
                            nn.Linear(
                                self.rank_compression,
                                original_fc.out_features,
                                bias=original_fc.bias,
                            ),
                        )
                        pl_module.model.fc = new_fc
                        pl_module.model.fc[0].weight = torch.nn.Parameter(
                            SVh, requires_grad=True
                        )
                        pl_module.model.fc[1].weight = torch.nn.Parameter(
                            U, requires_grad=True
                        )

        return pl_module
