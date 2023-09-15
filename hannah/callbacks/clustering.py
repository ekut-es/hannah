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
import logging

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans, MiniBatchKMeans

from ..models.factory.qat import Conv1d, ConvBn1d, ConvBnReLU1d, ConvReLU1d

logger = logging.getLogger(__name__)


def clustering(params, inertia, cluster):
    """

    Args:
      params:
      inertia:
      cluster:

    Returns:

    """
    sparse_matrix = csr_matrix(params)
    kmeans = MiniBatchKMeans(n_clusters=cluster, init="k-means++", random_state=1234)
    kmeans.fit(sparse_matrix.reshape(-1, 1))
    centers = kmeans.cluster_centers_.reshape(-1)
    inertia += kmeans.inertia_
    return centers, inertia


class kMeans(Callback):
    """ """

    def __init__(self, cluster):
        self.cluster = cluster

    def on_fit_end(self, trainer, pl_module):
        """

        Args:
          trainer:
          pl_module:

        Returns:

        """
        with torch.no_grad():
            for module in pl_module.modules():
                if hasattr(module, "scaled_weight"):
                    module.weight.data = module.scaled_weight
                    if not isinstance(module, nn.Linear):
                        bias_shape = [1] * len(module.weight.shape)
                        bias_shape[1] = -1
                        bias = torch.zeros(
                            module.out_channels, device=module.weight.device
                        )
                        bias = module.bias_fake_quant(
                            (bias - module.bn.running_mean) * module.scale_factor
                            + module.bn.bias
                        )  # .reshape(bias_shape) #.view(-1, 1, 1) #.reshape(bias_shape)
                        module.bias = torch.nn.Parameter(bias)

        def replace_modules(module):
            """

            Args:
              module:

            Returns:

            """
            for name, child in module.named_children():
                replace_modules(child)

                if isinstance(child, ConvBn1d):
                    tmp = Conv1d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        groups=child.groups,
                        padding_mode=child.padding_mode,
                        dilation=child.dilation,
                        bias=True,
                        qconfig=child.qconfig,
                    )
                    tmp.weight.data = child.weight
                    tmp.bias = child.bias
                    setattr(module, name, tmp)

                if isinstance(child, ConvBnReLU1d):
                    tmp = ConvReLU1d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        groups=child.groups,
                        padding_mode=child.padding_mode,
                        bias=True,
                        dilation=child.dilation,
                        qconfig=child.qconfig,
                    )
                    tmp.weight.data = child.weight
                    tmp.bias = child.bias
                    setattr(module, name, tmp)

        device = pl_module.device
        replace_modules(pl_module)
        pl_module.to(device=device)  # otherwise cuda error
        inertia = 0
        for name, module in pl_module.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                params = module.weight.data.cpu().numpy().flatten()
                distinct_weights = np.unique(params).size
                if self.cluster < distinct_weights:
                    centers, inertia = clustering(params, inertia, self.cluster)
                else:
                    continue

                # Returns center that is closest to given value x
                def replace_values_by_centers(x):
                    """

                    Args:
                      x:

                    Returns:

                    """
                    if x == 0.0:  # required if pruning is used
                        return x
                    else:
                        i = (np.abs(centers - x)).argmin()
                        return centers[i]

                module.weight.data = module.weight.data.cpu().apply_(
                    replace_values_by_centers
                )  # _ symbolizes inplace function, tensor moved to cpu, since apply_() only works that way
                module.to(device=device)  # move from cpu to gpu
        logger.critical("Clustering error: %f", float(inertia))

    def on_epoch_end(self, trainer, pl_module):
        """

        Args:
          trainer:
          pl_module:

        Returns:

        """
        inertia = 0
        device = pl_module.device
        for module in pl_module.modules():
            if hasattr(module, "weight") and module.weight is not None:
                w = module.weight.data.cpu().numpy().flatten()
                distinct_weights = np.unique(w).size
                if self.cluster < distinct_weights:
                    centers, inertia = clustering(w, inertia, self.cluster)
                else:
                    continue

                def replace_values_by_centers(x):
                    """

                    Args:
                      x:

                    Returns:

                    """
                    if x == 0.0:
                        return x
                    else:
                        i = (np.abs(centers - x)).argmin()
                        return centers[i]

                clustered_data = (
                    module.weight.data.cpu()
                    .apply_(replace_values_by_centers)
                    .to(device=device)
                )
                module.weight.data = clustered_data
                module.to(device=device)
        logger.info("Clustering error: %f", float(inertia))  # summed over all layers