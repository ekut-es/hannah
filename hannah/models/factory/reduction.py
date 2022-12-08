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
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.container import Sequential


class ReductionBlockAdd(nn.Module):
    """Reduction block that sums over its inputs"""

    def __init__(self, *chains: Sequential) -> None:
        super().__init__()
        self.chains = nn.ModuleList(chains)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
          x: Tensor:
          x: Tensor:

        Returns:

        """
        chain_outputs = [f(x) for f in self.chains]
        result = chain_outputs[0]
        for y in chain_outputs[1:]:
            result += y
        result = self.act(result)

        return result


class ReductionBlockConcat(nn.Module):
    """Reduction block that concatenates its inputs"""

    def __init__(self, *chains: Sequential) -> None:
        super().__init__()
        self.chains = chains

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
          x: Tensor:
          x: Tensor:

        Returns:

        """
        chain_outputs = [f(x) for f in self.chains]
        return torch.cat(chain_outputs, 1)
