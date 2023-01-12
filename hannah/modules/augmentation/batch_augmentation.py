#
# Copyright (c) 2023 Hannah contributors.
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
from typing import Any, List

import torch
import torch.nn as nn

from .transforms.registry import registry

logger = logging.getLogger(__name__)


class BatchAugmentationPipeline(nn.Module):
    def __init__(self, transforms={}):
        """Augmentation pipeline especially for self supervised learning

        Args:
            replica (int): number of replicated different random augmentations
            transforms (dict): configuration of transforms
        """
        super().__init__()
        # logger.info("Available transforms:")
        # for transform in registry.transforms.keys():
        #    logger.info("  - %s", str(transform))

        transform_instances: List[nn.Module] = []
        for name, args in transforms.items():
            transform = registry.instantiate(name, **args)
            transform_instances.append(transform)
        self.transforms = nn.Sequential(*transform_instances)
        # print(self.transforms)

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        """
        Perform Augmentations

        Args:
            x (torch.Tensor): a torch.Tensor representing the augementation pipeline

        Returns:
            Tuple[torch.Tensor, torch.Tensor]; Batch augmented with `replica` different random augmentations
        """

        result = self.transforms(x)

        return result
