import logging
from typing import Any, List

import torch
import torch.nn as nn

from .transforms.registry import registry

logger = logging.getLogger(__name__)


class BatchAugmentationPipeline(nn.Module):
    def __init__(self, replica=0, transforms={}):
        """Augmentation pipeline especially for self supervised learning

        Args:
            replica (int): number of replicated different random augmentations
            transforms (dict): configuration of transforms
        """
        super().__init__()

        self.replica = replica

        logger.info("Available transforms:")
        for transform in registry.transforms.keys():
            logger.info("  - %s", str(transform))

        transform_instances: List[nn.Module] = []
        for name, args in transforms.items():
            transform = registry.instantiate(name, **args)
            transform_instances.append(transform)

        self.transforms = nn.Sequential(*transform_instances)

        print(self.transforms)

    @torch.no_grad()
    def forward(self, x) -> List[Any]:
        """
        Perform Augmentations for Barlow Twins on a batch of input data

        Args:
            x (torch.Tensor): a torch.Tensor representing the augementation pipeline

        Returns:
            Tuple[torch.Tensor, torch.Tensor]; Batch augmented with `replica` different random augmentations
        """

        result: List[torch.Tensor] = []
        for _ in range(self.replica):
            result.append(self.transforms(x))

        return result
