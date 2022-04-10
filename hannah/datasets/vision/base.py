import logging

import numpy as np

from ..base import AbstractDataset

logger = logging.getLogger(__name__)


class VisionDatasetBase(AbstractDataset):
    """Wrapper around torchvision classification datasets"""

    def __init__(self, config, dataset, transform=None):
        self.config = config
        self.dataset = dataset
        self.transform = transform

    @property
    def class_counts(self):
        return None

    def __getitem__(self, index):
        data, target = self.dataset[index]
        data = np.array(data)
        if self.transform:
            data = self.transform(image=data)["image"]
        return data, target

    def size(self):
        dim = self[0][0].size()

        return list(dim)

    def __len__(self):
        return len(self.dataset)
