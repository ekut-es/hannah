#
# Copyright (c) 2024 Hannah contributors.
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

# Generic dataset loading data from a number of pickle files

import logging
import pickle

import numpy as np
import torch

from .base import AbstractDataset

logger = logging.getLogger(__name__)


class PickleDataseLoader:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class PickleDataset(AbstractDataset):
    """A dataset loading data from a number of pickle files"""

    def __init__(self, files, num_classes=2):
        self.files = files
        self.num_classes = num_classes

        self.data = []

        self.load_data()

    def load_data(self):
        logger.info("Loading data from %d files", len(self.files))

        for name in self.files:
            with open(name, "rb") as f:
                self.data.extend(pickle.load(f))

        logger.info("Loaded %d samples", len(self.data))

    def loader(self, batch_size, shuffle=True):
        """Return the data loader for the dataset"""

        logger.info("Creating data loader with batch size %d", batch_size)

        # Building preloaded batches
        batches = []

        data = self.data[0]
        labels = self.data[1]

        if shuffle:
            rng = np.random.default_rng()
            permutation = rng.permutation(len(data))

            data = data[permutation]
            labels = labels[permutation]

        for i in range(0, len(self.data[0]), batch_size):
            data_batch = torch.tensor(self.data[0][i : i + batch_size])
            labels_batch = torch.tensor(self.data[1][i : i + batch_size])

            batches.append((data_batch, labels_batch))

        return PickleDataseLoader(batches)

    def prepare(config):
        """Prepare the dataset"""
        pass

    def splits(config):
        """Return the dataset splits"""

        return (
            PickleDataset(config["train"]),
            PickleDataset(config["val"]),
            PickleDataset(config["test"]),
        )

    @property
    def class_names(self):
        """Return the class names"""
        return [f"c{i}" for i in range(self.num_classes)]

    @property
    def class_counts(self):
        """Return the class counts"""

        counts = {i: 0 for i in range(self.num_classes)}

        for cls in self.data[1]:
            counts[cls] += 1

        return counts

    def __getitem__(self, index):
        """Return the item at the index"""

        logger.critical(
            "Returning single item from dataset, this might have negative impact on data loader performance"
        )

        return self.data[index]

    def __len__(self):
        """Return the length of the dataset"""

        return len(self.data[0])

    def size(self):
        return self.data[0].shape[1:]

    @property
    def max_workers(self):
        """Not really needed as the number of workers processes is defined by the loader() method"""
        return 1

    def __str__(self):
        return f"PickleDataset({self.files})"
