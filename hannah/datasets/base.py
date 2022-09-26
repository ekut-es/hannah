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
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DatasetType(Enum):
    """The type of a dataset partition e.g. train, dev, test"""

    TRAIN = 0
    DEV = 1
    TEST = 2


class AbstractDataset(Dataset, ABC):
    @abstractclassmethod
    def prepare(cls, config: Dict[str, Any]) -> None:
        """Prepare the dataset.

        This method is run at the beginning of the dataset training.

        If possible it should download the dataset from its original source, if
        it is available for public download.


        Args:
            config (Dict[Any]): The dataset configuration
        """

        pass

    @abstractclassmethod
    def splits(
        cls, config: Dict[str, Any]
    ) -> Tuple["AbstractDataset", "AbstractDataset", "AbstractDataset"]:
        """Returns the test, validation and train split according to the Dataset config

        Args:
            config ([type]): [description]
        """

        pass  # pytype: disable=bad-return-type

    @abstractproperty
    def class_names(self) -> List[str]:
        """Returns the names of the classes in the classification dataset"""
        pass  # pytype: disable=bad-return-type

    @property
    def class_names_abbreviated(self) -> List[str]:
        max_len = 0
        for name in self.class_names:
            max_len = max(max_len, len(name))
        if max_len > 6:
            logger.warning(
                "Datasets class names contain classes that are longer than the recommended lenght of 5 characters, consider implementing class_names_abbreviated in your dataset"
            )

        return self.class_names

    @abstractproperty
    def class_counts(self) -> Optional[Dict[int, int]]:
        """Returns the number of items in each class of the dataset

        If this is not applicable to a dataset type e.g. ASR, Semantic Segementation,
        it may return None
        """
        pass

    @abstractmethod
    def __getitem__(self, index) -> List[torch.Tensor]:
        """Returns a torch.Tensor for the data item at the corresponding index

        The length of the list depends on the dataset item to use

        Args:
            index (int): the index of the data item
        """

        pass  # pytype: disable=bad-return-type

    @abstractmethod
    def __len__(self) -> int:
        """Returns number of samples in dataset"""

        pass  # pytype: disable=bad-return-type

    def size(self) -> List[int]:
        """Returns dimension of output output without batch dimension"""

        return [self.channels, self.input_length]


def ctc_collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Sequences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        data: list of tuple (src_seq, src_length, trg_seq, trg_length).
            - src_seq: torch tensor of shape (x,?); variable length.
            - src length: torch tenso of shape 1x1
            - trg_seq: torch tensor of shape (?); variable length.
            - trg_length: torch_tensor of shape (1x1)
    Returns: tuple of four torch tensors
        src_seqs: torch tensor of shape (batch_size, x, padded_length).
        src_lengths: torch_tensor of shape (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, x, padded_length).
        trg_lengths: torch tensor of shape (batch_size); valid length for each padded target sequence.
    """

    def merge(sequences):
        lengths = [seq.shape[-1] for seq in sequences]
        max_length = max(lengths)

        padded_seqs = []

        for item in sequences:
            padded = torch.nn.functional.pad(
                input=item,
                pad=(0, max_length - item.shape[-1]),
                mode="constant",
                value=0,
            )
            padded_seqs.append(padded)

        return padded_seqs, lengths

    # seperate source and target sequences
    src_seqs, src_lengths, trg_seqs, trg_lengths = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)

    return (
        torch.stack(src_seqs),
        torch.Tensor(src_lengths),
        torch.stack(trg_seqs),
        torch.Tensor(trg_lengths),
    )
