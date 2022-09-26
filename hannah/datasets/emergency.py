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
import os
import random
from collections import defaultdict

import torch

from ..utils import extract_from_download_cache, list_all_files
from .base import AbstractDataset, DatasetType
from .speech import load_audio


class EmergencySirenDataset(AbstractDataset):
    """Emergency Dataset"""

    def __init__(self, data, set_type, config):
        super().__init__()
        self.audio_files = list(data.keys())
        self.set_type = set_type
        self.audio_labels = list(data.values())

        self.random_order = list(range(len(data.keys())))

        random.shuffle(self.random_order)

        self.samplingrate = config["samplingrate"]
        self.input_length = config["input_length"]

        self.channels = 1  # Use mono

    @classmethod
    def class_labels(cls):
        return ["ambient", "siren"]

    @property
    def class_names(self):
        return self.class_labels()

    @property
    def class_counts(self):
        counts = defaultdict(int)
        for label in self.audio_labels:
            counts[label] += 1
        return counts

    @classmethod
    def prepare(cls, config):
        cls.download(config)

    @classmethod
    def download(cls, config):
        data_folder = config["data_folder"]
        clear_download = config["clear_download"]
        downloadfolder_tmp = config["download_folder"]
        target_folder = os.path.join(data_folder, "siren_detection")

        if os.path.isdir(target_folder):
            return

        if not os.path.isdir(downloadfolder_tmp):
            os.makedirs(downloadfolder_tmp)
            cached_files = list()
        else:
            cached_files = list_all_files(downloadfolder_tmp, ".tar.gz")

        extract_from_download_cache(
            "siren_detection.tar.gz",
            "https://atreus.informatik.uni-tuebingen.de/seafile/f/239614bc48604cb1bf1c/?dl=1",
            cached_files,
            os.path.join(downloadfolder_tmp, "siren_detection"),
            target_folder,
            clear_download=clear_download,
        )

    def __getitem__(self, index):

        index = self.random_order[index]

        label = torch.Tensor([self.audio_labels[index]])
        label = label.long()

        path = self.audio_files[index]

        data = load_audio(path, sr=self.samplingrate)

        data = torch.from_numpy(data)
        data = data[:, : self.input_length]

        return data, data.shape[0], label, label.shape[0]

    def __len__(self):
        return len(self.audio_labels)

    @classmethod
    def splits(cls, config):

        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]

        folder = os.path.join(
            config["data_folder"], "siren_detection", "siren_detection"
        )

        train_files = dict()
        test_files = dict()
        dev_files = dict()

        for subfolder in os.listdir(folder):
            subpath = os.path.join(folder, subfolder)
            num_files = len(os.listdir(subpath))
            for i, filename in enumerate(os.listdir(subpath)):
                path = os.path.join(folder, subfolder, filename)
                if i < num_files * dev_pct:
                    dev_files[path] = cls.class_labels().index(subfolder)
                elif i < num_files * (dev_pct + test_pct):
                    test_files[path] = cls.class_labels().index(subfolder)
                else:
                    train_files[path] = cls.class_labels().index(subfolder)

        datasets = (
            cls(train_files, DatasetType.TRAIN, config),
            cls(dev_files, DatasetType.DEV, config),
            cls(test_files, DatasetType.TEST, config),
        )
        return datasets
