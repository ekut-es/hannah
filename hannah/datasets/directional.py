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
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
import yaml

from ..utils import extract_from_download_cache, list_all_files
from .base import AbstractDataset, DatasetType
from .speech import load_audio

random.seed(2022)


class DirectionalDataset(AbstractDataset):
    """Directional Dataset"""

    def __init__(self, data, set_type, config):
        super().__init__()
        self.audio_files = list(data.keys())
        self.set_type = set_type
        self.audio_labels = list(data.values())

        self.mics = config["mics"]

        self.channels = len(self.mics)
        self.input_length = config["input_length"]
        self.samplingrate = config["samplingrate"]

    @property
    def class_counts(self):
        return dict()

    @property
    def class_names(self):
        return list()

    def __len__(self) -> int:
        return len(self.audio_labels)

    @classmethod
    def prepare(cls, config):
        cls.download(config)

    @classmethod
    def download(cls, config):
        data_folder = config["data_folder"]
        clear_download = config["clear_download"]
        downloadfolder_tmp = config["download_folder"]
        target_folder = os.path.join(data_folder, "directional")

        if os.path.isdir(target_folder):
            return

        if not os.path.isdir(downloadfolder_tmp):
            os.makedirs(downloadfolder_tmp)
            cached_files = list()
        else:
            cached_files = list_all_files(downloadfolder_tmp, ".tar.gz")

        extract_from_download_cache(
            "directional.tar.gz",
            "https://es-cloud.cs.uni-tuebingen.de/f/0bb98014608541ef9fee/?dl=1",
            cached_files,
            os.path.join(downloadfolder_tmp, "directional"),
            target_folder,
            clear_download=clear_download,
        )

    def get_data_with_label(self, index):
        label = self.audio_labels[index]

        folder = self.audio_files[index]

        channels = list()

        for channel_name in self.mics:
            path = os.path.join(folder, channel_name + ".wav")
            samples = np.squeeze(load_audio(path, sr=self.samplingrate))
            channels += [samples]

        data = np.stack(channels)

        return data, label

    def __getitem__(self, index):

        data, label = self.get_data_with_label(index)

        label = torch.Tensor(label)

        data = torch.from_numpy(data)

        return data, data.shape[0], label, label.shape[0]

    @classmethod
    def splits(cls, config):

        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]

        folder = os.path.join(config["data_folder"], "directional", "directional")

        train_files = dict()
        test_files = dict()
        dev_files = dict()

        train_sirens = set()
        test_sirens = set()
        dev_sirens = set()

        num_samples = len(os.listdir(folder))
        for i, subfolder in enumerate(sorted(os.listdir(folder))):
            subpath = os.path.join(folder, subfolder)
            siren_id = subfolder[:4]
            with open(os.path.join(subpath, "setting.yaml"), "r") as f:
                conf = yaml.safe_load(f)

            x = conf["siren"]["x"]
            y = conf["siren"]["y"]

            label = x, y

            if i < num_samples * dev_pct and (
                siren_id not in test_sirens.union(train_sirens)
            ):
                dev_files[subpath] = label
                dev_sirens.add(siren_id)
            elif i < num_samples * (dev_pct + test_pct) and (
                siren_id not in dev_sirens.union(train_sirens)
            ):
                test_files[subpath] = label
                test_sirens.add(siren_id)
            else:
                train_files[subpath] = label
                train_sirens.add(siren_id)

        datasets = (
            cls(train_files, DatasetType.TRAIN, config),
            cls(dev_files, DatasetType.DEV, config),
            cls(test_files, DatasetType.TEST, config),
        )

        return datasets


class CompassDataset(DirectionalDataset):
    @property
    def class_names(self):
        return self.class_labels()

    @classmethod
    def class_labels(cls):
        return [
            "n",
            "ne",
            "e",
            "se",
            "s",
            "sw",
            "w",
            "nw",
        ]

    def class_counts(self):
        classcounter = defaultdict(int)
        for x, y in self.audio_labels:
            c = self.cartesian_to_compass(x, y)
            classcounter[c] += 1
        return classcounter

    def cartesian_to_compass(self, x, y):
        angle = math.atan2(x, y)

        sin = math.sin(angle)
        cos = math.cos(angle)

        angles = [
            i * 2 * math.pi / len(self.class_labels())
            for i in range(len(self.class_labels()))
        ]

        sins_coss = [(math.sin(angle), math.cos(angle)) for angle in angles]

        dists = [
            (label, math.dist((sin, cos), sin_cos))
            for label, sin_cos in zip(self.class_labels(), sins_coss)
        ]

        label, _ = min(dists, key=lambda x: x[1])

        return label

    def __getitem__(self, index):

        data, label = super().get_data_with_label(index)

        x, y = label

        label = self.cartesian_to_compass(x, y)

        label = self.class_labels().index(label)

        label = torch.Tensor([label]).long()

        data = torch.from_numpy(data)

        return data, data.shape[0], label, label.shape[0]


class BeamformingDataset(DirectionalDataset):
    # Beamforming Dataset

    @classmethod
    def get_speed_of_sound(cls):
        return 340.0

    def __init__(self, data, set_type, config):
        super().__init__(data, set_type, config)

        self.channels = 3

    @classmethod
    def splits(cls, config):

        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]

        folder = os.path.join(config["data_folder"], "beamforming", "beamforming.multi")

        train_files = dict()
        test_files = dict()
        dev_files = dict()

        train_sirens = set()
        test_sirens = set()
        dev_sirens = set()

        num_samples = len(os.listdir(folder))
        for i, subfolder in enumerate(sorted(os.listdir(folder))):
            subpath = os.path.join(folder, subfolder)
            siren_id = subfolder[:4]
            with open(os.path.join(subpath, "setting.yaml"), "r") as f:
                conf = yaml.safe_load(f)

            siren = conf["siren"]

            mics_key = "microphones"

            mics = conf[mics_key]

            for variant in range(config["num_variants"]):

                idx = random.sample(list(range(len(mics))), 2)

                label = siren, mics[idx[0]], idx[0], mics[idx[1]], idx[1]

                conf_tuple = (subpath, variant)

                if i < num_samples * dev_pct and (
                    siren_id not in test_sirens.union(train_sirens)
                ):
                    dev_files[conf_tuple] = label
                    dev_sirens.add(siren_id)
                elif i < num_samples * (dev_pct + test_pct) and (
                    siren_id not in dev_sirens.union(train_sirens)
                ):
                    test_files[conf_tuple] = label
                    test_sirens.add(siren_id)
                else:
                    train_files[conf_tuple] = label
                    train_sirens.add(siren_id)

        datasets = (
            cls(train_files, DatasetType.TRAIN, config),
            cls(dev_files, DatasetType.DEV, config),
            cls(test_files, DatasetType.TEST, config),
        )

        return datasets

    @classmethod
    def class_labels(cls):
        return [
            "+",
            "-",
        ]

    def get_bin(self, siren, mic1, mic2):
        diff = math.dist((mic1["x"], mic1["y"]), (siren["x"], siren["y"])) - math.dist(
            (mic2["x"], mic2["y"]), (siren["x"], siren["y"])
        )

        return "+" if diff > 0 else "-"

    @property
    def class_counts(self):
        classcounter = defaultdict(int)
        for x, y in self.audio_labels:
            label = self.get_bin(x, y)
            classcounter[label] += 1
        return classcounter

    @property
    def class_names(self):
        return self.class_labels()

    def __getitem__(self, index):

        siren, mic1, mic1_id, mic2, mic2_id = self.audio_labels[index]

        label = self.get_bin(siren, mic1, mic2)

        label = self.class_labels().index(label)

        label = torch.Tensor([label]).long()

        folder, _ = self.audio_files[index]

        channels = list()

        for channel_name in [str(mic1_id), str(mic2_id)]:
            path = os.path.join(folder, channel_name + ".wav")
            audio = load_audio(path, sr=self.samplingrate)
            samples = np.squeeze(audio)
            samples = samples[: self.input_length]
            channels += [samples]

        num_bins = len(self.class_labels())

        bin_length = self.input_length // num_bins

        new_channel = channels[1].copy()

        new_channel = new_channel[num_bins:]

        new_channel = np.concatenate(
            (new_channel, np.zeros(bin_length - len(new_channel) % bin_length))
        )

        new_channel_split = np.split(new_channel, num_bins)

        new_channel = np.vstack(new_channel_split)

        new_channel = np.concatenate((new_channel, np.zeros((num_bins, 1))), axis=1)

        new_channel = new_channel.flatten()

        new_channel = new_channel[: self.input_length]

        channels += [new_channel * channels[0]]  # Lock-In Amp like idea

        data = np.stack(channels)

        data = torch.from_numpy(data).float()

        return data, data.shape[0], label, label.shape[0]
