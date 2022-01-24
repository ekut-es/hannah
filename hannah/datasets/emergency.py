import os
import random
import re
import json
import logging
import hashlib
import os
import csv
import time
import torchaudio
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import torch

from collections import defaultdict

from .speech import _load_audio
from .base import AbstractDataset, DatasetType
from ..utils import list_all_files, extract_from_download_cache

from .NoiseDataset import NoiseDataset
from .DatasetSplit import DatasetSplit
from .Downsample import Downsample
from joblib import Memory

msglogger = logging.getLogger()

CACHE_DIR = os.getenv("HANNAH_CACHE_DIR", None)

if CACHE_DIR:
    CACHE_SIZE = os.getenv("HANNAH_CACHE_SIZE", None)
    cache = Memory(location=CACHE_DIR, bytes_limit=CACHE_SIZE, verbose=0)
    load_audio = cache.cache(_load_audio)
else:
    load_audio = _load_audio


class EmergencySirenDataset(AbstractDataset):
    """ Emergency Dataset """

    def __init__(self, data, set_type, config):
        super().__init__()
        self.audio_files = list(data.keys())
        self.set_type = set_type
        self.audio_labels = list(data.values())

        self.random_order = list(range(len(data.keys())))

        random.shuffle(self.random_order)

        self.samplingrate = config["samplingrate"]
        self.input_length = config["input_length"]

        self.channels = 1 # Use mono

    @classmethod
    def class_labels(cls):
        return ["ambient", "siren"]


    @property
    def class_names(self):
        return self.class_labels()

    @property
    def class_counts(self):
        return len(self.class_names())

    @classmethod
    def prepare(cls, config):
        #cls.prepare_data(config)
        pass

    def get_class(self, index):
        return [self.audio_labels[index]]

    def get_classes(self):
        labels = []
        for i in range(len(self)):
            labels.append(self.get_class(i))

        return labels

    def get_class_nums(self):
        classcounter = defaultdict(int)
        for n in self.get_classes():
            for c in n:
                classcounter[c] += 1

        return classcounter

    def __getitem__(self, index):

        index = self.random_order[index]

        label = torch.Tensor(self.get_class(index))
        label = label.long()

        path = self.audio_files[index]

        data = load_audio(path)

        data = torch.from_numpy(data)

        return data, data.shape[0], label, label.shape[0]

    def __len__(self):
        return len(self.audio_labels)

    @classmethod
    def splits(cls, config):

        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]

        folder = os.path.join(config["data_folder"], "siren_detection")

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