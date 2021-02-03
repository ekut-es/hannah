import os
import random
import re
import json
import logging
import hashlib
import sys

from typing import Any, Dict, List, Optional

import pickle
import wfdb

import torchaudio
import numpy as np
import scipy.signal as signal
import torch
import torch.utils.data as data
import h5py

from enum import Enum
from collections import defaultdict
from chainmap import ChainMap

from ..utils import list_all_files, extract_from_download_cache
from .base import AbstractDataset, DatasetType

msglogger = logging.getLogger()

class Data3D:
    """ 3D-Data """

    last_x, last_y, last_z = 0.0, 0.0, 0.0

    def __init__(self, triple=None):
        if triple is None:
            self.x, self.y, self.z = 0.0, 0.0, 0.0
        elif None in triple:
            self.x = Data3D.last_x
            self.y = Data3D.last_y
            self.z = Data3D.last_z
        else:
            self.x = float(triple[0])
            self.y = float(triple[1])
            self.z = float(triple[2])

        Data3D.last_x = self.x
        Data3D.last_y = self.y
        Data3D.last_z = self.z

    def to_array(self):
        return np.array([self.x, self.y, self.z])


class PAMPAP2_IMUData:
    """ A IMU set defined by temperature (°C)
        3D-acceleration data (ms -2 ), scale: ±16g, resolution: 13-bit
        3D-acceleration data (ms -2 ), scale: ±6g, resolution: 13-bit
        3D-gyroscope data (rad/s)
        3D-magnetometer data (μT)
        orientation (invalid in this data collection) """

    last_temperature = 25.0

    def __init__(
        self,
        temperature=25.0,
        high_range_acceleration_data=Data3D(),
        low_range_acceleration_data=Data3D(),
        gyroscope_data=Data3D(),
        magnetometer_data=Data3D(),
    ):
        if temperature is None:
            self.temperature = PAMPAP2_IMUData.last_temperature
        else:
            self.temperature = float(temperature)

        self.high_range_acceleration_data = high_range_acceleration_data
        self.low_range_acceleration_data = low_range_acceleration_data
        self.gyroscope_data = gyroscope_data
        self.magnetometer_data = magnetometer_data

        PAMPAP2_IMUData.last_temperature = self.temperature

    def to_array(self):
        temperature_tensor = np.array([self.temperature])
        tensor_tuple = (
            temperature_tensor,
            self.high_range_acceleration_data.to_array(),
            self.low_range_acceleration_data.to_array(),
            self.gyroscope_data.to_array(),
            self.magnetometer_data.to_array(),
        )
        return np.concatenate(tensor_tuple)

    @staticmethod
    def from_elements(elements):
        return PAMPAP2_IMUData(
            temperature=elements[0],
            high_range_acceleration_data=Data3D(triple=elements[1:4]),
            low_range_acceleration_data=Data3D(triple=elements[4:7]),
            gyroscope_data=Data3D(triple=elements[7:10]),
            magnetometer_data=Data3D(triple=elements[10:13]),
        )


class PAMAP2_DataPoint:
    """ A temporal datapoint in the dataset"""

    ACTIVITY_MAPPING = {
        1: "lying",
        2: "sitting",
        3: "standing",
        4: "walking",
        5: "running",
        6: "cycling",
        7: "nordic_walking",
        9: "watching_tv",
        10: "computer_work",
        11: "car_driving",
        12: "ascending_stairs",
        13: "descending_stairs",
        16: "vacuum_cleaning",
        17: "ironing",
        18: "folding_laundry",
        19: "house_cleaning",
        20: "playing_soccer",
        24: "rope_jumping",
        # 0: "other",  # NOTE: This ("other") should be discarded from analysis
    }

    last_heart_rate = 0

    def __init__(
        self, timestamp, activityID, heart_rate, imu_hand, imu_chest, imu_ankle
    ):

        if timestamp is None:
            raise Exception("timestamp must not be NaN")
        self.timestamp = float(timestamp)

        if activityID is None:
            raise Exception("activityID must not be NaN")
        self.activityID = int(activityID)

        if heart_rate is None:
            self.heart_rate = PAMAP2_DataPoint.last_heart_rate
        else:
            self.heart_rate = float(heart_rate)
        PAMAP2_DataPoint.last_heart_rate = self.heart_rate

        self.imu_hand = imu_hand
        self.imu_chest = imu_chest
        self.imu_ankle = imu_ankle

    @staticmethod
    def from_elements(**kwargs):
        activityID = int(kwargs["activityID"])
        if activityID not in PAMAP2_DataPoint.ACTIVITY_MAPPING.keys():
            return None
        else:
            return PAMAP2_DataPoint(**kwargs)

    def to_array(self):
        heart_rate_tensor = np.array([self.heart_rate])
        tensor_tuple = (
            heart_rate_tensor,
            self.imu_hand.to_array(),
            self.imu_chest.to_array(),
            self.imu_ankle.to_array(),
        )

        return np.concatenate(tensor_tuple)

    def to_label(self):
        label = sorted(list(self.ACTIVITY_MAPPING.keys())).index(self.activityID)
        return label

    @staticmethod
    def from_line(line, split_character=" ", nan_string="NaN"):
        line = line.rstrip("\n\r")
        elements = line.split(split_character)

        elements = [None if element == nan_string else element for element in elements]

        return PAMAP2_DataPoint.from_elements(
            timestamp=elements[0],
            activityID=elements[1],
            heart_rate=elements[2],
            imu_hand=PAMPAP2_IMUData.from_elements(elements[3:19]),
            imu_chest=PAMPAP2_IMUData.from_elements(elements[20:36]),
            imu_ankle=PAMPAP2_IMUData.from_elements(elements[37:53]),
        )


class PAMAP2_DataChunk:
    """ A DataChunk is a item of the pytorch dataset """

    def __init__(self, source, start=None, stop=None):
        if type(source) is list:
            self.data = np.stack([datapoint.to_array() for datapoint in source])
            self.label = source[0].to_label()
        elif type(source) is str:
            with h5py.File(source, "r") as f:
                self.data = f["dataset"][start:stop]
                self.label = f["dataset"].attrs["label"]
        else:
            raise Exception("Unsupported DataChunk parameter")

    def to_file(self, file):
        with h5py.File(file, "w") as f:
            dataset = f.create_dataset("dataset", data=self.data)
            dataset.attrs["label"] = self.label

    def get_tensor(self):
        return torch.from_numpy(self.data).float()

    def get_label_tensor(self):
        return torch.Tensor([self.label]).long()

    def get_label(self):
        return self.label


class PAMAP2_Dataset(AbstractDataset):
    """ Class for the PAMAP2 activity dataset
        https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring"""

    def __init__(self, data_files, set_type, config):
        super().__init__()
        self.data_files = data_files
        self.channels = 40
        self.input_length = config["input_length"]
        self.label_names = [
            PAMAP2_DataPoint.ACTIVITY_MAPPING[index]
            for index in sorted(list(PAMAP2_DataPoint.ACTIVITY_MAPPING.keys()))
        ]

    def __getitem__(self, item):
        path, start = self.data_files[item]
        chunk = PAMAP2_DataChunk(path, start=start, stop=start + self.input_length)
        data = chunk.get_tensor().transpose(1, 0)
        label = chunk.get_label_tensor()
        return data, data.shape[0], label, label.shape[0]

    def __len__(self):
        return len(self.data_files)

    def prepare(cls, config: Dict[str, Any]) -> None:
        cls.download(config)


    @property
    def class_names(self) -> List[str]:
        return list(self.label_names)

    @property
    def class_counts(self) -> Optional[Dict[int, int]]:
        counts = defaultdict(int)
        for path, start in self.data_files:
            chunk = PAMAP2_DataChunk(path, start=start, stop=start + self.input_length)
            label = chunk.get_label()
            counts[label] += 1
        return counts


    @classmethod
    def splits(cls, config):

        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]

        input_length = config["input_length"]

        folder = os.path.join(config["data_folder"], "pamap2", "pamap2_prepared")

        sets = [[], [], []]

        for root, dirs, files in os.walk(folder):
            for file_name in files:
                path = os.path.join(root, file_name)
                with h5py.File(path, "r") as f:
                    length = len(f["dataset"][()])
                max_no_files = 2 ** 27 - 1
                start = 0
                stop = length
                step = input_length
                for i in range(start, stop, step):
                    if i + step >= stop - 1:
                        continue
                    chunk_hash = f"{path}{i}"
                    bucket = int(hashlib.sha1(chunk_hash.encode()).hexdigest(), 16)
                    bucket = (bucket % (max_no_files + 1)) * (100.0 / max_no_files)
                    if bucket < dev_pct:
                        tag = DatasetType.DEV
                    elif bucket < test_pct + dev_pct:
                        tag = DatasetType.TEST
                    else:
                        tag = DatasetType.TRAIN
                    sets[tag.value] += [(path, i)]

        datasets = (
            cls(sets[DatasetType.TRAIN.value], DatasetType.TRAIN, config),
            cls(sets[DatasetType.DEV.value], DatasetType.DEV, config),
            cls(sets[DatasetType.TEST.value], DatasetType.TEST, config),
        )
        return datasets

    @classmethod
    def download(cls, config):
        data_folder = config["data_folder"]
        clear_download = config["clear_download"]
        downloadfolder_tmp = config["download_folder"]

        if len(downloadfolder_tmp) == 0:
            downloadfolder_tmp = os.path.join(
                sys.argv[0].replace("speech_recognition/train.py", ""),
                "datasets/downloads",
            )

        if not os.path.isdir(downloadfolder_tmp):
            os.makedirs(downloadfolder_tmp)
            cached_files = list()
        else:
            cached_files = list_all_files(downloadfolder_tmp, ".zip")

        if not os.path.isdir(data_folder):
            os.makedirs(data_folder)

        variants = config["variants"]

        target_folder = os.path.join(data_folder, "pamap2")

        # download speech_commands dataset
        filename = "PAMAP2_Dataset.zip"

        if "pamap2" in variants:
            extract_from_download_cache(
                filename,
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip",
                cached_files,
                os.path.join(downloadfolder_tmp, "pamap2"),
                target_folder,
                clear_download=clear_download,
            )

        folder_prepared = os.path.join(target_folder, "pamap2_prepared")

        if not os.path.isdir(folder_prepared):
            cls.prepare_files(config, folder_prepared, target_folder)

    @classmethod
    def prepare_files(cls, config, folder_prepared, folder_source):
        os.makedirs(folder_prepared)

        folder_conf = ["Protocol", "Optional"]
        for conf in folder_conf:
            folder_samples = os.path.join(folder_source, "PAMAP2_Dataset", conf)
            for file in sorted(os.listdir(folder_samples)):
                path = os.path.join(folder_samples, file)
                datapoints = list()
                with open(path, "r") as f:
                    msglogger.info(f"Now processing {path}...")
                    for line in f:
                        datapoint = PAMAP2_DataPoint.from_line(line)
                        if datapoint is not None:
                            datapoints += [datapoint]
                msglogger.info("Now grouping...")
                groups = list()
                old_activityID = None
                for datapoint in datapoints:
                    if not datapoint.activityID == old_activityID:
                        groups += [[]]
                    groups[-1] += [datapoint]
                    old_activityID = datapoint.activityID
                msglogger.info("Now writing...")
                for nr, group in enumerate(groups):

                    subfolder = (
                        f"label_{str(group[0].to_label()).zfill(2)}"
                        f"_activityID_{str(group[0].activityID).zfill(2)}"
                        f"_{PAMAP2_DataPoint.ACTIVITY_MAPPING[group[0].activityID]}"
                    )

                    subfolder_path = os.path.join(folder_prepared, subfolder)
                    if not os.path.isdir(subfolder_path):
                        os.mkdir(subfolder_path)

                    data_chunk = PAMAP2_DataChunk(group)
                    data_chunk.to_file(
                        os.path.join(
                            folder_prepared, subfolder, f"{conf}_{file}_{nr}.hdf5"
                        )
                    )
<<<<<<< HEAD
=======

>>>>>>> 969a560... Add back pamap2 dataset
