import os
import hashlib
import sys
import csv
import pickle
import wfdb

import torchaudio
import numpy as np
import scipy.signal as signal
import torch
import torch.utils.data as data

from enum import Enum
from collections import defaultdict

from .base import AbstractDataset, DatasetType
from ..utils import list_all_files, extract_from_download_cache


class PhysioDataset(AbstractDataset):
    def __init__(self, data, set_type, config, dataset_name=None):
        super().__init__()
        self.physio_files = list(data.keys())
        self.set_type = set_type
        self.physio_labels = list(data.values())

        self.samplingrate = config["samplingrate"]

        self.input_length = config["input_length"]

        self.channels = config["num_channels"]

        self.dataset_name = dataset_name

    @property
    def class_names(self):
        return self.label_names.values()

    @property
    def class_counts(self):
        return self.get_categories_distribution()

    def __getitem__(self, index):
        if self.dataset_name == "PhysioCinc":
            label = torch.Tensor([self.physio_labels[index]]).long()
            with open(self.physio_files[index], "rb") as f:
                data = pickle.load(f)
            data = torch.from_numpy(data).float()
        elif self.dataset_name == "AtrialFibrillation":
            with open(self.physio_files[index], "rb") as f:
                data = pickle.load(f)
            data = torch.from_numpy(data)
            data = data.transpose(1, 0).float()
            if self.channels == 1:
                data = data[0]
        return data, data.shape[0], label, label.shape[0]

    def get_label_list(self):
        return self.physio_labels

    def get_categories_distribution(self):
        distribution = defaultdict(int)
        for label in self.get_label_list():
            distribution[label] += 1
        return distribution

    def __len__(self):
        return len(self.physio_files)


class PhysioCincDataset(PhysioDataset):

    LABEL_NORMAL = "N"
    LABEL_ATRIAL_FIBRILLATION = "A"
    LABEL_OTHER_RYTHM = "O"
    LABEL_NOISY = "~"

    def __init__(self, data, set_type, config):
        super().__init__(data, set_type, config, "PhysioCinc")
        self.samplingrate = config["samplingrate"]

        self.input_length = config["input_length"]

        self.channels = config["num_channels"]

        self.label_names = {
            0: self.LABEL_NORMAL,
            1: self.LABEL_ATRIAL_FIBRILLATION,
            2: self.LABEL_OTHER_RYTHM,
            3: self.LABEL_NOISY,
        }

    @classmethod
    def get_label_names(cls):
        return {
            0: cls.LABEL_NORMAL,
            1: cls.LABEL_ATRIAL_FIBRILLATION,
            2: cls.LABEL_OTHER_RYTHM,
            3: cls.LABEL_NOISY,
        }

    @classmethod
    def get_label_mapping(cls):
        return {
            cls.LABEL_NORMAL: 0,
            cls.LABEL_ATRIAL_FIBRILLATION: 1,
            cls.LABEL_OTHER_RYTHM: 2,
            cls.LABEL_NOISY: 3,
        }

    @classmethod
    def prepare(cls, config):
        cls.download(config)
        cls.prepare_files(config)

    @classmethod
    def download(cls, config):
        data_folder = config["data_folder"]
        clear_download = config["clear_download"]
        downloadfolder_tmp = config["download_folder"]

        target_folder = os.path.join(data_folder, "cinc_2017")
        if os.path.isdir(target_folder):
            return

        if len(downloadfolder_tmp) == 0:
            download_folder = os.path.join(data_folder, "downloads")

        if not os.path.isdir(downloadfolder_tmp):
            os.makedirs(downloadfolder_tmp)
            cached_files = list()
        else:
            cached_files = list_all_files(downloadfolder_tmp, ".zip")

        if not os.path.isdir(data_folder):
            os.makedirs(data_folder)

        variants = config["variants"]

        # download Physionet 2017 CinC dataset
        filename = "training2017.zip"

        if "cinc_2017" in variants:
            extract_from_download_cache(
                filename,
                "https://physionet.org/files/challenge-2017/1.0.0/training2017.zip",
                cached_files,
                os.path.join(downloadfolder_tmp, "cinc_2017"),
                target_folder,
                clear_download=clear_download,
            )

    @classmethod
    def prepare_files(cls, config):
        print("Preparing files...")
        files_list = list()
        data_folder = config["data_folder"]
        raw_folder = os.path.join(data_folder, "cinc_2017", "training2017")
        output_folder = os.path.join(data_folder, "cinc_2017_prepared")
        if os.path.isdir(output_folder):
            print("Preparation folder already exists, skipping...")
            return
        os.makedirs(output_folder)
        for label in cls.get_label_mapping().keys():
            os.makedirs(os.path.join(output_folder, label))

        for filename in os.listdir(raw_folder):
            file_path = os.path.join(raw_folder, filename)
            if os.path.isfile(file_path) and ".mat" in filename:
                name, extension = filename.split(".")
                files_list += [name]

        # Load Labels
        labels = {}
        label_file_path = os.path.join(raw_folder, "REFERENCE.csv")
        with open(label_file_path, "r") as data:
            for line in csv.reader(data):
                labels.update({(line[0]): (line[1])})

        raw_data = list()

        sample_length = config["input_length"]
        for name in files_list:
            sample_path = os.path.join(raw_folder, name)
            samples, _ = wfdb.rdsamp(sample_path)
            zero_pad_len = sample_length
            zero_pad = np.zeros(zero_pad_len)
            samples = np.append(samples, zero_pad)
            samples = samples[0:zero_pad_len]
            ##Sample length is 18000 = 60s ECG recording
            # raw_data += [{"name": name, "annotations": annotations, "samples": samples}]
            raw_data += [{"name": name, "sample": samples, "label": labels[name]}]

        total_samples = len(raw_data)
        for experiment_number, element in enumerate(raw_data):
            name = element["name"]
            sample = element["sample"]
            label = element["label"]
            out_path = os.path.join(output_folder, label)
            if not os.path.isdir(out_path):
                os.makedirs(out_path)
            path = os.path.join(output_folder, label, name)
            with open(path, "wb") as f:
                pickle.dump(sample, f)

    @classmethod
    def splits(cls, config):

        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]

        folder = os.path.join(config["data_folder"], "cinc_2017_prepared")

        train_files = dict()
        test_files = dict()
        dev_files = dict()

        for subfolder in os.listdir(folder):
            subpath = os.path.join(folder, subfolder)
            for filename in os.listdir(subpath):
                path = os.path.join(folder, subfolder, filename)
                max_no_files = 2 ** 27 - 1
                bucket = int(hashlib.sha1(path.encode()).hexdigest(), 16)
                bucket = (bucket % (max_no_files + 1)) * (100.0 / max_no_files)
                if bucket < dev_pct:
                    dev_files[path] = cls.get_label_mapping()[subfolder]
                elif bucket < test_pct + dev_pct:
                    test_files[path] = cls.get_label_mapping()[subfolder]
                else:
                    train_files[path] = cls.get_label_mapping()[subfolder]

        datasets = (
            cls(train_files, DatasetType.TRAIN, config),
            cls(dev_files, DatasetType.DEV, config),
            cls(test_files, DatasetType.TEST, config),
        )
        return datasets


class AtrialFibrillationDataset(PhysioDataset):
    """ Atrial Fibrillation Database (https://physionet.org/content/afdb/1.0.0/)"""

    LABEL_ATRIAL_FIBRILLATION = "atrial_fibrillation"
    LABEL_ATRIAL_FLUTTER = "atrial_flutter"
    LABEL_JUNCTIONAL_RYTHM = "junctional_rythm"
    LABEL_OTHER_RYTHM = "other"

    ANN_ATRIAL_FIBRILLATION = "(AFIB"
    ANN_ATRIAL_FLUTTER = "(AFL"
    ANN_JUNCTIONAL_RYTHM = "(J"
    ANN_OTHER_RYTHM = "(N"

    def __init__(self, data, set_type, config):
        self.label_names = self.get_label_names()

        self.annotation_names = self.get_annotation_names()
        super().__init__(data, set_type, config, "AtrialFibrillation")

        self.channels = config["num_channels"]

    @classmethod
    def get_annotation_names(cls):
        return {
            cls.ANN_ATRIAL_FIBRILLATION: 0,
            cls.ANN_ATRIAL_FLUTTER: 1,
            cls.ANN_JUNCTIONAL_RYTHM: 2,
            cls.ANN_OTHER_RYTHM: 3,
        }

    @classmethod
    def get_label_mapping(cls):
        return {
            cls.LABEL_ATRIAL_FIBRILLATION: 0,
            cls.LABEL_ATRIAL_FLUTTER: 1,
            cls.LABEL_JUNCTIONAL_RYTHM: 2,
            cls.LABEL_OTHER_RYTHM: 3,
        }

    @classmethod
    def get_label_names(cls):
        return {
            0: cls.LABEL_ATRIAL_FIBRILLATION,
            1: cls.LABEL_ATRIAL_FLUTTER,
            2: cls.LABEL_JUNCTIONAL_RYTHM,
            3: cls.LABEL_OTHER_RYTHM,
        }

    @classmethod
    def get_physiological_pattern(cls):
        return cls.get_annotation_names()[cls.ANN_OTHER_RYTHM]

    @classmethod
    def prepare(cls, config):
        cls.download(config)
        cls.prepare_files(config)

    @classmethod
    def download(cls, config):
        data_folder = config["data_folder"]
        clear_download = config["clear_download"]
        downloadfolder_tmp = config["download_folder"]

        target_folder = os.path.join(data_folder, "atrial_fibrillation")
        if os.path.isdir(target_folder):
            return

        if len(downloadfolder_tmp) == 0:
            download_folder = os.path.join(data_folder, "downloads")

        if not os.path.isdir(downloadfolder_tmp):
            os.makedirs(downloadfolder_tmp)
            cached_files = list()
        else:
            cached_files = list_all_files(downloadfolder_tmp, ".zip")

        if not os.path.isdir(data_folder):
            os.makedirs(data_folder)

        variants = config["variants"]

        # download atrial fibrillation dataset
        filename = "mit-bih-atrial-fibrillation-database-1.0.0.zip"

        if "atrial_fibrillation" in variants:
            extract_from_download_cache(
                filename,
                "https://physionet.org/static/published-projects/afdb/mit-bih-atrial-fibrillation-database-1.0.0.zip",
                cached_files,
                os.path.join(downloadfolder_tmp, "atrial_fibrillation"),
                target_folder,
                clear_download=clear_download,
            )

    @classmethod
    def prepare_files(cls, config):
        print("Preparing files...")
        files_list = list()
        data_folder = config["data_folder"]
        raw_folder = os.path.join(data_folder, "atrial_fibrillation", "files")
        output_folder = os.path.join(data_folder, "atrial_fibrillation_prepared")
        if os.path.isdir(output_folder):
            print("Preparation folder already exists, skipping...")
            return
        os.makedirs(output_folder)
        for label in cls.get_label_mapping().keys():
            os.makedirs(os.path.join(output_folder, label))

        for filename in os.listdir(raw_folder):
            file_path = os.path.join(raw_folder, filename)
            if os.path.isfile(file_path) and ".dat" in filename:
                name, extension = filename.split(".")
                annotation_path = os.path.join(raw_folder, name + ".atr")
                if os.path.isfile(annotation_path):
                    files_list += [name]

        raw_data = list()

        for name in files_list:
            sample_path = os.path.join(raw_folder, name)
            annotations = wfdb.rdann(
                sample_path,
                extension="atr",
                return_label_elements=["symbol", "label_store", "description"],
            )
            samples, _ = wfdb.rdsamp(sample_path)
            raw_data += [{"name": name, "annotations": annotations, "samples": samples}]

        sample_length = config["input_length"]

        for experiment_nr, element in enumerate(raw_data):
            samples = element["samples"]
            annotations = element["annotations"]
            for i, (typus, start_sample) in enumerate(
                zip(annotations.aux_note, annotations.sample)
            ):

                label_number = cls.get_annotation_names()[typus]
                if i < len(annotations.sample) - 1:
                    stop_sample = annotations.sample[i + 1]
                else:
                    stop_sample = len(samples) - 1
                start = start_sample
                stop = stop_sample
                step = int(sample_length * (1 - config["overlap_ratio"]))
                for i in range(start, stop, step):
                    if start_sample + sample_length < stop_sample:
                        chunk = samples[start_sample : start_sample + sample_length]
                    else:
                        chunk = samples[
                            stop_sample - sample_length - 1 : stop_sample - 1
                        ]
                    path = os.path.join(
                        output_folder,
                        cls.get_label_names()[label_number],
                        f"ex{experiment_nr}_{i}",
                    )

                    with open(path, "wb") as f:
                        pickle.dump(chunk, f)

    @classmethod
    def splits(cls, config):

        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]

        folder = os.path.join(config["data_folder"], "atrial_fibrillation_prepared")

        train_files = dict()
        test_files = dict()
        dev_files = dict()

        for subfolder in os.listdir(folder):
            subpath = os.path.join(folder, subfolder)
            for filename in os.listdir(subpath):
                path = os.path.join(folder, subfolder, filename)
                max_no_files = 2 ** 27 - 1
                bucket = int(hashlib.sha1(path.encode()).hexdigest(), 16)
                bucket = (bucket % (max_no_files + 1)) * (100.0 / max_no_files)
                if bucket < dev_pct:
                    dev_files[path] = cls.get_label_mapping()[subfolder]
                elif bucket < test_pct + dev_pct:
                    test_files[path] = cls.get_label_mapping()[subfolder]
                else:
                    train_files[path] = cls.get_label_mapping()[subfolder]

        datasets = (
            cls(train_files, DatasetType.TRAIN, config),
            cls(dev_files, DatasetType.DEV, config),
            cls(test_files, DatasetType.TEST, config),
        )
        return datasets
