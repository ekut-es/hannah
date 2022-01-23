from logging import config
import collections
import os
import logging
import pathlib
from posixpath import split
import tarfile
import requests

from hydra.utils import get_original_cwd
import numpy as np
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .base import AbstractDataset

from .utils import csv_dataset


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


class Cifar10Dataset(VisionDatasetBase):
    @classmethod
    def prepare(cls, config):
        data_folder = config.data_folder
        root_folder = os.path.join(data_folder, "CIFAR10")
        _ = datasets.CIFAR10(root_folder, train=False, download=True)

    @property
    def class_names(self):
        q = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        return q

    @classmethod
    def splits(cls, config):
        data_folder = config.data_folder
        root_folder = os.path.join(data_folder, "CIFAR10")

        # print(loaded_transform)
        train_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=32),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomCrop(height=32, width=32),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
                ToTensorV2(),
            ]
        )

        val_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=32),
                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
                ToTensorV2(),
            ]
        )

        test_set = datasets.CIFAR10(root_folder, train=False, download=False)
        train_val_set = datasets.CIFAR10(root_folder, train=True, download=False)
        train_val_len = len(train_val_set)

        split_sizes = [
            int(train_val_len * (1.0 - config.val_percent)),
            int(train_val_len * config.val_percent),
        ]
        train_set, val_set = data.random_split(train_val_set, split_sizes)

        return (
            cls(config, train_set, train_transform),
            cls(config, val_set, val_transform),
            cls(config, test_set, val_transform),
        )


class FakeDataset(VisionDatasetBase):
    @classmethod
    def prepare(cls, config):
        pass

    @classmethod
    def splits(cls, config):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        test_data = torchvision.datasets.FakeData(
            size=128,
            image_size=(3, 32, 32),
            num_classes=config.num_classes,
            transform=transform,
        )
        val_data = torchvision.datasets.FakeData(
            size=128,
            image_size=(3, 32, 32),
            num_classes=config.num_classes,
            transform=transform,
        )
        train_data = torchvision.datasets.FakeData(
            size=512,
            image_size=(3, 32, 32),
            num_classes=config.num_classes,
            transform=transform,
        )

        return cls(config, train_data), cls(config, val_data), cls(config, test_data)

    @property
    def class_names(self):
        return [f"class{n}" for n in range(self.config.num_classes)]


class KvasirCapsuleDataset(VisionDatasetBase):

    # Kvasir_Capsule
    DOWNLOAD_URL = "https://files.osf.io/v1/resources/dv2ag/providers/googledrive/labelled_images/?zip="

    @classmethod
    def prepare(cls, config):
        download_folder = os.path.join(config.data_folder, "download")
        extract_root = os.path.join(
            config.data_folder, "kvasir_capsule", "labelled_images"
        )

        # download and extract dataset
        if not os.path.isdir(extract_root):
            datasets.utils.download_and_extract_archive(
                cls.DOWNLOAD_URL,
                download_folder,
                extract_root=extract_root,
                filename="labelled_images.zip",
            )

            for tar_file in pathlib.Path(extract_root).glob("*.tar.gz"):
                # ampulla , hematin and polyp removed from official_splits due to small findings
                if str(tar_file) in [
                    extract_root + "/ampulla_of_vater.tar.gz",
                    extract_root + "/blood_hematin.tar.gz",
                    extract_root + "/polyp.tar.gz",
                ]:
                    tar_file.unlink()
                    continue
                logger.info("Extracting: %s", str(tar_file))
                with tarfile.open(tar_file) as archive:
                    archive.extractall(path=extract_root)
                tar_file.unlink()
            # rename dataset image labels to match official splits csv
            class_names = {
                "Angiectasia": "Angiectasia",
                "Erosion": "Erosion",
                "Pylorus": "Pylorus",
                "Blood - fresh": "Blood",
                "Erythema": "Erythematous",
                "Foreign body": "Foreign Bodies",
                "Ileocecal valve": "Ileo-cecal valve",
                "Lymphangiectasia": "Lymphangiectasia",
                "Normal clean mucosa": "Normal",
                "Pylorus": "Pylorus",
                "Reduced mucosal view": "Reduced Mucosal View",
                "Ulcer": "Ulcer",
            }
            for c_name in class_names:
                new_file = os.path.join(extract_root, class_names[c_name])
                old_file = os.path.join(extract_root, c_name)
                os.rename(old_file, new_file)

        # downlaod splits
        official_splits_path = os.path.join(
            config.data_folder, "kvasir_capsule", "official_splits"
        )
        if not os.path.isdir(official_splits_path):
            os.mkdir(official_splits_path)
            s0 = requests.get(
                "https://raw.githubusercontent.com/simula/kvasir-capsule/master/official_splits/split_0.csv"
            )
            with open(os.path.join(official_splits_path, "split_0.csv"), "wb") as f:
                f.write(s0.content)
            s1 = requests.get(
                "https://raw.githubusercontent.com/simula/kvasir-capsule/master/official_splits/split_1.csv"
            )
            with open(os.path.join(official_splits_path, "split_1.csv"), "wb") as f:
                f.write(s1.content)

    @classmethod
    def splits(cls, config):
        data_root = os.path.join(
            config.data_folder, "kvasir_capsule", "labelled_images"
        )

        train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        test_transofrm = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        train_val_set = csv_dataset.DatasetCSV(
            config.train_val_split, data_root, transform=train_transform
        )
        test_set = csv_dataset.DatasetCSV(
            config.test_split, data_root, transform=test_transofrm
        )
        train_val_len = len(train_val_set)
        split_sizes = [
            int(train_val_len * config.train_percent),
            int(train_val_len * config.val_percent),
        ]

        # split_1 has odd number
        if train_val_len != split_sizes[0] + split_sizes[1]:
            split_sizes[0] = split_sizes[0] + 1

        train_set, val_set = data.random_split(train_val_set, split_sizes)

        return (
            cls(config, train_set),
            cls(config, val_set),
            cls(config, test_set),
        )

    @property
    def class_names(self):
        data_root = os.path.join(
            self.config.data_folder, "kvasir_capsule", "labelled_images"
        )
        Dataset = csv_dataset.DatasetCSV(self.config.train_val_split, data_root)
        return Dataset.classes

    @property
    def class_counts(self):
        data_root = os.path.join(
            self.config.data_folder, "kvasir_capsule", "labelled_images"
        )
        Dataset = csv_dataset.DatasetCSV(self.config.train_val_split, data_root)
        class_counts = {}
        for c in Dataset.classes:
            class_name = os.path.join(data_root, c)
            class_counts[Dataset.class_to_idx[c]] = len(os.listdir(class_name))
        return class_counts

    # retuns a list of enumerated labels from train_val split
    def get_label_list(self) -> list[int]:
        data_root = os.path.join(
            self.config.data_folder, "kvasir_capsule", "labelled_images"
        )
        Dataset = csv_dataset.DatasetCSV(self.config.train_val_split, data_root)
        label_list = [
            Dataset.class_to_idx[Dataset.imgs.iloc[i, 1]] for i in range(len(Dataset))
        ]
        return label_list
