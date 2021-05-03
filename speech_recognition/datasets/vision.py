import os

import torchvision
import torchvision.datasets as datasets
import torch.utils.data as data
import torch

from .base import AbstractDataset


class VisionDatasetBase(AbstractDataset):
    """Wrapper around torchvision classification datasets"""

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

    @property
    def class_counts(self):
        return None

    def __getitem__(self, index):
        data, target = self.dataset[index]

        data = torch.tensor(data)
        target = torch.tensor([target])

        return data, data.size(), target, 1

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
        classes = [
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
        return classes

    @classmethod
    def splits(cls, config):
        data_folder = config.data_folder
        root_folder = os.path.join(data_folder, "CIFAR10")

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                ),
            ]
        )

        test_set = datasets.CIFAR10(
            root_folder, train=False, download=False, transform=transform
        )
        train_val_set = datasets.CIFAR10(
            root_folder, train=True, download=False, transform=transform
        )
        train_val_len = len(train_val_set)

        split_sizes = [int(train_val_len * 0.9), int(train_val_len * 0.1)]
        train_set, val_set = data.random_split(train_val_set, split_sizes)

        return cls(config, test_set), cls(config, val_set), cls(config, train_set)


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

        return cls(config, test_data), cls(config, val_data), cls(config, train_data)

    @property
    def class_names(self):
        return [f"class{n}" for n in range(self.config.num_classes)]
