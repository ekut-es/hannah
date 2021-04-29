import os

import torchvision
import torchvision.datasets as datasets
import torch.utils.data as data
import torch

from .base import AbstractDataset


class Cifar10Dataset(AbstractDataset):
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

    @classmethod
    def prepare(cls, config):
        data_folder = config.data_folder
        root_folder = os.path.join(data_folder, "CIFAR10")

        test_set = datasets.CIFAR10(root_folder, train=False, download=True)

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

    @property
    def class_counts(self):
        return None

    def __getitem__(self, index):
        data, target = self.dataset[index]

        data = torch.Tensor(data)
        target = torch.Tensor(target)

        return data, data.size(), target, 1

    @classmethod
    def splits(cls, config):
        data_folder = config.data_folder
        root_folder = os.path.join(data_folder, "CIFAR10")

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

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

    def size(self):
        dim = self.dataset[0][0].size()

        return list(dim)

    def __len__(self):
        return len(self.dataset)
