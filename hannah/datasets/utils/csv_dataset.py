from logging import root
import numpy as np
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from typing import Callable, Optional, Dict, List, Tuple, Sequence, Any
import pandas as pd
import os
from PIL import Image
from torch import default_generator, randperm
from torch._utils import _accumulate


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class DatasetCSV(VisionDataset):
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return find_classes(directory)

    def __init__(
        self,
        csv_file: str,
        root: str,
        transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform)
        self.root = root
        self.csv_file = csv_file
        self.imgs = pd.read_csv(csv_file)
        classes, class_to_idx = self.find_classes(self.root)
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        label = self.imgs.iloc[idx, 1]
        image_path = os.path.join(self.root, label, self.imgs.iloc[idx, 0])
        image = pil_loader(image_path)
        target_index = self.class_to_idx[label]

        if self.transform is not None:
            image = self.transform(image)

        return image, target_index


class Subset(DatasetCSV):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        csv_file (str) : Path to dataset csv file
        root (str) : Path to image folder
        tramsform : Transforms to be applied on images
    """
    dataset: DatasetCSV
    indices: Sequence[int]

    def __init__(
        self,
        dataset: DatasetCSV,
        indices: Sequence[int],
        csv_file: str,
        root: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(csv_file, root, transform=transform)
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(
    dataset: DatasetCSV, lengths: Sequence[int], generator=default_generator
):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()
    train_set_indices = [indices[i] for i in range(len(indices) - lengths[1])]
    val_set_indices = [indices[i] for i in range(len(indices) - lengths[0])]
    return (
        [
            Subset(
                dataset,
                indices[offset - length : offset],
                csv_file=dataset.csv_file,
                root=dataset.root,
            )
            for offset, length in zip(_accumulate(lengths), lengths)
        ],
        train_set_indices,
        val_set_indices,
    )
