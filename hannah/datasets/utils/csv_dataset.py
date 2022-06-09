import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset


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
        self.imgs = pd.read_csv(csv_file)
        self.imgs.drop(["filename", "label"], axis=1)
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
