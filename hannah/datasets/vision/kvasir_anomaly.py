import logging
import os
from pathlib import Path
from typing import List

from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

from hannah.modules.augmentation import rand_augment

from ..base import AbstractDataset

logger = logging.getLogger(__name__)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class KvasirCapsuleAnomalyDataset(AbstractDataset):
    DOWNLOAD_URL = "https://files.osf.io/v1/resources/dv2ag/providers/googledrive/labelled_images/?zip="

    def __init__(self, config, data, targets, transform=None):
        self.config = config
        self.data = data
        self.targets = targets
        self.transform = transform
        self.classes = ["Normal", "Anomaly"]

    def __getitem__(self, index):
        data = pil_loader(self.data[index])
        target = self.targets[index]
        if self.transform:
            data = self.transform(data)
        return data, target

    def __len__(self):
        return len(self.dataset)

    @classmethod
    def prepare(cls, config):
        pass

    @classmethod
    def splits(cls, config):
        data_root = Path(
            os.path.join(config.data_folder, "kvasir_capsule", "labelled_images")
        )

        folder2class = {
            "Angiectasia": 1,
            "Blood": 1,
            "Erosion": 1,
            "Erythematous": 1,
            "Foreign Bodies": 1,
            "Ileo-cecal valve": 0,
            "Lymphangiectasia": 1,
            "Normal": 0,
            "Pylorus": 0,
            "Reduced Mucosal View": 1,
            "Ulcer": 1,
        }

        X = []
        y = []

        for k, v in folder2class.items():
            current_folder = data_root / k
            for file in current_folder.glob("*.jpg"):
                X.append(file)
                y.append(v)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=1
        )  # 0.25 x 0.8 = 0.2

        train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Resize(224),
                rand_augment.RandAugment(
                    config.augmentations.rand_augment.N,
                    config.augmentations.rand_augment.M,
                    config=config,  # FIXME: make properly configurable
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        return (
            cls(
                config,
                X_train,
                y_train,
                transform=train_transform,
            ),
            cls(
                config,
                X_val,
                y_val,
                transform=test_transform,
            ),
            cls(
                config,
                X_test,
                y_test,
                transform=test_transform,
            ),
        )

    def __len__(self):
        assert len(self.data) == len(self.targets)
        return len(self.data)

    @property
    def class_names(self):
        return self.classes

    @property
    def class_counts(self):

        counts = {}
        for name in self.class_names:
            counts[name] = 0
        for t in self.targets:
            counts[self.class_names[t]] += 1

        return counts

    @property
    def num_classes(self):
        return len(self.class_counts)

    # retuns a list of class index for every sample
    @property
    def get_label_list(self) -> List[int]:
        return self.targets

    @property
    def class_names_abbreviated(self) -> List[str]:
        return [cn[0:3] for cn in self.class_names]

    @property
    def weights(self):
        # FIXME: move to base classes
        counts = list(self.class_counts.values())
        weights = [1 / i for i in counts]
        return weights
