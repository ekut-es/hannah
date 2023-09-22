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
"""Rhode island gastroenterology video capsule endoscopy dataset


    https://www.nature.com/articles/s41597-022-01726-3
    https://github.com/acharoen/Rhode-Island-GI-VCE-Technical-Validation
"""

import logging
import pathlib
import shutil

import albumentations as A
import pandas as pd
import torchvision
import tqdm
from albumentations.pytorch import ToTensorV2

from .base import ImageDatasetBase

BASE_PATH = pathlib.Path(__file__).parent
DATA_PATH = BASE_PATH / "ri_data"


LABELS = {
    "esophagus": 0,
    "stomach": 1,
    "small_bowel": 2,
    "colon": 3,
}

logger = logging.getLogger(__name__)


def read_official_val_train(study_folder: pathlib.Path, csv_file: pathlib.Path):
    data = pd.read_csv(csv_file)

    files = [str(study_folder / path) for path in data["path"].to_list()]
    labels = [label[2:] for label in data["organ"]]

    return files, labels


def read_official_test(study_folder: pathlib.Path, csv_file: pathlib.Path):
    data = pd.read_csv(csv_file, header=None)

    test_studies = data[0].values.tolist()

    files = []
    labels = []

    for study in test_studies:
        current_study_folder = study_folder / study
        assert current_study_folder.exists(), "Dataset download not complete"

        for label_folder in current_study_folder.iterdir():
            if label_folder.is_dir():
                label = label_folder.name[2:]

                for image_file in label_folder.glob("*.png"):
                    files.append(image_file)
                    labels.append(label)

    return files, labels


def split_train_set(csv_file: pathlib.Path, drop_rate: float):
    """Split train set in two and save as separate csv files."""
    assert 0.0 <= drop_rate <= 1.0
    data = pd.read_csv(csv_file)

    studies = data["path"].str.extract(r"(?P<STUDY>s\d+)").STUDY.unique()
    num_drop_labels = round(drop_rate * studies.size)
    studies_keep_labels = tuple(studies[num_drop_labels:])
    studies_drop_labels = tuple(studies[:num_drop_labels])

    X_train_keep = data[data["path"].str.startswith(studies_keep_labels)]
    X_train_drop = data[data["path"].str.startswith(studies_drop_labels)]
    X_train_keep.to_csv(
        DATA_PATH / f"path_train_keep_labels_{drop_rate}.csv", index=False
    )
    X_train_drop.to_csv(
        DATA_PATH / f"path_train_drop_labels_{drop_rate}.csv", index=False
    )


class RICapsuleDataset(ImageDatasetBase):
    DATA_FILES = [
        "Study 001-025.zip",
        "Study 101-125.zip",
        "Study 201-225.zip",
        "Study 301-325.zip",
        "Study 401-424.zip",
        "Study 026-050.zip",
        "Study 126-150.zip",
        "Study 226-250.zip",
        "Study 326-350.zip",
        "Study 051-075.zip",
        "Study 151-175.zip",
        "Study 251-275.zip",
        "Study351375.zip",
        "Study 076-100.zip",
        "Study 176-200.zip",
        "Study276300.zip",
        "Study 376-400.zip",
    ]

    PREPARED_STAMP = ".prepared"

    @classmethod
    def prepare(cls, config):
        download_folder = pathlib.Path(config["download_folder"])
        data_folder = pathlib.Path(config["data_folder"]) / "ri_capsule"
        zip_files = list(download_folder.glob("*.zip"))
        tmp_folder = data_folder / "tmp"
        study_folder = data_folder / "studies"

        stamp = data_folder / cls.PREPARED_STAMP
        if stamp.exists():
            return

        zip_file_names = [f.name for f in zip_files]

        assert set(zip_file_names) == set(cls.DATA_FILES)

        logger.info("Extracting data files")
        for zip_file in tqdm.tqdm(zip_files):
            torchvision.datasets.utils.extract_archive(
                str(zip_file.absolute()), str(tmp_folder.absolute())
            )
            for study_file in tqdm.tqdm(tmp_folder.glob("**/*.zip")):
                torchvision.datasets.utils.extract_archive(
                    str(study_file.absolute()), str(study_folder.absolute())
                )
            shutil.rmtree(tmp_folder)

        with stamp.open("w"):
            logger.info("Finshed dataset preparation")

    @classmethod
    def splits(cls, config):
        data_folder = pathlib.Path(config["data_folder"]) / "ri_capsule"
        study_folder = data_folder / "studies"

        if (
            "drop_labels" in config
            and config.drop_labels is not None
            and config.drop_labels > 0.0
        ):
            logger.info(
                "Dropping labels of %i %% of training set.", config.drop_labels * 100
            )
            split_train_set(DATA_PATH / "path_train.csv", config.drop_labels)
            train_csv = DATA_PATH / f"path_train_keep_labels_{config.drop_labels}.csv"
            X_train_unlabeled, y_train_unlabeled = read_official_val_train(
                study_folder,
                DATA_PATH / f"path_train_drop_labels_{config.drop_labels}.csv",
            )
        else:
            train_csv = DATA_PATH / "path_train.csv"
            X_train_unlabeled = []
            y_train_unlabeled = []

        X_train, y_train = read_official_val_train(study_folder, train_csv)

        X_val, y_val = read_official_val_train(
            study_folder, DATA_PATH / "path_valid.csv"
        )
        X_test, y_test = read_official_test(study_folder, DATA_PATH / "path_test.csv")

        transform = A.Compose(
            [
                A.augmentations.geometric.resize.Resize(
                    config.sensor.resolution[0], config.sensor.resolution[1]
                ),
                ToTensorV2(),
            ]
        )
        train_set = cls(X_train, y_train, list(LABELS.keys()), transform=transform)
        train_set_unlabeled = cls(
            X_train_unlabeled,
            y_train_unlabeled,  # FIXME labels must not be used
            list(LABELS.keys()),
            transform=transform,
        )
        val_set = cls(X_val, y_val, list(LABELS.keys()))
        test_set = cls(X_test, y_test, list(LABELS.keys()))

        # RANDOM, RANDOM_PER_STUDY Splits
        # preprocessing,

        return (
            train_set,
            train_set_unlabeled,
            val_set,
            test_set,
        )
