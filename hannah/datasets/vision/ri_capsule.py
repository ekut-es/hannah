#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
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

import pandas as pd
import torchvision
import tqdm

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
        assert not current_study_folder.exists(), "Dataset download not complete"

        for label_folder in current_study_folder.iterdir():
            if label_folder.is_dir():
                label = label_folder.name[2:]

                for image_file in label_folder.glob("*.png"):
                    files.append(image_file)
                    labels.append(label)

    return files, labels


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
            logger.info("Finshed datset preparation")

    @classmethod
    def splits(cls, config):
        data_folder = pathlib.Path(config["data_folder"]) / "ri_capsule"
        study_folder = data_folder / "studies"

        X_train, y_train = read_official_val_train(
            study_folder, DATA_PATH / "path_train.csv"
        )
        X_val, y_val = read_official_val_train(
            study_folder, DATA_PATH / "path_valid.csv"
        )
        X_test, y_test = read_official_test(study_folder, DATA_PATH / "path_test.csv")

        train_set = cls(X_train, y_train, list(LABELS.keys()))
        val_set = cls(X_val, y_val, list(LABELS.keys()))
        test_set = cls(X_test, y_test, list(LABELS.keys()))

        # RANDOM, RANDOM_PER_STUDY Splits
        # preprocessing,

        return (
            train_set,
            val_set,
            test_set,
        )
