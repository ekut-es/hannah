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
import warnings
from collections import Counter
from pathlib import Path

import omegaconf
from hydra import compose, initialize
from hydra.utils import get_class

topdir = Path(__file__).parent.absolute() / ".."
data_folder = topdir / "datasets"

config = f"""
defaults:
    - augmentation:
        - rand_augment
        - mixup
    - _self_


data_folder: {data_folder}
cls: hannah.datasets.vision.KvasirCapsuleDataset
dataset: kvasir_capsule
val_percent: 0.1
sampler: random
weighted_loss: false
anomaly: false

split: official
normalize:
    mean:
        - 0.5
        - 0.5
        - 0.5
    std:
        - 0.5
        - 0.5
        - 0.5


resolution: [224,224]
"""


def test_kvasir_labelled_dataset():
    data_folder = topdir / "datasets" / "kvasir_capsule" / "labelled_images"

    if not data_folder.exists():
        warnings.warn("Kvasir capsule is not available skipping test")
        return

    conf = omegaconf.OmegaConf.create(config)
    dataset_cls = get_class(conf.cls)
    train_set, dev_set, test_set = dataset_cls.splits(conf)

    total = 0
    class_counter = Counter()
    id_counter = Counter()
    class_names = train_set.class_names
    for dataset in [train_set, test_set]:
        for batch in dataset:
            x, y = batch
            class_name = class_names[y]
            class_counter[class_name] += 1
            id_counter[y] += 1
            total += 1
            if total % 100 == 0:
                print(total)

    for k, v in class_counter.items():
        print(f"{k}:", v)

    for k, v in sorted(id_counter.items(), key=lambda x: x[0]):
        print(f"{k}:", v)

    for k, v in sorted(id_counter.items(), key=lambda x: x[0]):
        weight = total / (v * len(id_counter))
        print(f"{k}:", weight)


def test_kvasir_unlabelled_dataset():
    unlabeled_folder = topdir / "datasets" / "kvasir_capsule" / "unlabelled_videos"

    if not unlabeled_folder.exists():
        warnings.warn(
            'Kvasir capsule is not available in "%s"skipping test' % unlabeled_folder
        )
        return

    data_folder = topdir / "datasets"

    with initialize(config_path="../hannah/conf/dataset"):
        cfg = compose(
            config_name="kvasir_unlabeled", overrides=[f"data_folder={data_folder}"]
        )
        dataset_cls = get_class(cfg.cls)
        train_set, dev_set, test_set = dataset_cls.splits(cfg)

    train_set[100]
    train_set[0]
    train_set[1]
    train_set[10000]
    train_set[200000]
    train_set[1000000]
    train_set[len(train_set) - 1]
