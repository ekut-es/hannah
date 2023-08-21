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
import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

topdir = Path(__file__).parent.absolute() / ".."


@pytest.mark.parametrize(
    "features,window_fn",
    [
        ("mfcc", "hann"),
        ("mfcc", "ones"),
        ("logspec", "hann"),
        ("logspec", "ones"),
        ("melspec", "hann"),
        ("melspec", "ones"),
        ("spectrogram", "hann"),
        ("spectrogram", "ones"),
    ],
)
def test_instantiate_window_fn(features, window_fn):
    sample_rate = 16000
    config_path = Path("..") / "hannah" / "conf" / "features"
    with initialize(version_base=None, config_path=str(config_path)):
        cfg = compose(features, [f"window_fn={window_fn}"])
        if "sample_rate" in cfg:
            cfg["sample_rate"] = sample_rate

        feature_extractor = instantiate(cfg)

        res = feature_extractor(torch.rand((1, sample_rate)))

        assert res is not None
