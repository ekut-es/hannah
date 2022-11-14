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

import os
import pathlib
import sys

import hydra
import pytest
from omegaconf import OmegaConf

import hannah.conf

topdir = pathlib.Path(__file__).parent.absolute() / ".."
config_dir = topdir / "hannah" / "conf"

project_config_dir = topdir / "configs"


@pytest.mark.skipif(
    sys.version_info >= (3, 9), reason="currently does not run on python 3.10 or later"
)
def test_parse_configs():
    """This simply tests that all configs are parsable by hydra"""
    for config in config_dir.glob("*.yaml"):
        with hydra.initialize_config_module(
            version_base="1.2", config_module="hannah.conf", job_name="test_config"
        ):
            cfg = hydra.compose(config_name=config.stem)
            print("")
            print("config:", config.stem)
            print(OmegaConf.to_yaml(cfg))

    for config in project_config_dir.glob("*/*.yaml"):
        with hydra.initialize_config_module(
            version_base="1.2", config_module="hannah.conf", job_name="test_config"
        ):
            cfg = hydra.compose(config_name=config.stem)
            print("")
            print("config:", config.stem)
            print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    test_parse_configs()
