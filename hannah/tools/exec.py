#
# Copyright (c) 2024 Hannah contributors.
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

# Compiles a model config using a target backend
import hydra
import os
from omegaconf import DictConfig
import logging
import pathlib
import sys

from .. import conf  # noqa
from ..logo import print_logo
from ..train import instantiate_module

import pandas as pd

logger = logging.getLogger(__name__)


@hydra.main(config_name="config", config_path="../conf", version_base="1.2")
def main(config: DictConfig):
    print_logo()

    lit_module = instantiate_module(config)
    lit_module.setup("predict")

    # Now we load the weights
    checkpoint_folder = pathlib.Path.cwd() / "checkpoints"

    best_ckpt = checkpoint_folder / "best.ckpt"
    last_ckpt = checkpoint_folder / "last.ckpt"

    input_file = None
    if config.get("input_file", None):
        input_file = pathlib.Path(config.input_file)
    elif best_ckpt.exists():
        input_file = best_ckpt
    elif last_ckpt.exists():
        input_file = last_ckpt

    if input_file is not None:
        logger.info("Loading weights from input_file: %s", str(input_file))
        import torch

        ckpt = torch.load(input_file, map_location="cpu")
        state_dict = ckpt["state_dict"]

        lit_module.load_state_dict(state_dict, strict=False)

    else:
        logger.info("Could not find input file using random weights for intialization")

    logger.info("Current working directory: %s", os.getcwd())
    if config.get("backend"):
        backend = hydra.utils.instantiate(config.backend)
    else:
        logger.error(
            "No inference backend given please configure a backend using backend=''"
        )
        sys.exit(-1)

    input = lit_module.example_input_array

    backend.prepare(lit_module)

    result = backend.profile(input)

    metrics = result.metrics

    logger.info("Target Metrics:")
    for k, v in metrics.items():
        logger.info("%s: %s", str(k), str(v))
