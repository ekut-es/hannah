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
import logging
import os
import shutil
import sys
import traceback
from copy import deepcopy

import omegaconf
import torch
import torch.nn as nn
from hydra.utils import instantiate
from lightning.fabric.utilities.seed import reset_seed, seed_everything
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from hannah.nas.functional_operators.executor import BasicExecutor
from hannah.nas.parameters.parametrize import set_parametrization
from hannah.nas.search.utils import save_graph_to_file, setup_callbacks
from hannah.utils.utils import common_callbacks

msglogger = logging.getLogger(__name__)


class SimpleModelTrainer:
    def __init__(self, per_process_memory_fraction = None) -> None:
        self.per_process_memory_fraction = per_process_memory_fraction

    def build_model(self, model, parameters):
        # model_instance = deepcopy(model)
        set_parametrization(parameters, model.parametrization(flatten=True))
        # model_instance.initialize()
        # model = model_instance
        mod = BasicExecutor(model)
        mod.initialize()

        return mod

    def run_training(self, model, num, global_num, config):
        # num is the number of jobs global_num is the number of models to be created
        if os.path.exists(str(num)):
            shutil.rmtree(str(num))

        os.makedirs(str(num), exist_ok=True)

        try:
            os.chdir(str(num))
            config = OmegaConf.create(config)
            logger = TensorBoardLogger(".")

            self.setup_seed(config)
            print("=======")
            print(config.trainer.devices)
            self.setup_devices(num, config, logger)
            print(config.trainer.devices)
            print("=========")
            callbacks, opt_monitor, opt_callback = setup_callbacks(config)
            try:
                trainer = instantiate(
                    config.trainer, callbacks=callbacks, logger=logger
                )
                module = model
                trainer.fit(module)
                ckpt_path = "best"
                if trainer.fast_dev_run:
                    logging.warning(
                        "Trainer is in fast dev run mode, switching off loading of best model for test"
                    )
                    ckpt_path = None

                reset_seed()
                trainer.validate(ckpt_path=ckpt_path, verbose=False)
                res = opt_callback.result(dict=True)
                save_graph_to_file(global_num, res, module)
            except Exception as e:
                msglogger.critical("Training failed with exception")
                msglogger.critical(str(e))
                # print(traceback.format_exc())
                # sys.exit(1)

                res = {}
                for monitor in opt_monitor:
                    # res[monitor] = float("inf")
                    res[
                        monitor
                    ] = 1  # FIXME: "inf" causes errors in performance prediction. Find "worst" value for each respective metric?

            return res
        finally:
            os.chdir("..")

    def set_result_handler(self, result_handler):
        self.result_handler = result_handler

    def setup_seed(self, config):
        seed = config.get("seed", 1234)
        if isinstance(seed, list) or isinstance(seed, omegaconf.ListConfig):
            seed = seed[0]
        seed_everything(seed, workers=True)

    def setup_devices(self, num, config, logger):
        if config.trainer.devices is not None:
            if isinstance(config.trainer.devices, int):
                num_devices = config.trainer.devices
                device = num % num_devices
            elif len(config.trainer.devices) == 0:
                num_devices = torch.cuda.device_count()
                device = num % num_devices
            else:
                device = config.trainer.devices[num % len(config.trainer.devices)]

            if device >= torch.cuda.device_count():
                msglogger.warning(
                    "GPU %d is not available on this device using GPU %d instead",
                    device,
                    device % torch.cuda.device_count(),
                )
                device = device % torch.cuda.device_count()

            if self.per_process_memory_fraction:
                torch.cuda.set_per_process_memory_fraction(self.per_process_memory_fraction, device=device)
                
            config.trainer.devices = [device]
