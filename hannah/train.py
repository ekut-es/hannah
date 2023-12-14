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
from pathlib import Path
from typing import Any, Dict, List, Type, Union

import pandas as pd
import tabulate
import torch
import torch.nn as nn
from hydra.utils import get_class, instantiate
from lightning.fabric.utilities.seed import reset_seed, seed_everything
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from . import conf  # noqa
from .callbacks.optimization import HydraOptCallback
from .utils import (
    auto_select_gpus,
    clear_outputs,
    common_callbacks,
    log_execution_env_state,
)

msglogger: logging.Logger = logging.getLogger(__name__)


@rank_zero_only
def handle_dataset(config=DictConfig):
    lit_module = instantiate(
        config.module,
        dataset=config.dataset,
        model=config.model,
        optimizer=config.optimizer,
        features=config.get("features", None),
        scheduler=config.get("scheduler", None),
        normalizer=config.get("normalizer", None),
        _recursive_=False,
    )
    lit_module.prepare_data()


def train(
    config: DictConfig,
) -> Union[float, Dict[Any, float], List[Union[float, Dict[Any, float]]]]:
    test_output = []
    results = []
    if isinstance(config.seed, int):
        config.seed = [config.seed]
    validate_output = False
    if hasattr(config, "validate_output") and isinstance(config.validate_output, bool):
        validate_output = config.validate_output

    for seed in config.seed:
        seed_everything(seed, workers=True)

        if isinstance(config.trainer.devices, int) and config.trainer.accelerator in [
            "gpu",
            "auto",
        ]:
            config.trainer.devices = auto_select_gpus(config.trainer.devices)

        if not config.trainer.fast_dev_run and not config.get("resume", False):
            clear_outputs()

        logging.info("Configuration: ")
        logging.info(OmegaConf.to_yaml(config))
        logging.info("Current working directory %s", os.getcwd())

        if config.get("input_file", None):
            msglogger.info("Loading initial weights from model %s", config.input_file)
            lit_module = get_class(config.module._target_).load_from_checkpoint(
                config.input_file
            )
        else:
            lit_module = instantiate(
                config.module,
                dataset=config.dataset,
                model=config.model,
                optimizer=config.optimizer,
                features=config.get("features", None),
                augmentation=config.get("augmentation", None),
                scheduler=config.get("scheduler", None),
                normalizer=config.get("normalizer", None),
                unlabeled_data=config.get("unlabeled_data"),
                pseudo_labeling=config.get("pseudo_labeling", None),
                _recursive_=False,
            )

        profiler = None
        if config.get("profiler", None):
            profiler = instantiate(config.profiler)

        logger = [
            TensorBoardLogger(
                ".", version=None, name="", default_hp_metric=False, log_graph=True
            )
        ]
        if config.trainer.get("stochastic_weight_avg", False):
            logging.critical(
                "CSVLogger is not compatible with logging with SWA, disabling csv logger"
            )
        else:
            logger.append(CSVLogger(".", version=None, name=""))

        callbacks = []
        if config.get("backend", None):
            backend = instantiate(config.backend)
            callbacks.append(backend)

        callbacks.extend(list(common_callbacks(config)))

        opt_monitor = config.get("monitor", ["val_error"])
        opt_callback = HydraOptCallback(monitor=opt_monitor)
        callbacks.append(opt_callback)

        checkpoint_callback = instantiate(config.checkpoint)
        callbacks.append(checkpoint_callback)

        # INIT PYTORCH-LIGHTNING
        lit_trainer: Trainer = instantiate(
            config.trainer,
            profiler=profiler,
            callbacks=callbacks,
            logger=logger,
            _convert_="partial",
        )

        logging.info("Starting training")
        # PL TRAIN
        ckpt_path = None
        if config.get("resume", False):
            expected_ckpt_path = Path(".") / "checkpoints" / "last.ckpt"
            if expected_ckpt_path.exists():
                logging.info(
                    "Resuming training from checkpoint: %s", str(expected_ckpt_path)
                )
                ckpt_path = str(expected_ckpt_path)
            else:
                logging.info(
                    "Checkpoint '%s' not found restarting training from scratch",
                    str(expected_ckpt_path),
                )
        lit_trainer.fit(lit_module, ckpt_path=ckpt_path)

        if lit_trainer.checkpoint_callback.kth_best_model_path:
            ckpt_path = "best"
        ckpt_path = None

        if not lit_trainer.fast_dev_run:
            reset_seed()
            lit_trainer.validate(ckpt_path=ckpt_path, verbose=validate_output)

            if not config.get("skip_test", False):
                # PL TEST
                reset_seed()
                lit_trainer.test(ckpt_path=ckpt_path, verbose=validate_output)

                test_output.append(opt_callback.test_result())

            results.append(opt_callback.result())

    @rank_zero_only
    def summarize_test(test_output) -> None:
        if not test_output:
            return
        result_frame = pd.DataFrame.from_dict(test_output)
        if result_frame.empty:
            return
        result_frame.to_json("test_results.json")
        result_frame.to_pickle("test_results.pkl")

        description = result_frame.describe()
        description = description.fillna(0.0)

        res = description.loc[["mean", "std", "count"]]

        desc_table = tabulate.tabulate(
            res.transpose(),
            headers=["Metric", "Mean", "Std", "Count"],
            tablefmt="github",
        )
        msglogger.info("Averaged Result Metrics:\n%s", desc_table)

    summarize_test(test_output)

    if len(results) == 1:
        return results[0]
    else:
        return results


def nas(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    nas_trainer = instantiate(config.nas, parent_config=config, _recursive_=False)
    nas_trainer.run()
