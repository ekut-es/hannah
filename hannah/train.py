#
# Copyright (c) 2023 Hannah contributors.
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
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Type, Union

import pandas as pd
import tabulate
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.seed import reset_seed, seed_everything

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
        if not torch.cuda.is_available():
            config.trainer.gpus = None

        if isinstance(config.trainer.gpus, int):
            config.trainer.gpus = auto_select_gpus(config.trainer.gpus)

        if not config.trainer.fast_dev_run and not config.get("resume", False):
            clear_outputs()

        logging.info("Configuration: ")
        logging.info(OmegaConf.to_yaml(config))
        logging.info("Current working directory %s", os.getcwd())
        lit_module = instantiate(
            config.module,
            dataset=config.dataset,
            model=config.model,
            optimizer=config.optimizer,
            features=config.get("features", None),
            augmentation=config.get("augmentation", None),
            scheduler=config.get("scheduler", None),
            normalizer=config.get("normalizer", None),
            gpus=config.trainer.get("gpus", None),
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
        lit_trainer = instantiate(
            config.trainer,
            profiler=profiler,
            callbacks=callbacks,
            logger=logger,
            _convert_="partial",
        )

        if config.get("input_file", None):
            msglogger.info("Loading initial weights from model %s", config.input_file)
            lit_module.setup("train")
            input_ckpt = pl_load(config.input_file)
            lit_module.load_state_dict(input_ckpt["state_dict"], strict=False)

        if config["auto_lr"]:
            # run lr finder (counts as one epoch)
            lr_finder = lit_trainer.lr_find(lit_module)

            # inspect results
            fig = lr_finder.plot()
            fig.savefig("./learning_rate.png")

            # recreate module with updated config
            suggested_lr = lr_finder.suggestion()
            config["lr"] = suggested_lr

        lit_trainer.tune(lit_module)

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

        if config.get("compression", None) and (
            config.get("compression").get("clustering", None)
            or config.get("compression").get("decomposition", None)
        ):
            # FIXME: this is a bad workaround
            lit_trainer.save_checkpoint("last")
            ckpt_path = "last"
        else:
            ckpt_path = "best"

        lit_module.save()

        if not lit_trainer.fast_dev_run:
            reset_seed()
            lit_trainer.validate(ckpt_path=ckpt_path, verbose=validate_output)

            # PL TEST
            reset_seed()
            lit_trainer.test(ckpt_path=ckpt_path, verbose=validate_output)

            if checkpoint_callback and checkpoint_callback.best_model_path:
                shutil.copy(checkpoint_callback.best_model_path, "best.ckpt")

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
