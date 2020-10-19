import argparse
import sys
import json
import os
import logging

import torch

import hydra
from omegaconf import DictConfig, OmegaConf


from . import models as mod
from . import dataset
from .utils import set_seed, config_pylogger, log_execution_env_state
from .config_utils import get_config_logdir

from .config import ConfigBuilder, ConfigOption
from .callbacks.backends import OnnxTFBackend, OnnxruntimeBackend, TorchMobileBackend
from .callbacks.distiller import DistillerCallback

from .utils import _fullname

from .lightning_model import SpeechClassifierModule

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.memory import ModelSummary


@hydra.main(config_name="config")
def main(config=DictConfig):
    set_seed(config)
    gpu_no = config["gpu_no"]
    n_epochs = config["n_epochs"]  # max epochs
    log_dir = get_config_logdir(
        config["model_name"], config
    )  # path for logs and checkpoints

    log_execution_env_state()

    logging.info(OmegaConf.to_yaml(config))
    logging.info("Current working directory %s", os.getcwd())

    # Configure checkpointing
    checkpoint_callback = ModelCheckpoint(
        filepath=log_dir,
        save_top_k=-1,  # with PL 0.9.0 only possible to save every epoch
        verbose=True,
        monitor="checkpoint_on",
        mode="min",
        prefix="",
    )

    lit_module = SpeechClassifierModule(config)

    kwargs = {
        "max_epochs": n_epochs,
        "default_root_dir": log_dir,
        "row_log_interval": 1,  # enables logging of metrics per step/batch
        "checkpoint_callback": checkpoint_callback,
        "callbacks": [],
    }

    # TODO distiller only available without auto_lr because compatibility issues
    if config["compress"] and not config["auto_lr"]:
        callbacks = kwargs["callbacks"]
        callbacks.append(
            DistillerCallback(config["compress"], fold_bn=config["fold_bn"])
        )
        kwargs.update({"callbacks": callbacks})

    if config["cuda"]:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        kwargs.update({"gpus": [gpu_no]})

    if "limits_datasets" in config:
        limits = config["limits_datasets"]
        kwargs.update(
            {
                "limit_train_batches": limits[0],
                "limit_val_batches": limits[1],
                "limit_test_batches": limits[2],
            }
        )

    loggers = [
        TensorBoardLogger(log_dir + "/tb_logs", version="", name=""),
        CSVLogger(log_dir, version="", name=""),
    ]
    kwargs["logger"] = loggers

    if config["backend"] == "torchmobile":
        backend = TorchMobileBackend()
        kwargs["callbacks"].append(backend)
    if config["backend"] == "onnx-tf":
        backend = OnnxTFBackend()
        kwargs["callbacks"].append(backend)
    elif config["backend"] == "onnxrt":
        backend = OnnxruntimeBackend()
        kwargs["callbacks"].append(backend)

    if config["fast_dev_run"]:
        kwargs.update({"fast_dev_run": True})

    if config["type"] == "train":

        if config["profile"]:
            profiler = AdvancedProfiler()
            kwargs.update({"profiler": profiler})

        # INIT PYTORCH-LIGHTNING
        lit_trainer = Trainer(**kwargs)

        if config["auto_lr"]:
            # run lr finder (counts as one epoch)
            lr_finder = lit_trainer.lr_find(lit_module)
            # inspect results
            fig = lr_finder.plot()
            fig.savefig(f"{log_dir}/learing_rate.png")
            # recreate module with updated config
            suggested_lr = lr_finder.suggestion()
            config["lr"] = suggested_lr
            lit_module = SpeechClassifierModule(dict(config))

        # PL TRAIN
        logging.info(ModelSummary(lit_module, "full"))
        lit_trainer.fit(lit_module)

        # PL TEST
        lit_trainer.test(ckpt_path=None)

        if config["profile"]:
            logging.info(profiler.summary())

    elif config["type"] == "eval":
        logging.error("eval mode is not supported at the moment")
    elif config["type"] == "eval_vad_keyword":
        logging.error("eval_vad_keyword is not supported at the moment")


if __name__ == "__main__":
    main()
