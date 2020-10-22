import os
import logging

import torch

import hydra
from omegaconf import DictConfig, OmegaConf

from .utils import set_seed, log_execution_env_state

from .callbacks.backends import OnnxTFBackend, OnnxruntimeBackend, TorchMobileBackend
from .callbacks.distiller import DistillerCallback

from .lightning_model import SpeechClassifierModule

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.memory import ModelSummary

from . import conf  # noqa


@hydra.main(config_name="config", config_path="conf")
def main(config=DictConfig):
    set_seed(config)
    config.cuda = config.cuda if torch.cuda.is_available() else None
    gpu_no = config["gpu_no"]
    n_epochs = config["n_epochs"]  # max epochs

    log_execution_env_state()

    logging.info(OmegaConf.to_yaml(config))
    logging.info("Current working directory %s", os.getcwd())

    # Configure checkpointing
    checkpoint_callback = ModelCheckpoint(
        filepath="./checkpoints",
        save_top_k=5,
        verbose=False,
        monitor="val_loss",
        mode="min",
        prefix="",
    )

    lit_module = SpeechClassifierModule(config)

    kwargs = {
        "max_epochs": n_epochs,
        "default_root_dir": ".",
        "row_log_interval": 1,  # enables logging of metrics per step/batch
        "checkpoint_callback": checkpoint_callback,
        "callbacks": [],
    }

    # TODO distiller only available without auto_lr because compatibility issues
    if "compress" in config or config.fold_bn >= 0:
        if config["auto_lr"]:
            raise Exception(
                "Automated learning rate finder is not compatible with compression"
            )
        callbacks = kwargs["callbacks"]
        callbacks.append(DistillerCallback(config.compress, fold_bn=config.fold_bn))
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
        TensorBoardLogger("./tb_logs", version="", name=""),
        CSVLogger(".", version="", name=""),
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
            fig.savefig("./learning_rate.png")
            # recreate module with updated config
            suggested_lr = lr_finder.suggestion()
            config["lr"] = suggested_lr

        # PL TRAIN
        logging.info(ModelSummary(lit_module, "full"))
        lit_trainer.fit(lit_module)

        # PL TEST
        lit_trainer.test(ckpt_path=None)

        if config["profile"]:
            logging.info(profiler.summary())

        return lit_trainer

    elif config["type"] == "eval":
        logging.error("eval mode is not supported at the moment")
    elif config["type"] == "eval_vad_keyword":
        logging.error("eval_vad_keyword is not supported at the moment")


if __name__ == "__main__":
    main()
