import os
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from .utils import log_execution_env_state

from .callbacks.distiller import DistillerCallback

from .lightning_model import SpeechClassifierModule

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.utilities.seed import seed_everything
from hydra.utils import instantiate

from . import conf  # noqa


@hydra.main(config_name="config", config_path="conf")
def main(config=DictConfig):
    seed_everything(config.seed)
    if torch.cuda.is_available():
        config.trainer.gpus = []

    log_execution_env_state()

    logging.info("Configuration: ")
    logging.info(OmegaConf.to_yaml(config))
    logging.info("Current working directory %s", os.getcwd())

    checkpoint_callback = instantiate(config.checkpoint)
    lit_module = SpeechClassifierModule(config)
    callbacks = []

    # TODO distiller only available without auto_lr because compatibility issues
    if "compress" in config or config.fold_bn >= 0:
        if config["auto_lr"]:
            raise Exception(
                "Automated learning rate finder is not compatible with compression"
            )
        callbacks.append(DistillerCallback(config.compress, fold_bn=config.fold_bn))

    logger = [
        TensorBoardLogger("./tb_logs", version="", name=""),
        CSVLogger(".", version="", name=""),
    ]

    if "backend" in config:
        backend = instantiate(config.backend)
        callbacks.append(backend)

    logging.info("type: '%s'", config.type)
    if config["type"] == "train":
        logging.info("Starting training")

        profiler = None
        if "profiler" in config:
            profiler = instantiate(config.profiler)

        # INIT PYTORCH-LIGHTNING
        lit_trainer = Trainer(
            **config.trainer,
            profiler=profiler,
            callbacks=callbacks,
            checkpoint_callback=checkpoint_callback,
            logger=logger
        )

        if config["auto_lr"]:
            # run lr finder (counts as one epoch)
            lr_finder = lit_trainer.lr_find(lit_module)

            # inspect results
            fig = lr_finder.plot()
            fig.savefig("./learning_rate.png")

            # recreate module with updated config
            suggested_lr = lr_finder.suggestion()
            config["lr"] = suggested_lr

        logging.info(ModelSummary(lit_module, "full"))

        # PL TRAIN
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
