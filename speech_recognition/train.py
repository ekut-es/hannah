import os
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from .utils import log_execution_env_state

from .callbacks.distiller import DistillerCallback
from .callbacks.summaries import MacSummaryCallback

from .callbacks.optimization import HydraOptCallback

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import GPUStatsMonitor
from pytorch_lightning.utilities.seed import seed_everything
from hydra.utils import instantiate

from . import conf  # noqa


def handleDataset(config=DictConfig):
    lit_module = instantiate(
        config.module,
        dataset=config.dataset,
        model=config.model,
        optimizer=config.optimizer,
        features=config.features,
        scheduler=config.get("scheduler", None),
        normalizer=config.get("normalizer", None),
    )
    lit_module.prepare_data()


def train(config=DictConfig):
    seed_everything(config.seed)
    if not torch.cuda.is_available():
        config.trainer.gpus = None

    log_execution_env_state()

    logging.info("Configuration: ")
    logging.info(OmegaConf.to_yaml(config))
    logging.info("Current working directory %s", os.getcwd())

    checkpoint_callback = instantiate(config.checkpoint)
    lit_module = instantiate(
        config.module,
        dataset=config.dataset,
        model=config.model,
        optimizer=config.optimizer,
        features=config.features,
        scheduler=config.get("scheduler", None),
        normalizer=config.get("normalizer", None),
    )
    callbacks = []

    # TODO distiller only available without auto_lr because compatibility issues
    if "compress" in config:
        if config["auto_lr"]:
            raise Exception(
                "Automated learning rate finder is not compatible with compression"
            )
        callbacks.append(DistillerCallback(config.compress))

    logger = [
        TensorBoardLogger(".", name="", default_hp_metric=False),
        CSVLogger(".", name=""),
    ]

    if "backend" in config:
        backend = instantiate(config.backend)
        callbacks.append(backend)

    logging.info("type: '%s'", config.type)

    logging.info("Starting training")

    profiler = None
    if "profiler" in config:
        profiler = instantiate(config.profiler)

    lr_monitor = LearningRateMonitor()
    callbacks.append(lr_monitor)

    if "gpu_stats" in config and config.gpu_stats:
        gpu_stats = GPUStatsMonitor()
        callbacks.append(gpu_stats)

    mac_summary_callback = MacSummaryCallback()
    callbacks.append(mac_summary_callback)

    opt_monitor = config.get("monitor", ["val_loss"])
    opt_callback = HydraOptCallback(monitor=opt_monitor)
    callbacks.append(opt_callback)

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

    # PL TRAIN
    lit_trainer.fit(lit_module)

    # PL TEST
    lit_trainer.test(ckpt_path=None)

    return opt_callback.result()


def eval(model_name, config):
    lit_trainer, lit_module, profiler = build_trainer(model_name, config)
    test_loader = lit_module.test_dataloader()

    lit_module.eval()
    lit_module.freeze()

    results = None
    for batch in test_loader:
        result = lit_module.forward(batch[0])
        if results is None:
            results = result
        else:
            results = torch.cat([results, result])
    return results


@hydra.main(config_name="config", config_path="conf")
def main(config=DictConfig):

    if config["type"] == "train":
        return train(config)
    elif config["type"] == "eval":
        return eval(config)
    elif config["type"] == "eval_vad_keyword":
        logging.error("eval_vad_keyword is not supported at the moment")
    elif config["type"] == "dataset":
        print("Only the dataset will be created and downloaded")
        handleDataset(config)


if __name__ == "__main__":
    main()
