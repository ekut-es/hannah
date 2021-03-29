import abc
import os
import logging
import shutil
import pathlib

from omegaconf import DictConfig, OmegaConf
import torch

from speech_recognition.utils import log_execution_env_state

from speech_recognition.callbacks.summaries import MacSummaryCallback

from speech_recognition.callbacks.optimization import HydraOptCallback
from speech_recognition.callbacks.pruning import PruningAmountScheduler
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import GPUStatsMonitor
from pytorch_lightning.utilities.seed import seed_everything
from hydra.utils import instantiate

from speech_recognition import conf  # noqa


class BaseValidator(metaclass=abc.ABCMeta):
    def __init__(self, config):
        self.config = config

    def handleDataset(self):
        lit_module = instantiate(
            self.config.module,
            dataset=self.config.dataset,
            model=self.config.model,
            optimizer=self.config.optimizer,
            features=self.config.features,
            scheduler=self.config.get("scheduler", None),
            normalizer=self.config.get("normalizer", None),
        )
        lit_module.prepare_data()

    def train(self):
        seed_everything(self.config.seed)
        if not torch.cuda.is_available():
            self.config.trainer.gpus = None

        if not self.config.trainer.fast_dev_run:
            current_path = pathlib.Path(".")
            for component in current_path.iterdir():
                if component.name == "checkpoints":
                    shutil.rmtree(component)
                elif component.name.startswith("version_"):
                    shutil.rmtree(component)

        log_execution_env_state()

        logging.info("Configuration: ")
        logging.info(OmegaConf.to_yaml(self.config))
        logging.info("Current working directory %s", os.getcwd())

        checkpoint_callback = instantiate(self.config.checkpoint)
        lit_module = instantiate(
            self.config.module,
            dataset=self.config.dataset,
            model=self.config.model,
            optimizer=self.config.optimizer,
            features=self.config.features,
            scheduler=self.config.get("scheduler", None),
            normalizer=self.config.get("normalizer", None),
        )
        callbacks = []

        logger = [
            TensorBoardLogger(".", version=None, name="", default_hp_metric=False),
            CSVLogger(".", version=None, name=""),
        ]

        if self.config.get("backend", None):
            backend = instantiate(self.config.backend)
            callbacks.append(backend)

        logging.info("type: '%s'", self.config.type)

        logging.info("Starting training")

        profiler = None
        if self.config.get("profiler", None):
            profiler = instantiate(self.config.profiler)

        lr_monitor = LearningRateMonitor()
        callbacks.append(lr_monitor)

        if self.config.get("gpu_stats", None):
            gpu_stats = GPUStatsMonitor()
            callbacks.append(gpu_stats)

        mac_summary_callback = MacSummaryCallback()
        callbacks.append(mac_summary_callback)

        opt_monitor = self.config.get("monitor", ["val_error"])
        opt_callback = HydraOptCallback(monitor=opt_monitor)
        callbacks.append(opt_callback)

        if self.config.get("early_stopping", None):
            stop_callback = instantiate(self.config.early_stopping)
            callbacks.append(stop_callback)

        if self.config.get("pruning", None):
            pruning_scheduler = PruningAmountScheduler(
                self.config.pruning.amount, self.config.trainer.max_epochs
            )
            pruning_config = dict(self.config.pruning)
            del pruning_config["amount"]
            pruning_callback = instantiate(pruning_config,
                                           amount=pruning_scheduler)
            callbacks.append(pruning_callback)

        results = self.training_core(lit_module, profiler,
                                  callbacks, checkpoint_callback, opt_callback,
                                  logger)
        
        print(results)

        return results

    @abc.abstractmethod
    def training_core(self, lit_module, profiler,
                      callbacks, checkpoint_callback, opt_callback,
                      logger):
        pass

    @abc.abstractmethod
    def eval(self, model_name):
        pass
