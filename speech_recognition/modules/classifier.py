import os
import shutil
import random
import logging
import numpy as np
import sys

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.functional import accuracy, f1, recall
from pytorch_lightning.loggers import TensorBoardLogger
from .config_utils import get_loss_function, get_model, save_model
from typing import Optional

from speech_recognition.datasets.dataset import ctc_collate_fn

import torch
import torch.utils.data as data
from hydra.utils import instantiate, get_class

from torchvision.datasets.utils import (
    download_and_extract_archive,
    extract_archive,
    list_files,
    list_dir,
)

from ..datasets.NoiseDataset import NoiseDataset
from ..datasets.DatasetSplit import DatasetSplit
from ..datasets.Downsample import Downsample
import torchaudio

from omegaconf import DictConfig


class StreamClassifierModule(LightningModule):
    def __init__(
        self,
        dataset: DictConfig,
        model: DictConfig,
        optimizer: DictConfig,
        features: DictConfig,
        num_workers: int = 0,
        batch_size: int = 128,
        scheduler: Optional[DictConfig] = None,
        normalizer: Optional[DictConfig] = None,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.msglogger = logging.getLogger()
        self.initialized = False
        self.train_set = None
        self.test_set = None
        self.dev_set = None

    def prepare_data(self):
        # get all the necessary data stuff
        if not self.train_set or not self.test_set or not self.dev_set:
            get_class(self.hparams.dataset.cls).download(self.hparams.dataset)
            NoiseDataset.download_noise(self.hparams.dataset)
            DatasetSplit.split_data(self.hparams.dataset)
            Downsample.downsample(self.hparams.dataset)

    def setup(self, stage):

        self.msglogger.info("Setting up model")

        if self.initialized:
            return

        self.initialized = True

        # trainset needed to set values in hparams
        self.train_set, self.dev_set, self.test_set = get_class(
            self.hparams.dataset.cls
        ).splits(self.hparams.dataset)

        # Create example input
        device = (
            self.trainer.root_gpu if self.trainer.root_gpu is not None else self.device
        )
        self.example_input_array = torch.zeros(
            1, self.train_set.channels, self.train_set.input_length
        )
        dummy_input = self.example_input_array.to(device)

        # Instantiate features
        self.features = instantiate(self.hparams.features)
        self.features.to(device)

        features = self._extract_features(dummy_input)
        self.example_feature_array = features

        # Instantiate normalizer
        if self.hparams.normalizer is not None:
            self.normalizer = instantiate(self.hparams.normalizer)
        else:
            self.normalizer = torch.nn.Identity()

        # Instantiate Model
        self.hparams.model.width = self.example_feature_array.size(2)
        self.hparams.model.height = self.example_feature_array.size(1)
        self.num_classes = len(self.train_set.label_names)
        self.hparams.model.n_labels = self.num_classes

        self.model = get_model(self.hparams.model)

        # loss function
        self.criterion = get_loss_function(self.model, self.hparams)

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        schedulers = []

        if self.hparams.scheduler is not None:
            scheduler = instantiate(self.hparams.scheduler, optimizer=optimizer)
            schedulers.append(scheduler)

        return [optimizer], schedulers

    def get_batch_metrics(self, output, y, loss, prefix):

        # in case of multiple outputs
        if isinstance(output, list):
            # log for each output
            for idx, out in enumerate(output):
                acc = accuracy(out, y)
                self.log(f"{prefix}_accuracy/exit_{idx}", acc)
                self.log(f"{prefix}_error/exit_{idx}", 1.0 - acc)
                self.log(f"{prefix}_recall/exit_{idx}", recall(out, y))
                self.log(f"{prefix}_f1/exit_{idx}", f1(out, y, self.num_classes))

        else:
            acc = accuracy(output, y)
            self.log(f"{prefix}_accuracy", acc)
            self.log(f"{prefix}_error", 1.0 - acc)
            self.log(f"{prefix}_f1", f1(output, y, self.num_classes))
            self.log(f"{prefix}_recall", recall(output, y))

        # only one loss allowed
        # also in case of branched networks
        self.log(f"{prefix}_loss", loss)

    @staticmethod
    def get_balancing_sampler(dataset):
        distribution = dataset.get_categories_distribution()
        weights = 1.0 / torch.tensor(
            [distribution[i] for i in range(len(distribution))],
            dtype=torch.float
        )

        sampler_weights = weights[dataset.get_label_list()]

        sampler = data.WeightedRandomSampler(sampler_weights, len(dataset))
        return sampler

    # TRAINING CODE
    def training_step(self, batch, batch_idx):
        x, x_len, y, y_len = batch

        output = self(x)
        y = y.view(-1)
        loss = self.criterion(output, y)

        # --- after loss
        for callback in self.trainer.callbacks:
            if hasattr(callback, "on_before_backward"):
                callback.on_before_backward(self.trainer, self, loss)
        # --- before backward

        # METRICS
        self.get_batch_metrics(output, y, loss, "train")

        return loss

    def train_dataloader(self):
        train_batch_size = self.hparams["batch_size"]
        dataset_conf = self.hparams.dataset
        sampler = None
        if "balance_train_set_by_sampler" in dataset_conf.keys() and dataset_conf["balance_train_set_by_sampler"]:
            sampler = self.get_balancing_sampler(self.train_set)
        train_loader = data.DataLoader(
            self.train_set,
            batch_size=train_batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=self.hparams["num_workers"],
            collate_fn=ctc_collate_fn,
            sampler=sampler,
            multiprocessing_context='fork' if self.hparams['num_workers'] > 0 else None, 
        )

        self.batches_per_epoch = len(train_loader)

        return train_loader

    # VALIDATION CODE
    def validation_step(self, batch, batch_idx):

        # dataloader provides these four entries per batch
        x, x_length, y, y_length = batch

        # INFERENCE
        output = self(x)
        y = y.view(-1)
        loss = self.criterion(output, y)

        # METRICS
        self.get_batch_metrics(output, y, loss, "val")
        return loss

    def val_dataloader(self):

        dev_loader = data.DataLoader(
            self.dev_set,
            batch_size=min(len(self.dev_set), 16),
            shuffle=False,
            num_workers=self.hparams["num_workers"],
            collate_fn=ctc_collate_fn,
            multiprocessing_context='fork' if self.hparams['num_workers'] > 0 else None, 
        )

        return dev_loader

    # TEST CODE
    def test_step(self, batch, batch_idx):

        # dataloader provides these four entries per batch
        x, x_length, y, y_length = batch

        output = self(x)
        y = y.view(-1)
        loss = self.criterion(output, y)

        # METRICS
        self.get_batch_metrics(output, y, loss, "test")

        return loss

    def test_dataloader(self):

        test_loader = data.DataLoader(
            self.test_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams["num_workers"],
            collate_fn=ctc_collate_fn,
            multiprocessing_context='fork' if self.hparams['num_workers'] > 0 else None, 
        )

        return test_loader

    def _extract_features(self, x):
        x = self.features(x)

        if x.dim() == 4:
            new_channels = x.size(1) * x.size(2)
            x = torch.reshape(x, (x.size(0), new_channels, x.size(3)))

        return x

    def forward(self, x):
        x = self._extract_features(x)
        x = self.normalizer(x)
        return self.model(x)

    # CALLBACKS
    def on_train_end(self):
        # TODO currently custom save, in future proper configure lighting for saving ckpt
        save_model(".", self)

    def on_fit_end(self):
        for logger in self.trainer.logger:
            if isinstance(logger, TensorBoardLogger):
                logger.log_hyperparams(
                    self.hparams,
                    metrics={
                        "val_loss": self.trainer.callback_metrics["val_loss"],
                        "val_accuracy": self.trainer.callback_metrics["val_accuracy"],
                    },
                )
