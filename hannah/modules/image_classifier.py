import logging

import torch
import torch.nn.functional as F
import torch.utils.data as data

from torchmetrics.functional import accuracy
from hydra.utils import instantiate, get_class
from torchmetrics import ConfusionMatrix
from ..utils import set_deterministic

from .base import ClassifierModule

logger = logging.getLogger(__name__)


class ImageClassifierModule(ClassifierModule):
    def total_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps > 0:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return int((batches // effective_accum) * self.trainer.max_epochs)

    def setup(self, stage):
        if self.logger:
            self.logger.log_hyperparams(self.hparams)

        if self.initialized:
            return

        self.initialized = True

        dataset_cls = get_class(self.hparams.dataset.cls)
        self.train_set, self.dev_set, self.test_set = dataset_cls.splits(
            self.hparams.dataset
        )
        self.example_input_array = torch.tensor(self.test_set[0][0]).unsqueeze(0)
        self.example_feature_array = torch.tensor(self.test_set[0][0]).unsqueeze(0)

        self.num_classes = len(self.train_set.class_names)

        logger.info("Setting up model %s", self.hparams.model.name)
        self.model = instantiate(
            self.hparams.model,
            labels=self.num_classes,
            input_shape=self.example_input_array.shape,
        )

        self.test_confusion = ConfusionMatrix(num_classes=self.num_classes)

    def get_class_names(self):
        return self.train_set.class_names

    def prepare_data(self):
        # get all the necessary data stuff
        if not self.train_set or not self.test_set or not self.dev_set:
            get_class(self.hparams.dataset.cls).prepare(self.hparams.dataset)

    def forward(self, x):
        return self.model(x)

    def _get_dataloader(self, dataset, shuffle=False):
        batch_size = self.hparams["batch_size"]
        dataset_conf = self.hparams.dataset
        sampler = None
        if shuffle:
            sampler_type = dataset_conf.get("sampler", "random")
            if sampler_type == "weighted":
                sampler = self.get_balancing_sampler(dataset)
            else:
                sampler = data.RandomSampler(dataset)

        train_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=self.hparams["num_workers"],
            sampler=sampler,
            multiprocessing_context="fork" if self.hparams["num_workers"] > 0 else None,
        )

        self.batches_per_epoch = len(train_loader)

        return train_loader

    def training_step(self, batch, batch_idx):
        x, y = batch

        if batch_idx == 0:
            loggers = self._logger_iterator()

            for logger in loggers:
                if hasattr(logger.experiment, "add_image"):
                    import torchvision.utils

                    images = torchvision.utils.make_grid(x, normalize=True)
                    logger.experiment.add_image(f"input{batch_idx}", images)

        logits = self(x)
        loss = F.cross_entropy(
            logits,
            y.squeeze(),
            weight=torch.tensor(
                [
                    0.0285,
                    1.0000,
                    0.1068,
                    0.1667,
                    0.0373,
                    0.0196,
                    0.0982,
                    0.0014,
                    0.0235,
                    0.0236,
                    0.0809,
                ],
                device=self.device,
            ),
        )
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y.squeeze())
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y.squeeze())

        self.log("val_loss", loss)
        self.log("val_error", 1 - acc)
        self.log("val_accuracy", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y.squeeze())
        preds = torch.argmax(logits, dim=1)
        softmax = F.softmax(logits)
        acc = accuracy(preds, y.squeeze())

        with set_deterministic(False):
            self.test_confusion(preds, y)

        self.log("test_loss", loss)
        self.log("test_error", 1 - acc)
        self.log("test_accuracy", acc)

    def train_dataloader(self):
        return self._get_dataloader(self.train_set, shuffle=True)

    def test_dataloader(self):
        return self._get_dataloader(self.test_set)

    def val_dataloader(self):
        return self._get_dataloader(self.dev_set)

    def on_train_epoch_end(self):
        self.eval()
        self._log_weight_distribution()
        self.train()
