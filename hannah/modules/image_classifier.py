import logging

import torch
import torch.nn.functional as F
import torch.utils.data as data

from torchmetrics.functional import accuracy
from hydra.utils import instantiate, get_class


from .base import ClassifierModule

logger = logging.getLogger(__name__)


class ImageClassifierModule(ClassifierModule):
    def setup(self, stage):
        if self.logger:
            self.logger.log_hyperparams(self.hparams)

        if self.initialized:
            return
        
        self.initialized = True

        dataset_cls = get_class(
            self.hparams.dataset.cls
        )
        self.train_set, self.dev_set, self.test_set = dataset_cls.splits(self.hparams.dataset)
        self.example_input_array = self.test_set[0][0].unsqueeze(0)
        self.example_feature_array = self.test_set[0][0].unsqueeze(0)

        logger.info("Setting up model %s", self.hparams.model.name)
        self.model = instantiate(self.hparams.model, labels=len(self.train_set.class_names), input_shape=self.example_input_array.shape)

    def prepare_data(self):
        # get all the necessary data stuff
        if not self.train_set or not self.test_set or not self.dev_set:
            get_class(self.hparams.dataset.cls).prepare(self.hparams.dataset)

    def forward(self, x):
        return self.model(x)

    def _get_dataloader(self, dataset):
        batch_size = self.hparams["batch_size"]
        dataset_conf = self.hparams.dataset
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

        loss = F.cross_entropy(logits, y.squeeze())
        self.log("train_loss", loss)

        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y.squeeze())
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y.squeeze())

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=False)
            self.log(f"{stage}_accuracy", acc, prog_bar=False)
            self.log(f"{stage}_error", 1.0 - acc, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def train_dataloader(self):
        return self._get_dataloader(self.train_set)

    def test_dataloader(self):
        return self._get_dataloader(self.test_set)

    def val_dataloader(self):
        return self._get_dataloader(self.dev_set)

    def on_train_epoch_end(self):
        self.eval()
        self._log_weight_distribution()
        self.train()

    def on_test_start(self):
        from ..visualization import log_distribution_install_hooks

        print("install_hooks")
        self.distribution_hooks = log_distribution_install_hooks(self)

    def on_test_end(self):
        from ..visualization import log_distribution_plot

        # log_distribution_plot(self.distribution_hooks)
