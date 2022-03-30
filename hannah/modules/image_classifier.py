import logging

import torch
import torch.nn.functional as F
import torch.utils.data as data
from hydra.utils import get_class, instantiate
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import accuracy, f1_score, precision, recall

from ..utils import set_deterministic
from .base import ClassifierModule

logger = logging.getLogger(__name__)


class ImageClassifierModule(ClassifierModule):
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

        example_data = self._decode_batch(self.test_set[0])["data"]

        if not isinstance(example_data, torch.Tensor):
            example_data = torch.tensor(example_data, device=self.device)

        self.example_input_array = example_data.clone().detach().unsqueeze(0)
        self.example_feature_array = example_data.clone().detach().unsqueeze(0)

        self.num_classes = 0
        if self.train_set.class_names:
            self.num_classes = len(self.train_set.class_names)

        logger.info("Setting up model %s", self.hparams.model.name)

        self.model = instantiate(
            self.hparams.model,
            input_shape=self.example_input_array.shape,
            labels=self.num_classes,
            _recursive_=False,
        )

        if self.num_classes > 0:
            self.test_confusion = ConfusionMatrix(num_classes=self.num_classes)

    def _decode_batch(self, batch):
        if isinstance(batch, list) or isinstance(batch, tuple):
            assert len(batch) == 2
            ret = {"data": batch[0], "labels": batch[1]}
        else:
            ret = batch

        return ret

    def get_class_names(self):
        return self.train_set.class_names

    def prepare_data(self):
        # get all the necessary data stuff
        if not self.train_set or not self.test_set or not self.dev_set:
            get_class(self.hparams.dataset.cls).prepare(self.hparams.dataset)

    def forward(self, x):
        return self.model(x)

    def common_step(self, step_name, batch, batch_idx):
        batch = self._decode_batch(batch)

        x = batch["data"]
        labels = batch.get("labels", None)

        if batch_idx == 0:
            loggers = self._logger_iterator()

            for logger in loggers:
                if hasattr(logger.experiment, "add_image"):
                    import torchvision.utils

                    images = torchvision.utils.make_grid(x, normalize=True)
                    logger.experiment.add_image(f"input{batch_idx}", images)

        prediction_result = self.forward(x)

        loss = torch.tensor([0.0], device=self.device)
        if labels is not None and "logits" in prediction_result:
            logits = prediction_result["logits"]
            classifier_loss = F.cross_entropy(logits, labels.squeeze())
            self.log(f"{step_name}_classifier_loss", classifier_loss)
            loss += classifier_loss

            preds = torch.argmax(logits, dim=1)
            acc = accuracy(preds, labels)

            precision_micro = precision(preds, labels)
            recall_micro = recall(preds, labels)
            f1_micro = f1_score(preds, labels)
            precision_macro = precision(
                preds, labels, num_classes=self.num_classes, average="macro"
            )
            recall_macro = recall(
                preds, labels, num_classes=self.num_classes, average="macro"
            )
            f1_macro = f1_score(
                preds, labels, num_classes=self.num_classes, average="macro"
            )
            self.log(f"{step_name}_error", 1 - acc, sync_dist=True)
            self.log(f"{step_name}_accuracy", acc, sync_dist=True)
            self.log(f"{step_name}_precision_micro", precision_micro, sync_dist=True)
            self.log(f"{step_name}_recall_micro", recall_micro, sync_dist=True)
            self.log(f"{step_name}_f1_micro", f1_micro, sync_dist=True)
            self.log(f"{step_name}_precision_macro", precision_macro, sync_dist=True)
            self.log(f"{step_name}_recall_macro", recall_macro, sync_dist=True)
            self.log(f"{step_name}_f1_macro", f1_macro, sync_dist=True)

        if "decoded" in prediction_result:
            decoded = prediction_result["decoded"]
            decoder_loss = F.mse_loss(decoded, x)
            self.log(f"{step_name}_decoder_loss", decoder_loss)
            loss += decoder_loss

        self.log(f"{step_name}_loss", loss)
        return loss, prediction_result, batch

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.common_step("train", batch, batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        self.common_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        _, step_results, batch = self.common_step("test", batch, batch_idx)

        y = batch.get("labels", None)
        preds = step_results.get("preds", None)
        if y is not None and preds is not None:
            with set_deterministic(False):
                self.test_confusion(preds, y)

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
