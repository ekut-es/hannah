import logging
from typing import Sequence

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.utils
from hydra.utils import get_class, instantiate
from timm.data.mixup import Mixup
from torchmetrics import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    MetricCollection,
    Precision,
    Recall,
)

from ..utils import set_deterministic
from .augmentation.batch_augmentation import BatchAugmentationPipeline
from .base import ClassifierModule
from .metrics import Error

msglogger = logging.getLogger(__name__)


class ImageClassifierModule(ClassifierModule):
    def setup(self, stage):
        if self.trainer:
            for logger in self.trainer.loggers:
                logger.log_hyperparams(self.hparams)

        if self.initialized:
            return

        self.initialized = True

        dataset_cls = get_class(self.hparams.dataset.cls)
        self.train_set, self.dev_set, self.test_set = dataset_cls.splits(
            self.hparams.dataset
        )

        self.train_set_unlabeled, self.dev_set_unlabeled, self.test_set_unlabeled = (
            None,
            None,
            None,
        )

        example_data = self._decode_batch(self.test_set[0])["data"]

        if not isinstance(example_data, torch.Tensor):
            example_data = torch.tensor(example_data, device=self.device)

        self.example_input_array = example_data.clone().detach().unsqueeze(0)
        self.example_feature_array = example_data.clone().detach().unsqueeze(0)

        self.num_classes = 0
        if self.train_set.class_names:
            self.num_classes = len(self.train_set.class_names)

        msglogger.info("Setting up model %s", self.hparams.model.name)
        self.model = instantiate(
            self.hparams.model,
            input_shape=self.example_input_array.shape,
            labels=self.num_classes,
            _recursive_=False,
        )

        if self.hparams.dataset.get("weighted_loss", False) is True:
            loss_weights = torch.tensor(self.train_set.weights)
            loss_weights *= len(self.train_set) / self.num_classes

            msglogger.info("Using weighted loss with weights:")
            for num, (weight, name) in enumerate(
                zip(loss_weights, self.train_set.class_names)
            ):
                msglogger.info("- %s [%d]: %f", name, num, weight.item())

            self.register_buffer("loss_weights", loss_weights)
        else:
            self.loss_weights = None

        self.mixup_fn = self.train_set.get_mixup_fn()
        self.batch_augment_fn = self.train_set.get_batch_augment_fn()

        # Setup Metrics
        metrics = {}
        if self.num_classes > 0:
            self.test_confusion = ConfusionMatrix(num_classes=self.num_classes)

            for step_name in ["train", "val", "test"]:
                step_metrics = MetricCollection(
                    {
                        f"{step_name}_accuracy": Accuracy(num_classes=self.num_classes),
                        f"{step_name}_error": Error(num_classes=self.num_classes),
                        f"{step_name}_precision_micro": Precision(
                            num_classes=self.num_classes, average="micro"
                        ),
                        f"{step_name}_recall_micro": Recall(
                            num_classes=self.num_classes, average="micro"
                        ),
                        f"{step_name}_f1_micro": F1Score(
                            num_classes=self.num_classes, average="micro"
                        ),
                        f"{step_name}_precision_macro": Precision(
                            num_classes=self.num_classes, average="macro"
                        ),
                        f"{step_name}_recall_macro": Recall(
                            num_classes=self.num_classes, average="macro"
                        ),
                        f"{step_name}_f1_macro": F1Score(
                            num_classes=self.num_classes, average="macro"
                        ),
                    }
                )
                metrics[f"{step_name}_metrics"] = step_metrics

        self.metrics = torch.nn.ModuleDict(metrics)

    def _decode_batch(self, batch):
        if isinstance(batch, Sequence):
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
        # print("step_name", step_name)
        batch = self._decode_batch(batch)

        breakpoint()

        x = batch["data"]
        labels = batch.get("labels", None)
        boxes = batch.get("bbox", None)

        if batch_idx == 0:
            self._log_batch_images("input", batch_idx, x)

        mixup_labels = labels
        augmented_data = x
        if step_name == "train":
            if labels is not None:
                if self.mixup_fn is not None:
                    x, mixup_labels = self.mixup_fn(x, labels)

            if self.batch_augment_fn is not None:
                augmented_data = self.batch_augment_fn(data=x)

        if batch_idx == 0:
            self._log_batch_images("augmented", batch_idx, augmented_data)

        prediction_result = self.forward(augmented_data)

        loss = torch.tensor([0.0], device=self.device)
        if labels is not None and prediction_result.logits is not None:
            logits = prediction_result.logits

            classifier_loss = F.cross_entropy(
                logits, mixup_labels.squeeze(), weight=self.loss_weights
            )

            self.log(f"{step_name}_classifier_loss", classifier_loss)
            loss += classifier_loss

            preds = torch.argmax(logits, dim=1)
            self.metrics[f"{step_name}_metrics"](preds, labels)

            self.log_dict(self.metrics[f"{step_name}_metrics"])

        if prediction_result.decoded is not None:
            decoded = prediction_result.decoded
            decoder_loss = F.mse_loss(decoded, x)
            # print(f"{step_name}_decoder_loss", decoder_loss)
            self.log(f"{step_name}_decoder_loss", decoder_loss)
            loss += decoder_loss

            if batch_idx == 0:
                self._log_batch_images("decoded", batch_idx, decoded)

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

    def on_train_epoch_end(self):
        self.eval()
        self._log_weight_distribution()
        self.train()
