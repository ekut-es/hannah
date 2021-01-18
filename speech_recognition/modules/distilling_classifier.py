from .classifier import SpeechClassifierModule
from omegaconf import DictConfig
from typing import Optional
from .config_utils import get_model
import torch.nn as nn


class SpeechKDClassifierModule(SpeechClassifierModule):
    def __init__(
        self,
        dataset: DictConfig,
        # TODO how to pass teacher model?
        model: DictConfig,  # student model
        teacher_model: DictConfig,
        optimizer: DictConfig,
        features: DictConfig,
        num_workers: int = 0,
        batch_size: int = 128,
        scheduler: Optional[DictConfig] = None,
        normalizer: Optional[DictConfig] = None,
    ):
        super().__init__(
            dataset,
            model,
            optimizer,
            features,
            num_workers,
            batch_size,
            scheduler,
            normalizer,
        )

        # TODO which loss?
        self.mse_loss = nn.MSELoss()
        self.teacher_model = get_model(self.hparams.teacher_model)

        # no training for teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        # x inputs, y labels
        x, x_len, y, y_len = batch

        student_logits = self.forward(x)
        teacher_logits = self.teacher_model.forward(x)
        y = y.view(-1)

        loss = self.mse_loss(student_logits, teacher_logits)

        # --- after loss
        for callback in self.trainer.callbacks:
            if hasattr(callback, "on_before_backward"):
                callback.on_before_backward(self.trainer, self, loss)
        # --- before backward

        # METRICS
        self.get_batch_metrics(student_logits, y, loss, "train")

        return loss
