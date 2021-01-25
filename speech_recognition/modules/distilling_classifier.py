from .classifier import SpeechClassifierModule
from omegaconf import DictConfig
from typing import Optional
import torch.nn as nn


class SpeechKDClassifierModule(SpeechClassifierModule):
    def __init__(
        self,
        dataset: DictConfig,
        # TODO how to pass teacher model?
        model: DictConfig,  # student model
        optimizer: DictConfig,
        features: DictConfig,
        num_workers: int = 0,
        batch_size: int = 128,
        scheduler: Optional[DictConfig] = None,
        normalizer: Optional[DictConfig] = None,
        teacher_model: DictConfig = None,
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
            teacher_model,
        )

        # TODO which loss?
        self.mse_loss = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        # x inputs, y labels
        x, x_len, y, y_len = batch

        student_logits = super().forward(x)
        teacher_logits = self.forward(x)
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

    def forward(self, x):
        x = super()._extract_features(x)
        x = self.normalizer(x)
        return self.teacher_model(x)
