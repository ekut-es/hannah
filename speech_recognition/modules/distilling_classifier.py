from .classifier import SpeechClassifierModule
from omegaconf import DictConfig
from typing import Optional
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Union


class SpeechKDClassifierModule(SpeechClassifierModule):
    def __init__(
        self,
        dataset: DictConfig,
        model: DictConfig,  # student model
        optimizer: DictConfig,
        features: DictConfig,
        num_workers: int = 0,
        batch_size: int = 128,
        scheduler: Optional[DictConfig] = None,
        normalizer: Optional[DictConfig] = None,
        # TODO how to pass pre trained teacher model?
        teacher_model: DictConfig = None,
        teacher_checkpoint: str = None,
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
            teacher_checkpoint,
        )
        loss_config = "MSE"
        # TODO which loss?
        if loss_config == "MSE":
            self.loss_func = nn.MSELoss()
        else:
            self.loss_func = None

        print(f"!!! teacher model is {teacher_model} with type {type(teacher_model)}")

    def training_step(self, batch, batch_idx):
        # x inputs, y labels
        x, x_len, y, y_len = batch

        student_logits = super().forward(x)
        teacher_logits = self.forward(x)
        y = y.view(-1)

        loss = self.loss_func(student_logits, teacher_logits)

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
