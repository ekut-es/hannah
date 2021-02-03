import logging
from .classifier import StreamClassifierModule
from omegaconf import DictConfig
from typing import Optional
import torch.nn as nn
import torch
import random
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Union, List


class SpeechKDClassifierModule(StreamClassifierModule):
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
        teacher_checkpoint: Union[str, List[str], None] = None,
        freeze_teachers: bool = True,
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
        self.save_hyperparameters()

        # TODO Loss configuration dynamic
        loss_config = "MSE"
        if loss_config == "MSE":
            self.loss_func = nn.MSELoss()
        elif loss_config == "TFself":
            self.loss_func = self.teacher_free_selfkd_loss()
        elif loss_config == "TFFramework":
            self.loss_func = self.teacher_free_framework_loss()
        elif loss_config == "noisyTeacher":
            self.loss_func = self.noisyTeacher_loss()
        else:
            self.loss_func = nn.MSELoss()

        self.teachers = nn.ModuleList()

        if teacher_checkpoint is None:
            teacher_checkpoint = []
        if isinstance(teacher_checkpoint, str):
            teacher_checkpoint = [teacher_checkpoint]

        if teacher_checkpoint is []:
            # TODO: raise exception if this is not a teacher free distillation
            logging.warning("No teachers defined")

        self.teacher_checkpoints = teacher_checkpoint
        self.freeze_teachers = freeze_teachers

    def setup(self, stage):
        super().setup(stage)

        for checkpoint_file in self.teacher_checkpoints:
            checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))

            print(checkpoint.keys())

            hparams = checkpoint["hyper_parameters"]
            print(hparams)

            # TODO: train new teacher checkpoints and remove from model
            if "teacher_model" in hparams:
                del hparams["teacher_model"]
            if "teacher_checkpoint" in hparams:
                del hparams["teacher_checkpoint"]

            # Overwrite dataset
            hparams["dataset"] = self.hparams["dataset"]
            teacher_module = StreamClassifierModule(**hparams)
            teacher_module.trainer = self.trainer
            teacher_module.setup("fit")
            teacher_module.load_state_dict(checkpoint["state_dict"])

            if self.freeze_teachers:
                for param in teacher_module.parameters():
                    param.requires_grad = False

            self.teachers.append(teacher_module)

    """
    Code taken from Paper: "KD-Lib: A PyTorch library for Knowledge Distillation, Pruning and Quantization"
    arxiv: 2011.14691
    License: MIT

    Original idea coverd in: "Revisit Knowledge Distillation: a Teacher-free Framework"
    arxiv: 1909.11723
    """

    def teacher_free_framework_loss(
        self, y_pred_student, y_true, distil_weight=0.5, correct_prob=0.9, temp=10.0
    ):
        local_loss = nn.KLDivLoss()
        num_classes = y_pred_student.shape[1]

        soft_label = torch.ones_like(
            y_pred_student
        )  # .to(self.device) Brauchen wir das?
        soft_label = soft_label * (1 - correct_prob) / (num_classes - 1)

        for i in range(y_pred_student.shape[0]):
            soft_label[i, y_true[i]] = correct_prob

        loss = (1 - distil_weight) * torch.nn.functional.cross_entropy(
            y_pred_student, y_true
        )
        loss += (distil_weight) * local_loss(
            torch.nn.functional.log_softmax(y_pred_student, dim=1),
            torch.nn.functional.softmax(soft_label / temp, dim=1),
        )
        return loss

    """
    Code taken from Paper: "KD-Lib: A PyTorch library for Knowledge Distillation, Pruning and Quantization"
    arxiv: 2011.14691
    License: MIT

    Original idea coverd in: "Revisit Knowledge Distillation: a Teacher-free Framework"
    arxiv: 1909.11723
    """

    def teacher_free_selfkd_loss(
        self, y_pred_student, y_pred_teacher, y_true, distil_weight=0.5
    ):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """
        local_loss = nn.KLDivLoss()
        loss = (1 - distil_weight) * torch.nn.functional.cross_entropy(
            y_pred_student, y_true
        )
        loss += (distil_weight) * local_loss(
            torch.nn.functional.log_softmax(y_pred_student, dim=1),
            torch.nn.functional.softmax(y_pred_teacher / self.temp, dim=1),
        )
        return loss

    """
    Code taken from Paper: "KD-Lib: A PyTorch library for Knowledge Distillation, Pruning and Quantization"
    arxiv: 2011.14691
    License: MIT

    Original idea coverd in: "Deep Model Compression: Distilling Knowledge from Noisy Teachers"
    arxiv: 1610.09650
    """

    def noisyTeacher_loss(
        self,
        y_pred_student,
        y_pred_teacher,
        y_true,
        distil_weight=0.5,
        temp=20.0,
        alpha=0.5,
        noise_variance=0.1,
    ):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """
        local_loss = nn.MSELoss()

        if random.uniform(0, 1) <= alpha:
            y_pred_teacher = self.add_noise(y_pred_teacher, noise_variance)

        loss = (1.0 - distil_weight) * torch.nn.functional.cross_entropy(
            y_pred_student, y_true
        )

        loss += (distil_weight * temp * temp) * local_loss(
            torch.nn.functional.log_softmax(y_pred_student / temp, dim=1),
            torch.nn.functional.softmax(y_pred_teacher / temp, dim=1),
        )

        return loss

    """
    Code taken from Paper: "KD-Lib: A PyTorch library for Knowledge Distillation, Pruning and Quantization"
    arxiv: 2011.14691
    License: MIT

    Original idea coverd in: "Deep Model Compression: Distilling Knowledge from Noisy Teachers"
    arxiv: 1610.09650
    """

    def add_noise(x, variance=0.1):
        """
        Function for adding gaussian noise

        :param x (torch.FloatTensor): Input for adding noise
        :param variance (float): Variance for adding noise
        """

        return x * (1 + (variance ** 0.5) * torch.randn_like(x))

    def training_step(self, batch, batch_idx):
        # x inputs, y labels
        x, x_len, y, y_len = batch

        student_logits = self.forward(x)
        teacher_logits = []
        for teacher in self.teachers:
            teacher_logits.append(teacher(x))

        self.forward(x)
        y = y.view(-1)

        # TODO: handle multiple teachers
        assert len(teacher_logits) == 1
        loss = self.loss_func(student_logits, teacher_logits[0])

        # --- after loss
        for callback in self.trainer.callbacks:
            if hasattr(callback, "on_before_backward"):
                callback.on_before_backward(self.trainer, self, loss)
        # --- before backward

        # METRICS
        self.get_batch_metrics(student_logits, y, loss, "train")

        return loss
