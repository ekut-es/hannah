import logging
import math
import random

from copy import deepcopy
from typing import Optional, Union, List

import torch.nn as nn
import torch

from omegaconf import DictConfig
from pytorch_lightning.metrics import Accuracy

from .classifier import StreamClassifierModule


class SpeechKDClassifierModule(StreamClassifierModule):
    def __init__(
        self,
        dataset: DictConfig,
        model: DictConfig,  # student model
        optimizer: DictConfig,
        features: DictConfig,
        num_workers: int = 0,
        batch_size: int = 128,
        time_masking: int = 0,
        frequency_masking: int = 0,
        scheduler: Optional[DictConfig] = None,
        normalizer: Optional[DictConfig] = None,
        teacher_checkpoint: Union[str, List[str], None] = None,
        freeze_teachers: bool = True,
        distillation_loss: str = "DGKD",
        temp: float = 10.0,
        distil_weight: float = 0.5,
        alpha: float = 0.5,
        noise_variance: float = 0.1,
        correct_prob: float = 0.9,
    ):
        super().__init__(
            dataset=dataset,
            model=model,
            optimizer=optimizer,
            features=features,
            num_workers=num_workers,
            batch_size=batch_size,
            scheduler=scheduler,
            normalizer=normalizer,
            time_masking=time_masking,
            frequency_masking=frequency_masking,
        )
        self.save_hyperparameters()
        self.distillation_loss = distillation_loss
        self.temp = temp
        self.distil_weight = distil_weight
        self.alpha = alpha
        self.noise_variance = noise_variance
        self.correct_prob = correct_prob
        self.teacher_loaded = False

        if distillation_loss == "TFself":
            self.loss_func = self.teacher_free_selfkd_loss
        elif distillation_loss == "TFself_Loss":
            self.loss_func = self.teacher_free_selfkd_loss
        elif distillation_loss == "TFVirtual":
            self.loss_func = self.teacher_free_virtual_loss
        elif distillation_loss == "noisyTeacher":
            self.loss_func = self.noisyTeacher_loss
        elif distillation_loss == "DGKD":
            self.loss_func = self.densely_guided_kd
        else:
            raise Exception("Distillation loss %s unknown", distillation_loss)

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
        super().prepare_data()
        if not self.teacher_loaded and self.distillation_loss == "TFself":
            # TODO Multiple Teacher could maybe done here
            params = deepcopy(self.hparams)
            params.pop("teacher_checkpoint")
            params.pop("freeze_teachers")
            params.pop("distillation_loss")
            params.pop("temp")
            params.pop("distil_weight")
            params.pop("alpha")
            params.pop("noise_variance")
            params.pop("correct_prob")
            teacher_module = StreamClassifierModule(**params)
            teacher_module.trainer = deepcopy(self.trainer)
            teacher_module.model = deepcopy(self.model)
            teacher_module.setup("fit")
            teacher_module.trainer.fit(teacher_module)
            teacher_module.trainer.test(ckpt_path=None)
            self.teachers.append(teacher_module)

        elif not self.teacher_loaded:
            for checkpoint_file in self.teacher_checkpoints:
                checkpoint = torch.load(
                    checkpoint_file, map_location=torch.device("cpu")
                )
                hparams = checkpoint["hyper_parameters"]

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
                # Train the self part of the algorithm

                if self.freeze_teachers:
                    for param in teacher_module.parameters():
                        param.requires_grad = False

                self.teachers.append(teacher_module)

        self.teacher_accuracies = nn.ModuleList([Accuracy() for t in self.teachers])
        self.teacher_loaded = True

    """
    Code taken from Paper: "KD-Lib: A PyTorch library for Knowledge Distillation, Pruning and Quantization"
    arxiv: 2011.14691
    License: MIT

    Original idea coverd in: "Revisit Knowledge Distillation: a Teacher-free Framework"
    arxiv: 1909.11723
    """

    def teacher_free_virtual_loss(self, y_pred_student, y_true):
        local_loss = nn.KLDivLoss()
        num_classes = y_pred_student.shape[1]

        soft_label = torch.ones_like(y_pred_student)
        soft_label = soft_label * (1 - self.correct_prob) / (num_classes - 1)

        for i in range(y_pred_student.shape[0]):
            soft_label[i, y_true[i]] = self.correct_prob

        loss = (1 - self.distil_weight) * torch.nn.functional.cross_entropy(
            y_pred_student, y_true
        )
        loss += (self.distil_weight) * local_loss(
            torch.nn.functional.log_softmax(y_pred_student, dim=1),
            torch.nn.functional.softmax(soft_label / self.temp, dim=1),
        )
        return loss

    """
    Code taken from Paper: "KD-Lib: A PyTorch library for Knowledge Distillation, Pruning and Quantization"
    arxiv: 2011.14691
    License: MIT

    Original idea coverd in: "Revisit Knowledge Distillation: a Teacher-free Framework"
    arxiv: 1909.11723
    """

    def teacher_free_selfkd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """
        local_loss = nn.KLDivLoss()
        loss = (1 - self.distil_weight) * torch.nn.functional.cross_entropy(
            y_pred_student, y_true
        )
        loss += (self.distil_weight) * local_loss(
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

    def noisyTeacher_loss(self, y_pred_student=None, y_pred_teacher=None, y_true=None):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """
        local_loss = nn.MSELoss()

        if random.uniform(0, 1) <= self.alpha:
            y_pred_teacher = self.add_noise(y_pred_teacher, self.noise_variance)

        loss = (1.0 - self.distil_weight) * torch.nn.functional.cross_entropy(
            y_pred_student, y_true
        )

        teacher_loss = local_loss(
            torch.nn.functional.log_softmax(y_pred_student / self.temp, dim=1),
            torch.nn.functional.softmax(y_pred_teacher / self.temp, dim=1),
        )

        loss += (self.distil_weight * self.temp * self.temp) * teacher_loss

        return loss

    """
    Code taken from Paper: "KD-Lib: A PyTorch library for Knowledge Distillation, Pruning and Quantization"
    arxiv: 2011.14691
    License: MIT

    Original idea coverd in: "Deep Model Compression: Distilling Knowledge from Noisy Teachers"
    arxiv: 1610.09650
    """

    def add_noise(self, x, variance=0.1):
        """
        Function for adding gaussian noise

        :param x (torch.FloatTensor): Input for adding noise
        :param variance (float): Variance for adding noise
        """

        return x * (1 + (variance ** 0.5) * torch.randn_like(x))

    """
        Code taken from Paper: "MEAL V2: Boosting Vanilla ResNet-50 to 80%+ Top-1 Accuracy on ImageNet without Tricks"
        arxiv: https://arxiv.org/abs/2009.08453
    """

    """
        Original idea coverd in: "Densely Guided Knowledge Distillation using Multiple Teacher Assistants"
        arxiv: 2009.08825
    """
    # densely guided knowledge distillation using multiple teachers
    def densely_guided_kd(self, student_logits, teacher_logits, y):

        # setup
        assert len(teacher_logits) >= 2  # at least one teacher and one assistant

        n = len(teacher_logits[1:])

        softmax = torch.nn.Softmax(dim=1)  # adds to one along dim 1
        cross_entropy = torch.nn.CrossEntropyLoss()
        kl_div = torch.nn.KLDivLoss(
            reduction="batchmean"
        )  # TODO batchmean removes warning but unsure whether good desicion

        # specific loss:

        l_ce_s = cross_entropy(student_logits, y)

        temp = self.temp
        student_logits_scaled = student_logits / temp
        teacher_logits_scaled = teacher_logits[0] / temp
        y_hat_s = softmax(student_logits_scaled.squeeze(1))
        y_hat_t = softmax(teacher_logits_scaled.squeeze(1))

        kl_div_t_s = (temp ** 2) * kl_div(y_hat_t, y_hat_s)

        # removing teacher logits
        assi_logits = teacher_logits[1:]

        # pop del_n random elements from assistant logits
        # works as kind of regularizer (not mandatory)
        del_n = self.alpha
        assert isinstance(del_n, int)
        if del_n >= n:
            del_n = n - 1
        if del_n > 0:
            for k in range(del_n):
                assi_logits.pop(random.randrange(len(assi_logits)))

        # TODO is there a smarter (numpy) way to sum?
        sum_kl_div_assis_s = 0
        for logits in assi_logits:
            y_hat_assi = softmax(logits)
            sum_kl_div_assis_s += (temp ** 2) * kl_div(y_hat_assi, y_hat_s)

        # balancing cross entropy of student and Kullback-Leibler div
        lam = self.distil_weight
        # equation (7) in paper
        loss = (n + 1) * (1 - lam) * l_ce_s + lam * (kl_div_t_s + sum_kl_div_assis_s)

        return loss

    def calculate_loss(self, student_logits, teacher_logits, y):
        loss = math.inf
        if self.distillation_loss == "TFVirtual":
            loss = self.loss_func(student_logits, y)
        elif self.distillation_loss == "DGKD":
            loss = self.loss_func(student_logits, teacher_logits, y)
        else:
            loss = self.loss_func(student_logits, teacher_logits[0], y)
        return loss

    def training_step(self, batch, batch_idx):
        # x inputs, y labels
        x, x_len, y, y_len = batch

        student_logits = self.forward(x)
        teacher_logits = []
        for teacher in self.teachers:
            teacher.eval()
            with torch.no_grad():
                teacher_logits.append(teacher(x))

        y = y.view(-1)

        assert len(teacher_logits) >= 1
        loss = self.calculate_loss(student_logits, teacher_logits, y)

        # METRICS
        self.calculate_batch_metrics(
            student_logits, y, loss, self.train_metrics, "train"
        )

        num = 0
        for teacher_output, acc in zip(teacher_logits, self.teacher_accuracies):
            acc(torch.nn.functional.softmax(teacher_output, dim=1), y)
            self.log(f"teacher_acc/{num}", acc, on_epoch=True, on_step=False)
            num += 1
        return loss
