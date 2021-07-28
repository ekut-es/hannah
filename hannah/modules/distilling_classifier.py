import logging
import math
import random

from copy import deepcopy
from typing import Optional, Union, List

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import grad

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
        export_onnx: bool = True,
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
            export_onnx=export_onnx,
        )
        self.model = model
        self.save_hyperparameters()
        self.distillation_loss = distillation_loss
        self.temp = temp
        self.distil_weight = distil_weight
        self.alpha = alpha
        self.noise_variance = noise_variance
        self.correct_prob = correct_prob
        self.teacher_loaded = False

        calls = {
            "TFself": self.teacher_free_selfkd_loss,
            "TFself_Loss": self.teacher_free_selfkd_loss,
            "TFVirtual": self.teacher_free_virtual_loss,
            "noisyTeacher": self.noisyTeacher_loss,
            "DGKD": self.densely_guided_kd,
            "Sobolev": self.Sobolev,
            "DML": self.DML,
            "AT": self.AT,
            "RKD": self.RKD,
            "ST": self.ST,
            "PKT": self.PKT,
            "CC": self.CC,
            "Hint": self.Hint,
            "Logits": self.Logits,
            "LwM": self.LwM,
            "NST": self.NST,
            "SP": self.SP,
        }
        if self.distillation_loss in calls:
            self.loss_func = calls[self.distillation_loss]
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
        Sobolev Training for Neural Networks
        https://arxiv.org/pdf/1706.04859.pdf
        Knowledge Transfer with Jacobian Matching
        http://de.arxiv.org/pdf/1803.00443
    """

    def Sobolev(self, y_pred_student=None, y_pred_teacher=None, y_true=None, x=None):
        target_out_s = torch.gather(y_pred_student, 1, y_true.view(-1, 1))
        grad_s = grad(
            outputs=target_out_s,
            inputs=x,
            grad_outputs=torch.ones_like(target_out_s),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        norm_grad_s = F.normalize(grad_s.view(grad_s.size(0), -1), p=2, dim=1)
        target_out_t = torch.gather(y_pred_teacher, 1, y_true.view(-1, 1))
        grad_t = grad(
            outputs=target_out_t,
            inputs=x,
            grad_outputs=torch.ones_like(target_out_t),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        norm_grad_t = F.normalize(grad_t.view(grad_t.size(0), -1), p=2, dim=1)

        loss = F.mse_loss(norm_grad_s, norm_grad_t.detach())

        return loss

    """
    Deep Mutual Learning
    https://zpascal.net/cvpr2018/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf
    """

    def DML(self, out1, out2):
        softmax = torch.nn.Softmax(dim=1)
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = F.kl_div(logsoftmax(out1), softmax(out2), reduction="batchmean")

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
        logsoftmax = torch.nn.LogSoftmax(dim=1)  # adds to one along dim
        cross_entropy = torch.nn.CrossEntropyLoss()
        kl_div = torch.nn.KLDivLoss(
            reduction="batchmean"
        )  # batchmean fits kl div math definition

        # specific loss:

        l_ce_s = cross_entropy(student_logits, y)

        temp = self.temp
        student_logits_scaled = student_logits / temp
        teacher_logits_scaled = teacher_logits[0] / temp
        y_hat_s = softmax(student_logits_scaled.squeeze(1))
        y_hat_t = logsoftmax(teacher_logits_scaled.squeeze(1))

        kl_div_t_s = (temp ** 2) * kl_div(y_hat_t, y_hat_s)

        # removing teacher logits
        assi_logits = teacher_logits[1:]

        # pop del_n random elements from assistant logits
        # works as kind of regularizer (not mandatory)
        del_n = int(self.alpha)
        # assert isinstance(del_n, int)
        if del_n >= n:
            del_n = n - 1
        if del_n > 0:
            for k in range(del_n):
                assi_logits.pop(random.randrange(len(assi_logits)))

        # TODO is there a smarter (numpy) way to sum?
        sum_kl_div_assis_s = 0
        for logits in assi_logits:
            y_hat_assi = logsoftmax(logits)
            sum_kl_div_assis_s += (temp ** 2) * kl_div(y_hat_assi, y_hat_s)

        # balancing cross entropy of student and Kullback-Leibler div
        lam = self.distil_weight
        # equation (7) in paper
        loss = (n + 1) * (1 - lam) * l_ce_s + lam * (
            kl_div_t_s + sum_kl_div_assis_s / len(assi_logits)
        )

        return loss

    """
    Paying More Attention to Attention: Improving the Performance of Convolutional
    Neural Netkworks wia Attention Transfer
    https://arxiv.org/pdf/1612.03928.pdf
    """

    def AT(self, fm_s, fm_t, p=2.0):
        loss = F.mse_loss(self.attention_map(fm_s, p), self.attention_map(fm_t, p))

        return loss

    def attention_map(self, fm, p, eps=1e-6):
        am = torch.pow(torch.abs(fm), p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(-2, 1), keepdim=True)
        am = torch.div(am, norm + eps)

        return am

    """
    Relational Knowledge Distillation
    https://arxiv.org/pdf/1904.05068.pdf
    """

    def RKD(self, feat_s, feat_t, w_angle=50.0, w_dist=25.0):
        loss = w_dist * self.rkd_dist(feat_s, feat_t) + w_angle * self.rkd_angle(
            feat_s, feat_t
        )

        return loss

    def rkd_dist(self, feat_s, feat_t):
        feat_t_dist = self.pdist(feat_t, squared=False)
        mean_feat_t_dist = feat_t_dist[feat_t_dist > 0].mean()
        feat_t_dist = feat_t_dist / mean_feat_t_dist

        feat_s_dist = self.pdist(feat_s, squared=False)
        mean_feat_s_dist = feat_s_dist[feat_s_dist > 0].mean()
        feat_s_dist = feat_s_dist / mean_feat_s_dist

        loss = F.smooth_l1_loss(feat_s_dist, feat_t_dist)

        return loss

    def rkd_angle(self, feat_s, feat_t):
        # N x C --> N x N x C
        feat_t_vd = feat_t.unsqueeze(0) - feat_t.unsqueeze(1)
        norm_feat_t_vd = F.normalize(feat_t_vd, p=2, dim=2)
        feat_t_angle = torch.bmm(norm_feat_t_vd, norm_feat_t_vd.transpose(1, 2)).view(
            -1
        )

        feat_s_vd = feat_s.unsqueeze(0) - feat_s.unsqueeze(1)
        norm_feat_s_vd = F.normalize(feat_s_vd, p=2, dim=2)
        feat_s_angle = torch.bmm(norm_feat_s_vd, norm_feat_s_vd.transpose(1, 2)).view(
            -1
        )

        loss = F.smooth_l1_loss(feat_s_angle, feat_t_angle)

        return loss

    def pdist(self, feat, squared=False, eps=1e-12):
        feat_square = feat.pow(2).sum(dim=1)
        feat_prod = torch.mm(feat, feat.t())
        feat_dist = (
            feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod
        ).clamp(min=eps)

        if not squared:
            feat_dist = feat_dist.sqrt()

        feat_dist = feat_dist.clone()
        feat_dist[range(len(feat)), range(len(feat))] = 0

        return feat_dist

    """
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    """

    def ST(self, out_s, out_t, T=0.45):
        loss = (
            F.kl_div(
                F.log_softmax(out_s / T, dim=1),
                F.softmax(out_t / T, dim=1),
                reduction="batchmean",
            )
            * T
            * T
        )

        return loss

    """
    Learning Deep Representations with Probabilistic Knowledge Transfer
    http://openaccess.thecvf.com/content_ECCV_2018/papers/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.pdf
    """

    def PKT(self, feat_s, feat_t, eps=1e-6):
        # Normalize each vector by its norm
        feat_s_norm = torch.sqrt(torch.sum(feat_s ** 2, dim=1, keepdim=True))
        feat_s = feat_s / (feat_s_norm + eps)
        feat_s[feat_s != feat_s] = 0

        feat_t_norm = torch.sqrt(torch.sum(feat_t ** 2, dim=1, keepdim=True))
        feat_t = feat_t / (feat_t_norm + eps)
        feat_t[feat_t != feat_t] = 0

        # Calculate the cosine similarity
        feat_s_cos_sim = torch.mm(feat_s, feat_s.transpose(0, 1))
        feat_t_cos_sim = torch.mm(feat_t, feat_t.transpose(0, 1))

        # Scale cosine similarity to [0,1]
        feat_s_cos_sim = (feat_s_cos_sim + 1.0) / 2.0
        feat_t_cos_sim = (feat_t_cos_sim + 1.0) / 2.0

        # Transform them into probabilities
        feat_s_cond_prob = feat_s_cos_sim / torch.sum(
            feat_s_cos_sim, dim=1, keepdim=True
        )
        feat_t_cond_prob = feat_t_cos_sim / torch.sum(
            feat_t_cos_sim, dim=1, keepdim=True
        )

        # Calculate the KL-divergence
        loss = torch.mean(
            feat_t_cond_prob
            * torch.log((feat_t_cond_prob + eps) / (feat_s_cond_prob + eps))
        )

        return loss

    """
    Correlation Congruence for Knowledge Distillation
    http://openaccess.thecvf.com/content_ICCV_2019/papers/
    Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.pdf
    """

    def CC(self, feat_s, feat_t, gamma=0.4, P_order=2):
        corr_mat_s = self.get_correlation_matrix(feat_s, gamma=gamma, P_order=P_order)
        corr_mat_t = self.get_correlation_matrix(feat_t, gamma=gamma, P_order=P_order)

        loss = F.mse_loss(corr_mat_s, corr_mat_t)

        return loss

    def get_correlation_matrix(self, feat, gamma=0.4, P_order=2):
        feat = F.normalize(feat, p=2, dim=-1)
        sim_mat = torch.matmul(feat, feat.t())
        corr_mat = torch.zeros_like(sim_mat)

        for p in range(P_order + 1):
            corr_mat += (
                math.exp(-2 * gamma)
                * (2 * gamma) ** p
                / math.factorial(p)
                * torch.pow(sim_mat, p)
            )

        return corr_mat

    """
    FitNets: Hints for Thin Deep Nets
    https://arxiv.org/pdf/1412.6550.pdf
    """

    def Hint(self, fm_s, fm_t):
        loss = F.mse_loss(fm_s, fm_t)

        return loss

    """
    Do Deep Nets Really Need to be Deep?
    http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf
    """

    def Logits(self, out_s, out_t):
        loss = F.mse_loss(out_s, out_t)

        return loss

    def LwM(self, out_s, fm_s, out_t, fm_t, target):
        target_out_t = torch.gather(out_t, 1, target.view(-1, 1))
        grad_fm_t = grad(
            outputs=target_out_t,
            inputs=fm_t,
            grad_outputs=torch.ones_like(target_out_t),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        weights_t = F.adaptive_avg_pool2d(grad_fm_t, 1)
        cam_t = torch.sum(torch.mul(weights_t, grad_fm_t), dim=1, keepdim=True)
        cam_t = F.relu(cam_t)
        cam_t = cam_t.view(cam_t.size(0), -1)
        norm_cam_t = F.normalize(cam_t, p=2, dim=1)

        target_out_s = torch.gather(out_s, 1, target.view(-1, 1))
        grad_fm_s = grad(
            outputs=target_out_s,
            inputs=fm_s,
            grad_outputs=torch.ones_like(target_out_s),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        weights_s = F.adaptive_avg_pool2d(grad_fm_s, 1)
        cam_s = torch.sum(torch.mul(weights_s, grad_fm_s), dim=1, keepdim=True)
        cam_s = F.relu(cam_s)
        cam_s = cam_s.view(cam_s.size(0), -1)
        norm_cam_s = F.normalize(cam_s, p=2, dim=1)

        loss = F.l1_loss(norm_cam_s, norm_cam_t.detach())

        return loss

    """
    Like What You Like: Knowledge Distill via Neuron Selectivity Transfer
    https://arxiv.org/pdf/1707.01219.pdf
    """

    def NST(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), fm_s.size(1), -1)
        fm_s = F.normalize(fm_s, dim=2)

        fm_t = fm_t.view(fm_t.size(0), fm_t.size(1), -1)
        fm_t = F.normalize(fm_t, dim=2)

        loss = (
            self.poly_kernel(fm_t, fm_t).mean()
            + self.poly_kernel(fm_s, fm_s).mean()
            - 2 * self.poly_kernel(fm_s, fm_t).mean()
        )

        return loss

    def poly_kernel(self, fm1, fm2):
        fm1 = fm1.unsqueeze(1)
        fm2 = fm2.unsqueeze(2)
        out = (fm1 * fm2).sum(-1).pow(2)

        return out

    """
    Similarity-Preserving Knowledge Distillation
    https://arxiv.org/pdf/1907.09682.pdf
    """

    def SP(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        G_s = torch.mm(fm_s, fm_s.t())
        norm_G_s = F.normalize(G_s, p=2, dim=1)

        fm_t = fm_t.view(fm_t.size(0), -1)
        G_t = torch.mm(fm_t, fm_t.t())
        norm_G_t = F.normalize(G_t, p=2, dim=1)

        loss = F.mse_loss(norm_G_s, norm_G_t)

        return loss

    def calculate_loss(
        self,
        student_logits,
        teacher_logits,
        y,
        x=None,
        student_feat=None,
        teacher_feat=None,
    ):
        loss = math.inf
        if self.distillation_loss == "TFVirtual":
            loss = self.loss_func(student_logits, y)
        elif self.distillation_loss == "DGKD":
            loss = self.loss_func(student_logits, teacher_logits, y)
        elif self.distillation_loss == "Sobolev":
            loss = self.loss_func(student_logits, teacher_logits[0], y, x)
        elif self.distillation_loss in ["DML", "ST", "Logits"]:
            loss = self.loss_func(student_logits, teacher_logits[0])
        elif self.distillation_loss in ["RKD", "PKT", "AT", "CC", "Hint", "NST", "SP"]:
            loss = self.loss_func(student_feat, teacher_feat[0])
        elif self.distillation_loss in ["LwM"]:
            loss = self.loss_func(
                student_logits, student_feat, teacher_logits[0], teacher_feat[0], y
            )
        else:
            loss = self.loss_func(student_logits, teacher_logits[0], y)
        return loss

    def training_step(self, batch, batch_idx):
        # x inputs, y labels
        x, x_len, y, y_len = batch

        student_logits = self.forward(x)
        student_feat = self.model.feat
        teacher_logits = []
        teacher_feat = []
        for teacher in self.teachers:
            teacher.eval()
            with torch.no_grad():
                teacher_logits.append(teacher(x))
                teacher_feat.append(teacher.model.feat)

        y = y.view(-1)

        assert len(teacher_logits) >= 1
        loss = self.calculate_loss(
            student_logits,
            teacher_logits,
            y,
            x,
            student_feat=student_feat,
            teacher_feat=teacher_feat,
        )

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
