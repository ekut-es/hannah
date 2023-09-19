#
# Copyright (c) 2023 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import logging
import weakref

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

msglogger = logging.getLogger(__name__)


class HardLabeling:
    def __init__(
        self,
        model: nn.Module,
        loss=None,
        loss_ratio: float = 1.0,
        constc_reg_augm: bool = False,
        th_confdc_pos: float = None,
        th_uncert_pos: float = None,
        th_confdc_neg: float = None,
        th_uncert_neg: float = None,
        num_fwd_passes: int = 10,
        augment=None,
    ):
        super().__init__()
        self.model = model
        self.augment = augment
        self.loss = (
            loss if loss is not None else nn.CrossEntropyLoss(label_smoothing=0.00)
        )
        self.loss_ratio = loss_ratio
        self.fixmatch = constc_reg_augm
        self.tau_p = th_confdc_pos
        self.tau_n = th_confdc_neg
        self.kappa_p = th_uncert_pos
        self.kappa_n = th_uncert_neg
        self.num_fwd_passes = num_fwd_passes
        self.cfg_confidence = self.tau_p is not None or self.tau_n is not None
        self.cfg_uncertainty = self.kappa_p is not None or self.kappa_n is not None

        if self.tau_p is None and self.tau_n is None:
            msglogger.warning(
                "Performing Pseudo-Labeling without confidence thresholds."
            )
        if self.cfg_uncertainty and len(self.get_dropout_layers()) == 0:
            msglogger.critical(
                "Monte Carlo uncertainty threshold is specified, but model \
                 does not contain dropout layers."
            )

    def training_step(
        self,
        unlabeled_data: torch.Tensor,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch_idx: int = -1,
    ) -> torch.Tensor:
        """Calculate pseudo label loss from unlabeled data."""
        x = unlabeled_data["data"]
        if self.fixmatch:
            x_aug_wk, _ = self.augment(x, pipeline="weak")
            x = x_aug_wk
        else:
            x, _ = self.augment(x)

        self.model.eval()
        for layer in self.get_dropout_layers():
            layer.train()
        num_passes = self.num_fwd_passes if self.cfg_uncertainty else 1
        with torch.no_grad():
            logits = [
                self.model.forward(x).logits.detach() for i in range(num_passes - 1)
            ]
        last_logits = self.model.forward(x).logits
        logits.append(last_logits)
        logits = torch.stack(logits)
        probs = F.softmax(logits, dim=-1)
        self.model.train()

        dev = logits.device
        mean = probs.mean(0)
        std = probs.std(0)

        for c in range(pl_module.num_classes):
            if self.cfg_confidence:
                pl_module.logger.experiment.add_scalars(
                    f"pseudo_confidence_c{c}",
                    {"min": mean[:, c].amin(), "max": mean[:, c].amax()},
                    global_step=pl_module.global_step,
                )
            if self.cfg_uncertainty:
                pl_module.logger.experiment.add_scalars(
                    f"pseudo_uncertainty_c{c}",
                    {"min": std[:, c].amin(), "max": std[:, c].amax()},
                    global_step=pl_module.global_step,
                )

        loss = torch.tensor(0.0, device=dev)

        # positive learning
        mask_p = torch.zeros(last_logits.shape[1], dtype=torch.bool)
        if self.tau_p is not None:
            max_idx = mean.max(-1).indices
            max_mean = mean * F.one_hot(max_idx, pl_module.num_classes)
            threshold = torch.tensor(self.tau_p, device=dev)
            mask_p = max_mean.ge(threshold).any(-1)

            if self.kappa_p is not None:
                threshold = torch.tensor(self.kappa_p, device=dev)
                mask_p = mask_p.logical_and(std.le(threshold).any(-1))

            pseudo_labels_p = max_idx.masked_select(mask_p)

            pl_module.log(
                "numel_pseudo_labels_pos",
                float(pseudo_labels_p.numel()),
                batch_size=pl_module.batch_size,
            )
            if pseudo_labels_p.numel() > 0:
                loss += self.compute_loss(
                    inputs=x[mask_p],
                    logits=last_logits[mask_p],
                    targets=pseudo_labels_p,
                )

        # negative learning (multi-label)
        if self.tau_n is not None:
            mask_n = (~mask_p).unsqueeze(-1).expand(last_logits.shape)
            threshold = torch.tensor(self.tau_n, device=dev)
            mask_n = mask_n.logical_and(mean.le(threshold))
            if self.kappa_n is not None:
                threshold = torch.tensor(self.kappa_n, device=dev)
                mask_n = mask_n.logical_and(std.le(threshold))

            pseudo_labels_n = mask_n[mask_n.any(-1)].to(dtype=torch.int64)

            pl_module.log(
                "numel_pseudo_labels_neg",
                float(pseudo_labels_n.shape[0]),
                batch_size=pl_module.batch_size,
            )
            if pseudo_labels_n.numel() > 0:
                loss += self.compute_loss(
                    inputs=x[mask_n.any(-1)],
                    logits=last_logits[mask_n.any(-1)],
                    targets=pseudo_labels_n,
                    loss_fn=self.negative_cre_loss(),
                )

        # otherwise, non-thresholded pseudo labeling
        if not self.cfg_confidence:
            loss += self.compute_loss(x, last_logits, last_logits.argmax(-1))

        return self.loss_ratio * loss

    def get_dropout_layers(self):
        """Returns all model layers of class dropout or dropblock."""
        classes = ["dropout", "drop_block", "dropblock"]
        layers = [
            module
            for module in self.model.modules()
            if any(c in module.__class__.__name__.lower() for c in classes)
        ]
        return layers

    def compute_loss(self, inputs, logits, targets, loss_fn=None):
        """Helper function to compute loss, possibly with consistency
        regularization by augmentations (FixMatch)."""
        if loss_fn is None:
            loss_fn = self.loss

        if self.fixmatch:
            x_aug_st, _ = self.augment(inputs, pipeline="strong")
            logits_aug_st = self.model.forward(x_aug_st).logits
            loss = loss_fn(logits_aug_st, targets.detach())
        else:
            loss = loss_fn(logits, targets.detach())

        return loss

    def negative_cre_loss(logits, targets):
        """Cross Entropy Loss for negative learning which requires a mutli-
        class and multi-label loss function."""
        sm_inv = 1 - F.softmax(logits, dim=-1)
        loss = -(sm_inv.log() * targets).sum(-1) / targets.sum(-1)
        return torch.mean(loss)
