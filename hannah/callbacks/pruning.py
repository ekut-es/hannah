#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
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
from typing import Optional

import tabulate
import torch
from pytorch_lightning.callbacks import ModelPruning
from torch.nn import BatchNorm1d


class PruningAmountScheduler:
    def __init__(self, target_amount, max_epochs):
        self.target_amount = target_amount
        self.max_epochs = max_epochs

    def __call__(self, current_epoch) -> float:
        if current_epoch == 0:
            return 0.00
        else:
            amount = self.target_amount / (self.max_epochs - 1.0)
            logging.info("Pruning amount %f", amount)
            return amount


class FilteredPruning(ModelPruning):
    def on_before_accelerator_backend_setup(self, trainer, pl_module):
        # FIXME: calling setup here breaks assumptions about call order and will lead to setup being called twice
        pl_module.setup("fit")
        super().on_before_accelerator_backend_setup(trainer, pl_module)

    def _run_pruning(self, current_epoch):
        prune = (
            self._apply_pruning(current_epoch)
            if callable(self._apply_pruning)
            else self._apply_pruning
        )
        amount = self.amount(current_epoch) if callable(self.amount) else self.amount
        if not prune or not amount:
            return

        from hannah.utils import set_deterministic

        with set_deterministic(False):
            self.apply_pruning(amount)

        if (
            self._use_lottery_ticket_hypothesis(current_epoch)
            if callable(self._use_lottery_ticket_hypothesis)
            else self._use_lottery_ticket_hypothesis
        ):
            self.apply_lottery_ticket_hypothesis()

    def filter_parameters_to_prune(self, parameters_to_prune=None):
        """
        Filter out unprunable parameters
        """

        def filter_func(x):
            if isinstance(x[0], BatchNorm1d):
                return False
            if hasattr(x[0], x[1]) and getattr(x[0], x[1]) is not None:
                return True
            return False

        parameters_to_prune = list(filter(filter_func, parameters_to_prune))
        return parameters_to_prune

    def on_test_end(self, trainer, pl_module) -> None:
        total_elements = 0.0
        total_zero_elements = 0.0
        sparsity_table = []
        for name, parameter in pl_module.named_parameters():
            layer_zero = float(torch.sum(parameter == 0))
            layer_total = float(parameter.nelement())
            layer_sparsity = 100.0 * layer_zero / layer_total
            sparsity_table.append([name, layer_sparsity])

            total_elements += layer_total
            total_zero_elements += layer_zero

        total_sparsity = 100.0 * total_zero_elements / total_elements

        sparsity_table.append(["Whole Model", total_sparsity])

        logging.info(
            "\n%s", tabulate.tabulate(sparsity_table, headers=["Layer", "Sparsity [%]"])
        )
