import logging

from typing import Optional

import tabulate

import torch
from torch.nn import CrossEntropyLoss
from pytorch_lightning.callbacks import ModelPruning


class PruningAmountScheduler:
    def __init__(self, target_amount, max_epochs):
        self.target_amount = target_amount
        self.max_epochs = max_epochs

    def __call__(self, current_epoch) -> float:
        if current_epoch == 0:
            return 0.00
        else:
            return self.target_amount / (self.max_epochs - 1.0)


class FilteredPruning(ModelPruning):
    def filter_parameters_to_prune(self, parameters_to_prune=None):
        """
        Filter out unprunable parameters
        """

        def filter_func(x):
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
