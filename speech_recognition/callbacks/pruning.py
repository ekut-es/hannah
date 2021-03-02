from typing import Optional

from torch.nn import CrossEntropyLoss
from pytorch_lightning.callbacks import ModelPruning


class PruningAmountScheduler:
    def __init__(self, target_amount, max_epochs):
        self.target_amount = target_amount
        self.max_epochs = max_epochs

    def __call__(self, current_epoch) -> float:
        if current_epoch == 0:
            return 0.01
        else:
            return self.target_amount * current_epoch / self.max_epochs


class FilteredPruning(ModelPruning):
    def filter_parameters_to_prune(self, parameters_to_prune = None):
        """
        Filter out unprunable parameters
        """

        def filter_func(x):
            if hasattr(x[0], x[1]) and getattr(x[0], x[1]) is not None:
                return True
            return False

        parameters_to_prune = list(filter(filter_func, parameters_to_prune))

        return parameters_to_prune