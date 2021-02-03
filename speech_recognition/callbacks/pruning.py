from typing import Optional

from torch.nn import CrossEntropyLoss
from pytorch_lightning.callbacks import ModelPruning


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