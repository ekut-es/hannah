import numpy as np
from .base_sampler import Sampler, SearchResult

class RandomSampler(Sampler):
    def __init__(self,
                 parametrization,
                ) -> None:
        super().__init__()
        self.parametrization = parametrization

        if (self.output_folder / "history.yml").exists():
            self.load()

    def next_parameters(self):
        parameter_values = {}
        for key, param in self.parametrization.items():
            parameter_values[key] = param.sample()
        return parameter_values