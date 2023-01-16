import numpy as np
from .base_sampler import Sampler

class RandomSampler(Sampler):
    def __init__(self, parametrization) -> None:
        super().__init__()
        self.parametrization = parametrization

    def next_parameters(self):
        parameter_values = {}
        for key, param in self.parametrization.items():
            parameter_values[key] = param.sample()
        self.history.append(parameter_values)
        return parameter_values