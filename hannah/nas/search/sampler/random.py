import numpy as np


class RandomSampler:
    def __init__(self, parameters) -> None:
        self.parameters = parameters
        self.history = []

    def sample(self):
        parametrization = {}
        for key, param in self.parameters.items():
            parametrization[key] = param.sample()
        self.history.append(parametrization)
        return parametrization