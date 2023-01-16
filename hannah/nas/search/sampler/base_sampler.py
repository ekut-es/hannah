from abc import ABC, abstractmethod

class Sampler(ABC):
    def __init__(self) -> None:
        self.history = []

    @abstractmethod
    def next_parameters(self):
        ...