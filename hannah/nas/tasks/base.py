from abc import ABC, abstractmethod
from hydra.utils import instantiate


class SearchTask(ABC):
    def __init__(self, budget=2000, parent_config=None, space=None):
        self.config = parent_config
        self.budget = budget
        self.search_space = instantiate(space)

    @abstractmethod
    def run(self):
        pass


# class WeightSharingTask:
#    def __init__(self, algorithm=DartsAlgorithm)
