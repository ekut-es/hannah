from abc import ABC, abstractmethod


class Presampler(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def check(self, model, estimated_metrics):
        ...


class SingleRangeChecker(Presampler):
    def __init__(self, min, max, metric='total_weights') -> None:
        super().__init__()
        self.min = min
        self.max = max
        self.metric = metric

    def check(self, model, estimated_metrics):
        value = estimated_metrics[self.metric]
        return value >= self.min and value <= self.max
    

class MultiRangeChecker(Presampler):
    def __init__(self, bounds) -> None:
        super().__init__()
        self.bounds = bounds
        self.metrics = list(self.bounds.keys())

    def check(self, model, estimated_metrics):
        for metric in self.metrics:
            value = estimated_metrics[metric]
            if not (value >= self.bounds[metric]['min'] and value <= self.bounds[metric]['max']):
                return False
        return True
