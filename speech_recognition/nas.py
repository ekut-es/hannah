from abc import ABC, abstractmethod

from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np
from pytorch_lightning import Trainer, LightningModule

from hannah_optimizer.aging_evolution import AgingEvolution

# TODO: i think this has already been implemented
def fullname(o):
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


class NASTrainerBase(ABC):
    def __init__(
        self, budget=2000, parent_config=None, parametrization=None, bounds=None
    ):
        self.config = parent_config
        self.budget = budget
        self.parametrization = parametrization
        self.bounds = bounds

    @abstractmethod
    def run(self, model):
        pass


class RandomNASTrainer(NASTrainerBase):
    def __init__(self, budget=2000, *args, **kwargs):
        super().__init__(*args, budget=budget, **kwargs)

    def fit(self, module: LightningModule):
        # Presample Population

        # Sample Population

        pass


class AgingEvolutionNASTrainer(NASTrainerBase):
    def __init__(
        self,
        population_size=100,
        budget=2000,
        parametrization=None,
        bounds=None,
        parent_config=None,
    ):
        super().__init__(
            budget=budget,
            bounds=bounds,
            parametrization=parametrization,
            parent_config=parent_config,
        )
        self.population_size = population_size
        self.history = []
        self.random_state = np.random.RandomState()
        self.optimizer = AgingEvolution(
            parametrization=parametrization,
            bounds=bounds,
            random_state=self.random_state,
        )
        self.backend = None

    def run(self):
        # Presample initial Population
        for i in range(self.budget):
            parameters = self.optimizer.next_parameters()
            self.history.append(parameters)

            config = OmegaConf.merge(self.config, parameters.flatten())
            backend = instantiate(config.backend)
            model = instantiate(
                config.module,
                dataset=config.dataset,
                model=config.model,
                optimizer=config.optimizer,
                features=config.features,
                scheduler=config.get("scheduler", None),
                normalizer=config.get("normalizer", None),
            )
            model.setup("train")
            backend_metrics = backend.estimate(model)

            for k, v in backend_metrics.items():
                print(f"{k}: {v}")

        # Sample initial Population

        # Mutate current population

        # validate population
        # self.trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloaders, datamodule=datamodule)
