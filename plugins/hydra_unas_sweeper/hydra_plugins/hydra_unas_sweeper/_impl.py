import logging

from typing import Dict, Any, List

from hydra.core.plugins import Plugins
from hydra.core.config_loader import ConfigLoader
from hydra.plugins.launcher import Launcher
from hydra.plugins.sweeper import Sweeper
from hydra.types import TaskFunction
from hydra.utils import instantiate

from omegaconf import DictConfig, ListConfig, OmegaConf

from .config import OptimConf
from .ageing_evolution import AgingEvolution


class UNASSweeperImpl(Sweeper):
    def __init__(self, optim: OptimConf, parametrization: Dict[str, Any]):

        self.optim_conf = optim
        self.parametrization_conf = parametrization

    def setup(
        self,
        config: DictConfig,
        config_loader: ConfigLoader,
        task_function: TaskFunction,
    ) -> None:
        self.job_idx = 0
        self.config = config
        self.config_loader = config_loader
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, config_loader=config_loader, task_function=task_function
        )
        if self.optim_conf.optimizer == "ageing_evolution":
            self.optimizer = AgingEvolution(
                population_size=self.optim_conf.population_size,
                sample_size=self.optim_conf.sample_size,
                eps=self.optim_conf.eps,
            )
        else:
            raise Exception("Undefined optimizer: {self.optim_conf.optimizer}")

    def sweep(self, arguments: List[str]) -> None:
        pass
