from typing import List, Optional

from hydra import TaskFunction
from hydra.core.config_loader import ConfigLoader
from hydra.plugins.sweeper import Sweeper
from omegaconf import DictConfig

from .config import OptimConf


class UNASSweeper(Sweeper):
    """Class for neural architecture search"""

    def __init__(self, optim: OptimConf, parametrization: Optional[DictConfig]):
        from ._impl import UNASSweeperImpl

        self.sweeper = UNASSweeperImpl(optim, parametrization)

    def setup(
        self,
        config: DictConfig,
        config_loader: ConfigLoader,
        task_function: TaskFunction,
    ) -> None:
        print("Setup sweeper")
        return self.sweeper.setup(config, config_loader, task_function)

    def sweep(self, arguments: List[str]) -> None:
        print("Sweep")
        return self.sweeper.sweep(arguments)
