import logging

from typing import Dict, Any, List, MutableMapping, MutableSequence

import numpy as np

from hydra.core.plugins import Plugins
from hydra.core.config_loader import ConfigLoader
from hydra.plugins.launcher import Launcher
from hydra.plugins.sweeper import Sweeper
from hydra.types import TaskFunction
from hydra.utils import instantiate

from omegaconf import DictConfig, ListConfig, OmegaConf

from .config import OptimConf
from .aging_evolution import AgingEvolution


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
        self.budget = self.optim_conf.budget
        self.num_workers = self.optim_conf.num_workers

        seed = self.optim_conf.seed
        if seed:
            np.random.seed(seed)

        self.config = config
        self.config_loader = config_loader
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, config_loader=config_loader, task_function=task_function
        )
        if self.optim_conf.optimizer == "aging_evolution":
            self.optimizer = AgingEvolution(
                population_size=self.optim_conf.population_size,
                sample_size=self.optim_conf.sample_size,
                eps=self.optim_conf.eps,
                bounds=self.optim_conf.bounds,
                parametrization=self.parametrization_conf,
            )
        else:
            raise Exception(f"Undefined optimizer: {self.optim_conf.optimizer}")

    def sweep(self, arguments: List[str]) -> None:

        while self.job_idx < self.budget:

            nw = min(self.num_workers, self.budget - self.job_idx)

            parameters = [self.optimizer.next_parameters() for _ in range(nw)]

            for override in parameters:
                from pprint import pprint

                pprint(override)

            overrides = [self._build_overrides(param) for param in parameters]

            from pprint import pprint

            pprint(overrides)

            overrides = [tuple([f"+{k}={v}" for k, v in x.items()]) for x in overrides]
            returns = self.launcher.launch(overrides, initial_job_idx=self.job_idx)

            for param, ret in zip(parameters, returns):
                self.optimizer.tell(param, ret.return_value)

            self.job_idx += nw

    def _build_overrides(self, choices, parent_key="", overrides=None):
        if overrides is None:
            overrides = {}

        if isinstance(choices, MutableMapping):
            if parent_key:
                parent_key += "."
            for k, v in choices.items():
                child_key = parent_key + k
                self._build_overrides(v, child_key, overrides)
        elif isinstance(choices, MutableSequence):
            if parent_key:
                parent_key += "."
            for num, v in enumerate(choices):
                child_key = parent_key + str(num)
                self._build_overrides(v, child_key, overrides)
        else:
            overrides[parent_key] = choices

        return overrides
