from pathlib import Path
import numpy as np
import pandas as pd
from .base_sampler import Sampler, SearchResult
import os


class DefinedSpaceSampler(Sampler):
    def __init__(self,
                 parent_config,
                 parametrization,
                 data_folder,
                 output_folder=".",
                ) -> None:
        super().__init__(parent_config=parent_config, output_folder=output_folder)
        self.parametrization = parametrization
        print(os.getcwd())
        self.data_folder = "." / Path(data_folder)
        print(self.data_folder)
        self.defined_space = list(pd.read_pickle(self.data_folder)['params'])

        if (self.output_folder / "history.yml").exists():
            self.load()

    def next_parameters(self):
        params = self.defined_space.pop()
        return params, []
