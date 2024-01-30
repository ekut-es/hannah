from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
from typing import Any, Dict

import numpy as np
import yaml

from hannah.nas.search.utils import np_to_primitive
msglogger = logging.getLogger(__name__)

@dataclass()
class SearchResult:
    index: int
    parameters: Dict[str, Any]
    result: Dict[str, float]

    def costs(self):
        return np.asarray(
            [float(self.result[k]) for k in sorted(self.result.keys())],
            dtype=np.float32,
        )


class Sampler(ABC):
    def __init__(self,
                 parent_config,
                 output_folder=".") -> None:
        self.history = []
        self.output_folder = Path(output_folder)
        self.parent_config = parent_config

    @abstractmethod
    def next_parameters(self):
        ...

    def tell(self, parameters, metrics):
        return self.tell_result(parameters, metrics)

    def tell_result(self, parameters, metrics):
        "Tell the result of a task"

        parameters = np_to_primitive(parameters)
        result = SearchResult(len(self.history), parameters, metrics)
        self.history.append(result)
        self.save()
        return None

    def save(self):
        history_file = self.output_folder / "history.yml"
        history_file_tmp = history_file.with_suffix(".tmp")

        with history_file_tmp.open("w") as history_data:
            yaml.dump(self.history, history_data)
        shutil.move(history_file_tmp, history_file)
        msglogger.info(f"Updated {history_file.name}")

    def load(self):
        history_file = self.output_folder / "history.yml"
        self.history = []
        with history_file.open("r") as history_data:
            self.history = yaml.unsafe_load(history_data)

        msglogger.info("Loaded %d points from history", len(self.history))