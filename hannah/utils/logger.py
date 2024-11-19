#
# Copyright (c) 2024 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import datetime
import json
import logging
import os
import pathlib
from argparse import Namespace
from typing import Any, Dict, List, Mapping, Optional, Set, Union

from lightning_fabric.loggers.logger import rank_zero_experiment
from lightning_fabric.utilities import rank_zero_only, rank_zero_warn
from lightning_fabric.utilities.cloud_io import get_filesystem
from pytorch_lightning.loggers import Logger
from torch import Tensor

_PATH = Union[str, pathlib.Path]

import fsspec

log = logging.getLogger(__name__)


def _is_dir(fs, path, strict=False):
    return fs.isdir(path) or (not strict and fs.exists(path) and not fs.isfile(path))


def _add_prefix(
    metrics: Mapping[str, Union[Tensor, float]], prefix: str, separator: str
) -> Mapping[str, Union[Tensor, float]]:
    """Insert prefix before each key in a dict, separated by the separator.

    Args:
        metrics: Dictionary with metric names as keys and measured quantities as values
        prefix: Prefix to insert before each key
        separator: Separates prefix and original key name

    Returns:
        Dictionary with prefix and separator inserted before each key

    """
    if not prefix:
        return metrics
    return {f"{prefix}{separator}{k}": v for k, v in metrics.items()}


class JSONLogger(Logger):
    LOGGER_JOIN_CHAR = "_"

    def __init__(
        self,
        root_dir: _PATH,
        name: str = "lightning_logs",
        version: Optional[Union[int, str]] = None,
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
    ):
        super().__init__()
        root_dir = os.fspath(root_dir)
        self._root_dir = root_dir
        self._name = name or ""
        self._version = version
        self._prefix = prefix
        self._fs = get_filesystem(root_dir)
        self._experiment: Optional[_ExperimentWriter] = None
        self._flush_logs_every_n_steps = flush_logs_every_n_steps

    @property
    def name(self) -> str:
        """Gets the name of the experiment.

        Returns:
            The name of the experiment.

        """
        return self._name

    @property
    def version(self) -> Union[int, str]:
        """Gets the version of the experiment.

        Returns:
            The version of the experiment if it is specified, else the next version.

        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    def root_dir(self) -> str:
        """Gets the save directory where the versioned JSON experiments are saved."""
        return self._root_dir

    @property
    def log_dir(self) -> str:
        """The log directory for this run.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.

        """
        # create a pseudo standard path
        version = (
            self.version if isinstance(self.version, str) else f"version_{self.version}"
        )
        return os.path.join(self._root_dir, self.name, version)

    @property
    @rank_zero_experiment
    def experiment(self) -> "_ExperimentWriter":
        """Actual ExperimentWriter object. To use ExperimentWriter features anywhere in your code, do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment is not None:
            return self._experiment

        os.makedirs(self._root_dir, exist_ok=True)
        self._experiment = _ExperimentWriter(log_dir=self.log_dir)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:  # type: ignore[override]
        pass

    @rank_zero_only
    def log_metrics(  # type: ignore[override]
        self, metrics: Dict[str, Union[Tensor, float]], step: Optional[int] = None
    ) -> None:
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        metrics["date"] = datetime.datetime.now().isoformat()
        if step is None:
            step = len(self.experiment.metrics)
        self.experiment.log_metrics(metrics, step)
        if (step + 1) % self._flush_logs_every_n_steps == 0:
            self.save()

    @rank_zero_only
    def save(self) -> None:
        super().save()
        self.experiment.save()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if self._experiment is None:
            # When using multiprocessing, finalize() should be a no-op on the main process, as no experiment has been
            # initialized there
            return
        self.save()

    def _get_next_version(self) -> int:
        versions_root = os.path.join(self._root_dir, self.name)

        if not _is_dir(self._fs, versions_root, strict=True):
            log.warning("Missing logger folder: %s", versions_root)
            return 0

        existing_versions = []
        for d in self._fs.listdir(versions_root):
            full_path = d["name"]
            name = os.path.basename(full_path)
            if _is_dir(self._fs, full_path) and name.startswith("version_"):
                dir_ver = name.split("_")[1]
                if dir_ver.isdigit():
                    existing_versions.append(int(dir_ver))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1


class _ExperimentWriter:
    r"""Experiment writer for CSVLogger.

    Args:
        log_dir: Directory for the experiment logs

    """

    NAME_METRICS_FILE = "metrics.jsonl"

    def __init__(self, log_dir: str) -> None:
        self.metrics: List[Dict[str, float]] = []
        self.metrics_keys: List[str] = []

        self._fs = get_filesystem(log_dir)
        self.log_dir = log_dir
        if self._fs.exists(self.log_dir) and self._fs.listdir(self.log_dir):
            rank_zero_warn(
                f"Experiment logs directory {self.log_dir} exists and is not empty."
                " Previous log files in this directory will be deleted when the new ones are saved!"
            )
        self._fs.makedirs(self.log_dir, exist_ok=True)

        self.metrics_file_path = os.path.join(self.log_dir, self.NAME_METRICS_FILE)

    def log_metrics(
        self, metrics_dict: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Record metrics."""

        def _handle_value(value: Union[Tensor, Any]) -> Any:
            if isinstance(value, Tensor):
                return value.item()
            return value

        if step is None:
            step = len(self.metrics)

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        metrics["step"] = step
        self.metrics.append(metrics)

    def save(self) -> None:
        """Save recorded metrics into files."""
        for metric in self.metrics:
            with self._fs.open(self.metrics_file_path, "a") as fp:
                fp.write(json.dumps(metric) + "\n")

        self.metrics = []  # reset
