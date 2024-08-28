#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
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
import logging
from collections import defaultdict
from typing import Any, Iterable, List, Mapping, Union

from pytorch_lightning.callbacks import Callback
from torch import Tensor

logger = logging.getLogger(__name__)

monitor_type = Union[Iterable[Mapping[str, Any]], Mapping[str, Any], Iterable[str], str]

logger = logging.getLogger(__name__)

monitor_type = Union[Iterable[Mapping[str, Any]], Mapping[str, Any], Iterable[str], str]


class HydraOptCallback(Callback):
    """ """

    def __init__(self, monitor: monitor_type = ["val_loss"]):
        self.values = {}
        self.val_values = {}
        self.test_values = {}
        self.train_values = {}
        self.monitor: List[str] = []
        self.directions: List[int] = []
        self._curves = defaultdict(dict)

        self._extract_monitor(monitor)

        logger.info("Monitoring the following values for optimization")
        for m, d in zip(self.monitor, self.directions):
            logger.info(
                "  - %s direction: %s(%d)", m, "maximize" if d < 0 else "minimize", d
            )

    def _extract_monitor(self, monitor):
        """

        Args:
          monitor:

        Returns:

        """
        if isinstance(monitor, Mapping):
            self._add_monitor_mapping(monitor)
        elif isinstance(monitor, Iterable):
            for m in monitor:
                if isinstance(m, Mapping):
                    self._add_monitor_mapping(m)
                else:
                    self.monitor.append(m)
                    self.directions.append(1)
        elif isinstance(monitor, str):
            self.monitor.append(monitor)
            self.directions.append(1)

    def _add_monitor_mapping(self, monitor):
        """

        Args:
          monitor:

        Returns:

        """
        self.monitor.append(monitor["metric"])
        if "direction" in monitor:
            direction = monitor["direction"]
            if direction in ["maximize", "maximise"]:
                direction = -1.0
            elif direction in ["minimise", "minimize"]:
                direction = 1.0
            self.directions.append(float(direction))
        else:
            self.directions.append(-1.0)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",  outputs: 'STEP_OUTPUT', batch: Any, batch_idx: int) -> None:
        callback_metrics =  trainer.callback_metrics

        for k, v in callback_metrics.items():
            if k.startswith("train"):
                if isinstance(v, Tensor):
                    if v.numel() == 1:
                        value = v.item()
                    else:
                        continue
                value = float(value)
                self.train_values[k] = value

        for monitor, direction in zip(self.monitor, self.directions):
            if monitor in callback_metrics:
                monitor_val = callback_metrics[monitor] * direction
                if monitor.startswith("train"):
                    self._curves[monitor][trainer.global_step] = monitor_val
                    
                    self.values[monitor] = monitor_val

    def on_test_end(self, trainer, pl_module):
        """

        Args:
          trainer:
          pl_module:

        Returns:

        """
        callback_metrics = trainer.callback_metrics

        for k, v in callback_metrics.items():
            if k.startswith("test"):
                value = v
                if isinstance(v, Tensor):
                    if v.numel() == 1:
                        value = v.item()
                    else:
                        continue
                value = float(value)
                self.test_values[k] = value

        for monitor, direction in zip(self.monitor, self.directions):
            if monitor in callback_metrics:
                monitor_val = callback_metrics[monitor] * direction
                self.values[monitor] = monitor_val

    def on_validation_end(self, trainer, pl_module):
        """

        Args:
          trainer:
          pl_module:

        Returns:

        """
        # Skip evaluation of validation metrics during sanity check
        if trainer.sanity_checking:
            return
        
        callback_metrics = trainer.callback_metrics

        for k, v in callback_metrics.items():
            if k.startswith("val"):
                self.val_values[k] = v

        for monitor, direction in zip(self.monitor, self.directions):
            if monitor in callback_metrics:
                try:
                    monitor_val = float(callback_metrics[monitor])
                    directed_monitor_val = monitor_val * direction
                   
                    self.values[monitor] = directed_monitor_val
                    self._curves[monitor][trainer.global_step] = directed_monitor_val
                except Exception:
                    pass

    def test_result(self):
        """ """
        return self.test_values

    def val_result(self):
        """ """
        return self.val_values

    def result(self, dict=False):
        """

        Args:
          dict:  (Default value = False)

        Returns:

        """

        return_values = {}
        for key, value in self.values.items():
            if isinstance(value, Tensor):
                value = float(value.cpu())
            else:
                value = float(value)

            return_values[key] = value

        if len(return_values) == 1 and dict is False:
            return list(return_values.values())[0]

        return return_values

    def curves(self, dict=False):
        """

        Args:
          dict:  (Default value = False)

        Returns:

        """

        return_values = defaultdict(list)
        for key, value_dict in self._curves.items():
            for step, value in value_dict.items():
                if isinstance(value, Tensor):
                    value = float(value.cpu())
                else:
                    value = float(value)

                return_values[key].append((step, value))

        if len(return_values) == 1 and dict is False:
            return list(return_values.values())[0]

        return return_values

