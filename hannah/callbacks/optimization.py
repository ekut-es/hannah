import logging
from typing import Iterable, Mapping, Union

from pytorch_lightning.callbacks import Callback
from torch import Tensor

logger = logging.getLogger(__name__)

monitor_type = Union[Iterable[Mapping[str, any]], Mapping[str, any], Iterable[str], str]

logger = logging.getLogger(__name__)

monitor_type = Union[Iterable[Mapping[str, any]], Mapping[str, any], Iterable[str], str]


class HydraOptCallback(Callback):
    def __init__(self, monitor: monitor_type = ["val_loss"]):
        self.values = {}
        self.val_values = {}
        self.test_values = {}
        self.monitor = []
        self.directions = []

        self._extract_monitor(monitor)

        logger.info("Monitoring the following values for optimization")
        for m, d in zip(self.monitor, self.directions):
            logger.info(
                "  - %s direction: %s(%d)", m, "maximize" if d < 0 else "minimize", d
            )

    def _extract_monitor(self, monitor):
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

    def on_test_end(self, trainer, pl_module):
        callback_metrics = trainer.callback_metrics

        for k, v in callback_metrics.items():
            if k.startswith("test"):
                self.test_values[k] = v

        for monitor, direction in zip(self.monitor, self.directions):
            if monitor in callback_metrics:
                self.values[monitor] = callback_metrics[monitor] * direction

    def on_validation_end(self, trainer, pl_module):
        callback_metrics = trainer.callback_metrics

        for k, v in callback_metrics.items():
            if k.startswith("val"):
                self.val_values[k] = v

        for monitor, direction in zip(self.monitor, self.directions):
            if monitor in callback_metrics:
                self.values[monitor] = callback_metrics[monitor] * direction

    def test_result(self):
        return self.test_values

    def val_result(self):
        return self.val_values

    def result(self, dict=False):

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
