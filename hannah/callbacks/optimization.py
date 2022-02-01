import logging
from pytorch_lightning.callbacks import Callback
from torch import Tensor


class HydraOptCallback(Callback):
    def __init__(self, monitor=["val_loss"]):
        self.monitor = monitor
        self.values = {}
        self.val_values = {}
        self.test_values = {}

    def on_test_end(self, trainer, pl_module):
        callback_metrics = trainer.callback_metrics

        for k, v in callback_metrics.items():
            if k.startswith("test"):
                self.test_values[k] = v

        for monitor in self.monitor:
            if monitor in callback_metrics:
                self.values[monitor] = callback_metrics[monitor]

    def on_validation_end(self, trainer, pl_module):
        callback_metrics = trainer.callback_metrics

        for k, v in callback_metrics.items():
            if k.startswith("val"):
                self.val_values[k] = v
                
        for monitor in self.monitor:
            if monitor in callback_metrics:
                self.values[monitor] = callback_metrics[monitor]

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
