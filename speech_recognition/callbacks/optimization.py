import logging
from pytorch_lightning.callbacks import Callback
from torch import Tensor


class HydraOptCallback(Callback):
    def __init__(self, monitor=["val_loss"]):
        self.monitor = monitor
        self.values = {}
        self.tests = {}

    def on_test_end(self, trainer, pl_module):
        callback_metrics = trainer.callback_metrics

        for telement in [
            "test_accuracy",
            "test_error",
            "test_recall",
            "test_precision",
            "rest_f1",
        ]:
            self.tests[telement] = callback_metrics[telement]

    def on_validation_end(self, trainer, pl_module):
        callback_metrics = trainer.callback_metrics

        for monitor in self.monitor:
            if monitor in callback_metrics:
                self.values[monitor] = callback_metrics[monitor]

    def test_result(self):
        return self.tests

    def result(self):

        return_values = {}
        for key, value in self.values.items():
            if isinstance(value, Tensor):
                value = float(value.cpu())
            else:
                value = float(value)

            return_values[key] = value

        if len(return_values) == 1:
            return list(return_values.values())[0]

        return return_values
