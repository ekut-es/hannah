from pytorch_lightning.callbacks import Callback


class HydraOptCallback(Callback):
    def __init__(self, monitor=["val_loss"]):
        self.monitor = monitor
        self.values = {}

    def on_validation_end(self, trainer, pl_module):
        callback_metrics = trainer.callback_metrics

        for monitor in self.monitor:
            if monitor in callback_metrics:
                self.values[monitor] = callback_metrics[monitor]

    def result(self):
        if len(self.values) == 1:
            return list(self.values.values())[0]

        return self.values
