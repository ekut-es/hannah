from pytorch_lightning.callbacks import Callback


class HydraOptCallback(Callback):
    def __init__(self, monitor="val_loss"):
        self.monitor = monitor
        self.value = 0.0

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.logger_connector.logged_metrics
        if self.monitor in metrics:
            self.value = metrics[self.monitor]

    def result(self):
        return self.value
