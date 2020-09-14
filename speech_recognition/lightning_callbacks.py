from pytorch_lightning.callbacks import Callback
import distiller


class DistillerCallback(Callback):
    def __init__(self, lit_module, model, optimizer, config, msglogger):
        super().__init__()
        self.hparams = config
        self.msglogger = msglogger
        self.optimizer = optimizer
        self.lit_module = lit_module
        self.model = lit_module.model

    def on_init_end(self, trainer):
        self.msglogger.info("!!! on_init_end")
        if self.hparams["compress"]:
            print(f"model device: {self.lit_module.device}")
            self.model.to(self.lit_module.device)
            self.msglogger.info("Activating compression scheduler")
            self.lit_module.compression_scheduler = distiller.file_config(
                self.model, self.optimizer, self.hparams["compress"]
            )
