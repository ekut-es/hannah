from pytorch_lightning.callbacks import Callback


class CompressionCallback(Callback):
    def __init__(self, fold_bn=1.0, quantization=None, pruning=None):
        self.fold_bn = fold_bn
        self.quantization = quantization
        self.pruning = pruning

    def on_train_start(self, trainer, pl_module):
        pass
