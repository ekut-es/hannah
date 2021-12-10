import torch
import torch.nn as nn
import numpy as np

from pytorch_lightning.callbacks import Callback
from torch.nn.modules.module import register_module_backward_hook
from ..models.factory.qconfig import SymmetricQuantization
from collections import Counter

class CompressionHuff(Callback):
    def __init__(self, compress_after):
        self.compress_after = compress_after


    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self.compress_after:
            ws = []
            #quantizer = SymmetricQuantization(6)
            with torch.no_grad():
                for name, module in pl_module.named_modules():
                    print(name)
                    if hasattr(module, "scaled_weight"):
                        print(name)
                        #print(module.scaled_weight)
                        
                    #module.data = quantizer(module)
                    #print(module.scaled_weight)
                #ws = np.append(ws, module.cpu().detach().numpy())
            #frq = Counter(ws.tolist())
            #print(frq)
            #print(len(frq))
            print('##############')
