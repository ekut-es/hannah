import torch
import torch.nn as nn
import numpy as np

from pytorch_lightning.callbacks import Callback
from torch.nn.modules.module import register_module_backward_hook
from ..models.factory.qconfig import SymmetricQuantization
from collections import Counter
from ..models.factory.qat import ConvBn1d, Conv1d, ConvBnReLU1d, ConvReLU1d

class CompressionHuff(Callback):
    def __init__(self, compress_after):
        self.compress_after = compress_after


    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self.compress_after-2:
            with torch.no_grad():
                for name, module in pl_module.named_modules():
                    if hasattr(module, "scaled_weight"):
                        module.weight.data = module.scaled_weight 
            

            def replace_modules(module):
                for name, child in module.named_children():
                    replace_modules(child)

                    if isinstance(child, ConvBn1d):
                        print(name)
                        setattr(module, name, Conv1d(child.in_channels, child.out_channels, child.kernel_size, qconfig=child.qconfig))

                    if isinstance(child, ConvBnReLU1d):
                        print(name)
                        setattr(module, name, ConvReLU1d(child.in_channels, child.out_channels, child.kernel_size, qconfig=child.qconfig))

            replace_modules(pl_module)
            #pl_module.cuda()
            ws = []
            #for name, module in pl_module.named_parameters():
            #    ws = np.append(ws, module.data.cpu().detach().numpy())
            #print(pl_module)
            #frq = Counter(ws.tolist())
            #print(frq)
            #print(len(frq))
            print('##############')
