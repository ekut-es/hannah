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
        if trainer.current_epoch == self.compress_after-1:
            with torch.no_grad():
                for name, module in pl_module.named_modules():
                    if hasattr(module, "scaled_weight"):
                        module.weight.data = module.scaled_weight
                        module.weight.bias = module.bias



            def replace_modules(module):
                for name, child in module.named_children():
                    replace_modules(child)

                    if isinstance(child, ConvBn1d):
                        tmp =  Conv1d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size, 
                        stride=child.stride,
                        padding=child.padding,
                        groups=child.groups,
                        padding_mode=child.padding_mode,
                        bias=child.bias,
                        dilation=child.dilation,
                        qconfig=child.qconfig,
                        out_quant=True)
                        tmp.weight.data = child.weight
                        tmp.bias = child.bias
                        setattr(module, name, tmp)
                        #print(getattr(module, name).bias)
                        #module[0].weight.data = child.weight
                        #print(module[0].bias)


                    if isinstance(child, ConvBnReLU1d):
                        tmp = ConvReLU1d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size, 
                        stride=child.stride,
                        padding=child.padding,
                        groups=child.groups,
                        padding_mode=child.padding_mode,
                        bias=child.bias,
                        dilation=child.dilation,
                        qconfig=child.qconfig,
                        out_quant=True)
                        tmp.weight.data = child.weight
                        tmp.bias = child.bias
                        setattr(module, name, tmp)
                        #module[0].weight.data = child.weight

            device = pl_module.device
            replace_modules(pl_module)
            pl_module.to(device=device) # otherwise cuda error
            #print(pl_module)


            # get frequencies
            ws = []
            for name, module in pl_module.named_parameters():
                #print(name)
                #print(module)
                ws = np.append(ws, module.cpu().detach().numpy())
            frq = Counter(ws.tolist())
            #print(frq)
            print('##############')
            print(len(frq))     

            print('##############')
