import torch
import torch.nn as nn
import numpy as np

from pytorch_lightning.callbacks import Callback
from torch.nn.modules.module import register_module_backward_hook
from ..models.factory.qconfig import SymmetricQuantization
from collections import Counter

class CompressionHuff(Callback):

    def on_epoch_end(self, trainer, pl_module):
        ws = []
        quantizer = SymmetricQuantization(6)
        for param in pl_module.parameters():
            param.data = quantizer(param.data)
            #ws = np.append(ws, param.cpu().detach().numpy())
        #ws.tolist()
        #frq = Counter(ws.tolist())
        #print(frq)
        '''frq = {} 
        for i in range(len(ws)):
            if ws[i] in frq.keys():
                frq[ws[i]] += 1
            else:
                frq[ws[i]] = 1
        total = sum(frq.values())
        frq = {key: value / total for key, value in frq.items()}'''
        #print(frq)
        #print('##############')
