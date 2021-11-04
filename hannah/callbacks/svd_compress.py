import torch
import torch.nn as nn
import numpy as np

from pytorch_lightning.callbacks import Callback
from torch.nn.modules.module import register_module_backward_hook



                

class SVD(Callback):
    def __init__(self, rank_svd):
        self.rank = rank_svd
        super().__init__()


    def on_epoch_end(self, trainer, pl_module):
        print(self.rank)
        compressed_weights = 0
        for name, module in pl_module.named_modules():
            if type(module) in [nn.Linear] or name == "model.linear.0.0":
                U, S, Vh = torch.linalg.svd(module.weight, full_matrices=True)
                size_S = list(S.size())[0]
                for i in range(self.rank, size_S):
                    S[i] = 0
                compressed_weights = torch.matmul(U, torch.matmul(torch.diag(S), Vh[:12, :]))

                if type(module) in [nn.Linear]:
                    pl_module.model.fc.weight = torch.nn.Parameter(compressed_weights, requires_grad=True)
                else:
                    pl_module.model.linear[0][0].weight = torch.nn.Parameter(compressed_weights, requires_grad=True)
                
        return pl_module




        


'''
                def svd_test(module, grad_input, grad_output):
                    ll_weights = module.weight
                    print(ll_weights)
                    U, S, Vh = torch.linalg.svd(ll_weights, full_matrices=True)
                    size_S = list(S.size())[0]
                    for i in range(self.rank, size_S):
                        S[i] = 0
                    compressed_weights = torch.matmul(U, torch.matmul(torch.diag(S), Vh[:12, :]))    
                    pl_module.model.fc.weight = torch.nn.Parameter(compressed_weights, requires_grad=True)
                module.register_full_backward_hook(svd_test)

        ########### Test if weights were updated #############
        for name, param in pl_module.named_parameters():
            if "linear.0.0" in name:
                print((param == compressed_weights).all())
        '''

        
