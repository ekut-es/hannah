import torch
import torch.nn as nn
import numpy as np

from pytorch_lightning.callbacks import Callback
from torch.nn.modules.module import register_module_backward_hook



                
''' Singular Value Decomposition of the linear layer of a neural network - tested for conv-net-trax and tc-res8'''
class SVD(Callback):
    def __init__(self, rank_svd, compress_after):
        self.rank = rank_svd
        self.compress_after = compress_after
        super().__init__()


    def on_epoch_start(self, trainer, pl_module):
        
        # Train - apply SVD - restructure - retrain
        if trainer.current_epoch == self.compress_after/2: 
            with torch.no_grad():
                for name, module in pl_module.named_modules():

                    # First case: conv-net-trax model with Sequential Layers
                    if name == "model.linear.0.0" and not isinstance(pl_module.model.linear[0][0], nn.Sequential):
                        U, S, Vh = torch.linalg.svd(module.weight, full_matrices=True) # apply SVD

                        # Slicing of matrices for rank r and reassembly
                        U = U[:, :self.rank]
                        SVh = torch.matmul(torch.diag(S), Vh[:S.size()[0], :]) 
                        SVh = SVh[:self.rank, :]

                        '''Replace linear layer by sequential layer with two linear layers,
                        one containing SVh and the other U, approximating the original fully connected layer'''
                        original_fc = pl_module.model.linear[0][0]
                        new_fc = nn.Sequential(
                                nn.Linear(original_fc.in_features, self.rank, bias=original_fc.bias),
                                nn.Linear(self.rank, original_fc.out_features, bias=original_fc.bias)
                            )
                        pl_module.model.linear[0][0] = new_fc

                        pl_module.model.linear[0][0][0].weight = torch.nn.Parameter(SVh, requires_grad=True)
                        pl_module.model.linear[0][0][1].weight = torch.nn.Parameter(U, requires_grad=True)
                    

                    # Second case: tc-res8 model
                    elif type(module) in [nn.Linear] and name != "model.linear.0.0.0" and name != "model.linear.0.0.1" and not isinstance(pl_module.model.fc, nn.Sequential):
                        U, S, Vh = torch.linalg.svd(module.weight, full_matrices=True)
                        U = U[:, :self.rank]
                        SVh = torch.matmul(torch.diag(S), Vh[:S.size()[0], :]) 
                        SVh = SVh[:self.rank, :]
                        original_fc = pl_module.model.fc
                        new_fc = nn.Sequential(
                                    nn.Linear(original_fc.in_features, self.rank, bias=original_fc.bias),
                                    nn.Linear(self.rank, original_fc.out_features, bias=original_fc.bias)
                                )
                        pl_module.model.fc = new_fc
                        pl_module.model.fc[0].weight = torch.nn.Parameter(SVh, requires_grad=True)
                        pl_module.model.fc[1].weight = torch.nn.Parameter(U, requires_grad=True)

        return pl_module




        '''compressed_weights = 0
        for name, module in pl_module.named_modules():
            if type(module) in [nn.Linear] or name == "model.linear.0.0":
                U, S, Vh = torch.linalg.svd(module.weight, full_matrices=True)
                size_S = list(S.size())[0]
                for i in range(self.rank, size_S):
                    S[i] = 0
                compressed_weights = torch.matmul(U, torch.matmul(torch.diag(S), Vh[:12, :]))
                print(compressed_weights.shape)
                if type(module) in [nn.Linear]:
                    pl_module.model.fc.weight = torch.nn.Parameter(compressed_weights, requires_grad=True)

                else:
                    pl_module.model.linear[0][0].weight = torch.nn.Parameter(compressed_weights, requires_grad=True)
                
        return pl_module'''
  


'''

        ########### Test if weights were updated #############
        for name, param in pl_module.named_parameters():
            if "linear.0.0" in name:
                print((param == compressed_weights).all())
        '''

        
