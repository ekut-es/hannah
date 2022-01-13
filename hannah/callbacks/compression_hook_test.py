from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
import numpy as np

from pytorch_lightning.callbacks import Callback
from torch.nn.modules.module import register_module_full_backward_hook
from ..models.factory.qconfig import SymmetricQuantization
from collections import Counter
from ..models.factory.qat import ConvBn1d, Conv1d, ConvBnReLU1d, ConvReLU1d, Linear
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.metrics import pairwise_distances_argmin_min

class CompressionHuff(Callback):
    def __init__(self, compress_after):
        self.compress_after = compress_after


    def on_epoch_end(self, trainer, pl_module):
        
        if trainer.current_epoch == self.compress_after-2:
            with torch.no_grad():
                counter = 0
                for name, module in pl_module.named_modules():
                    if hasattr(module, "scaled_weight"):
                        module.weight.data = module.scaled_weight
                        if not isinstance(module, nn.Linear):
                            bias_shape = [1] * len(module.weight.shape)
                            bias_shape[1] = -1
                            bias = torch.zeros(module.out_channels, device=module.weight.device)
                            bias = module.bias_fake_quant((bias - module.bn.running_mean) * module.scale_factor + module.bn.bias) #.reshape(bias_shape) #.view(-1, 1, 1) #.reshape(bias_shape)
                            module.bias = torch.nn.Parameter(bias)
    

            def replace_modules(module):
                for name, child in module.named_children():
                    replace_modules(child)

                    if isinstance(child, ConvBn1d):
                        tmp = Conv1d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size, 
                        stride=child.stride,
                        padding=child.padding,
                        groups=child.groups,
                        padding_mode=child.padding_mode,
                        dilation=child.dilation,
                        bias=True,
                        qconfig=child.qconfig
                        )
                        tmp.weight.data = child.weight
                        tmp.bias = child.bias
                        setattr(module, name, tmp)
                        #getattr(module,name, tmp).bias = child.bias



                    if isinstance(child, ConvBnReLU1d):
                        tmp = ConvReLU1d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size, 
                        stride=child.stride,
                        padding=child.padding,
                        groups=child.groups,
                        padding_mode=child.padding_mode,
                        bias=True,
                        dilation=child.dilation,
                        qconfig=child.qconfig)
                        tmp.weight.data = child.weight
                        tmp.bias = child.bias
                        setattr(module, name, tmp)
                        #getattr(module,name, tmp).bias = child.bias
                        #getattr(module,name, tmp).weight.data = child.weight
                        

            device = pl_module.device
            replace_modules(pl_module)
            pl_module.to(device=device) # otherwise cuda error

            # get frequencies
            ws = []
            for name, module in pl_module.named_modules():
                if hasattr(module, "weight") and module.weight != None:
                    ws = np.append(ws, module.weight.data.cpu().detach().numpy())
            frq = Counter(ws.tolist())     



            ##### Testing manipulation of weights with backward hook ###
            # 1. Idea: hold most frequent weights with hook so that they cannot change and accumulate
            # 2. Idea: to shift the rarest elements (with longest Bitcode) to another value
            # 3. Idea: use clustering
            # 4. Idea: depending on gradient, hold value with hook or not

            max_key = max(frq, key=frq.get) # most frequent element
            min_key = min(frq, key=frq.get) # key of rarest element

            #second_max_key = list(dict(sorted(frq.items(), key=lambda item: item[1])))[-2]
            #print(dict(sorted(frq.items(), key=lambda item: item[1])))

            #import collections
            #import bisect
            #frq = collections.OrderedDict(sorted(frq.items()))
            #min_v = list(frq)[bisect.bisect_left(list(frq.keys()), min_key)-1] # neighbour of min_key
            

            def test_hook(module, grad_input, grad_output):
                with torch.no_grad():
                    #module.weight.data = torch.tensor(module.weight.data*0.0) 
                    grad_output = grad_input
                    grad_output[module.weight==max_key] = 0
                    #module.weight.data[module.weight==max_key] = torch.tensor(max_key).to(device=device)
                    #module.weight.data[module.weight==min_key] = torch.tensor(min_v).to(device=device)


            for name, module in pl_module.named_modules():
                if hasattr(module, "weight") and module.weight != None:
                    #module.weight.data[module.weight==min_key] = torch.tensor(min_v).to(device=device)
                    module.register_backward_hook(test_hook)
                    # module.register_full_backward_hook(test_hook) # runtime error in reduction.py (Output 0 of BackwardHookFunctionBackward is a view and is being modified inplace.)
                    # module.register_module_full_backward_hook(test_hook) # 'Conv1d' object has no attribute 'register_module_full_backward_hook'
 