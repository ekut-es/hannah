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


    def on_fit_end(self, trainer, pl_module):
        
        if True: #if trainer.current_epoch % 3 == 0: #if trainer.current_epoch > self.compress_after-2:
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


            c = []
            #### Testing kmeans clustering ###
            for module in pl_module.modules():
                if hasattr(module, "weight") and module.weight != None:
                    params = module.weight.data.cpu().numpy().flatten()
                    sparse_matrix = csr_matrix(params)
                    max_value = max(sparse_matrix.data)
                    min_value = min(sparse_matrix.data)
                    range_cluster = np.linspace(min_value, max_value, num=4)

                    # KMeans applied to each layer
                    kmeans = KMeans(n_clusters=len(range_cluster), init=range_cluster.reshape(-1,1), n_init=1, algorithm="full")
                    kmeans.fit(sparse_matrix.reshape(-1,1))
                    centers = kmeans.cluster_centers_.reshape(-1)
                    #c.append(centers)
                    c.extend(centers)
                    # Returns center that is closest to given value x
                    '''def replace_values_by_centers(x):
                        i = (np.abs(centers - x)).argmin() 
                        return centers[i] 
                    module.weight.data = module.weight.data.cpu().apply_(replace_values_by_centers) #_ symbolizes inplace function, tensor moved to cpu, since apply_() only works that way
                    module.to(device=device) # move from cpu to gpu'''
            '''params = []
            for module in pl_module.modules():
                if hasattr(module, "weight") and module.weight != None:
                    params.append(module.weight.data.cpu().numpy().flatten())
                    sparse_matrix = csr_matrix(params)'''
            # Perform KMeans clustering again on all center coordinates
            m = csr_matrix(c) 
            min_value = min(m.data)
            max_value = max(m.data)
            range_cluster = np.linspace(min_value, max_value, num=18)
            kmeans = KMeans(n_clusters=len(range_cluster), init=range_cluster.reshape(-1,1), n_init=1, algorithm="full")
            kmeans.fit(m.data.reshape(-1,1))
            c = kmeans.cluster_centers_.reshape(-1)
            def replace_values_by_centers(x):
                '''m = []
                for j in range(len(c)):
                    i = (np.abs(c[j] - x)).argmin()
                    m.append(c[j][i])
                center = (np.abs(np.asarray(m) - x)).argmin() 
                return m[center]'''
                i = (np.abs(np.asarray(c) - x)).argmin() 
                return c[i] 
            for module in pl_module.modules():
                if hasattr(module, "weight") and module.weight != None:
                    module.weight.data = module.weight.data.cpu().apply_(replace_values_by_centers) #_ symbolizes inplace function, tensor moved to cpu, since apply_() only works that way
                    module.to(device=device)

        
    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == 7: #if trainer.current_epoch >= self.compress_after-2: #
            device = pl_module.device
            c = []
            #### Testing kmeans clustering ###
            for module in pl_module.modules():
                if hasattr(module, "weight") and module.weight != None:
                    w = module.weight.data.cpu().numpy().flatten()
                    m = csr_matrix(w) 
                    min_value = min(m.data)
                    max_value = max(m.data)
                    range_cluster = np.linspace(min_value, max_value, num=4)
                    kmeans = KMeans(n_clusters=len(range_cluster), init=range_cluster.reshape(-1,1), n_init=1, algorithm="full")
                    #kmeans.fit(w.reshape(-1,1))
                    kmeans.fit(m.data.reshape(-1,1))
                    centers = kmeans.cluster_centers_.reshape(-1)
                    #c.append(centers)
                    c.extend(centers)
                    def replace_values_by_centers(x):
                        i = (np.abs(centers - x)).argmin() 
                        return centers[i] # returns center that is closest to given value x
                    #module.apply(replace_values_by_centers)
                    #module.weight.data = module.weight.data.cpu().apply_(replace_values_by_centers) #_ symbolizes inplace function, tensor moved to cpu, since apply_() only works that way
                    #module.to(device=device) # move from cpu to gpu
            
            # Perform KMeans clustering again on all center coordinates
            m = csr_matrix(c) 
            min_value = min(m.data)
            max_value = max(m.data)
            range_cluster = np.linspace(min_value, max_value, num=18)
            kmeans = KMeans(n_clusters=len(range_cluster), init=range_cluster.reshape(-1,1), n_init=1, algorithm="full")
            kmeans.fit(m.data.reshape(-1,1))
            c = kmeans.cluster_centers_.reshape(-1)

            def replace_values_by_centers(x):
                i = (np.abs(np.asarray(c) - x)).argmin() 
                return c[i] 

            for module in pl_module.modules():
                if hasattr(module, "weight") and module.weight != None:
                    module.weight.data = module.weight.data.cpu().apply_(replace_values_by_centers) #_ symbolizes inplace function, tensor moved to cpu, since apply_() only works that way
                    module.to(device=device)
                    