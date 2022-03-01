from asyncio.log import logger
import torch
import torch.nn as nn
import numpy as np

from pytorch_lightning.callbacks import Callback
from ..models.factory.qat import ConvBn1d, Conv1d, ConvBnReLU1d, ConvReLU1d
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix


def clustering(params, inertia, cluster):
    sparse_matrix = csr_matrix(params)
    kmeans = KMeans(n_clusters=cluster, n_init=1, init='k-means++', algorithm="full", random_state=1234)
    kmeans.fit(sparse_matrix.reshape(-1, 1))
    centers = kmeans.cluster_centers_.reshape(-1)
    inertia += kmeans.inertia_
    return centers, inertia


class kMeans(Callback):
    def __init__(self, compress_after, cluster):
        self.compress_after = compress_after
        self.cluster = cluster

    def on_fit_end(self, trainer, pl_module):
        with torch.no_grad():
            for module in pl_module.modules():
                if hasattr(module, "scaled_weight"):
                    module.weight.data = module.scaled_weight
                    if not isinstance(module, nn.Linear):
                        bias_shape = [1] * len(module.weight.shape)
                        bias_shape[1] = -1
                        bias = torch.zeros(module.out_channels, device=module.weight.device)
                        bias = module.bias_fake_quant((bias - module.bn.running_mean) * module.scale_factor + module.bn.bias)  # .reshape(bias_shape) #.view(-1, 1, 1) #.reshape(bias_shape)
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

        device = pl_module.device
        replace_modules(pl_module)
        pl_module.to(device=device)  # otherwise cuda error
        inertia = 0
        for name, module in pl_module.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                params = module.weight.data.cpu().numpy().flatten()
                centers, inertia = clustering(params, inertia, self.cluster)

                # Returns center that is closest to given value x
                def replace_values_by_centers(x):
                    if x == 0.0: # required if pruning is used
                        return x
                    else:
                        i = (np.abs(centers - x)).argmin()
                        return centers[i]
                module.weight.data = module.weight.data.cpu().apply_(replace_values_by_centers)  # _ symbolizes inplace function, tensor moved to cpu, since apply_() only works that way
                module.to(device=device)  # move from cpu to gpu
        print('Clustering error: ', inertia)

    def on_epoch_end(self, trainer, pl_module):
        inertia = 0
        if trainer.current_epoch % 2 == 0 and trainer.current_epoch < self.compress_after-1 and trainer.callback_metrics['val_accuracy'].item() > 0.9:
            logger.info('Training validation accuracy: %s', trainer.callback_metrics['val_accuracy'].item())
            device = pl_module.device
            for module in pl_module.modules():
                if hasattr(module, "weight") and module.weight is not None:
                    w = module.weight.data.cpu().numpy().flatten()
                    centers, inertia = clustering(w, inertia, self.cluster)

                    def replace_values_by_centers(x):
                        if x == 0.0:
                            return x
                        else:
                            i = (np.abs(centers - x)).argmin()
                            return centers[i]
                    module.weight.data = module.weight.data.cpu().apply_(replace_values_by_centers)
                    module.to(device=device)
            logger.info('Clustering error: %s', inertia)
