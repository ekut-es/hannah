from collections import namedtuple
from torch.quantization.fake_quantize import FakeQuantize 
from torch.quantization.observer import MovingAverageMinMaxObserver, ObserverBase
import torch
import torch.nn as nn 

# FIXME: accumulator is not used at the moment
QConfig = namedtuple("QConfig", ["activation", "weight", "bias"])

class GlobalObserverStatistics(nn.Module):
    def __init__(self):
        super().__init__()
        min_val = torch.Tensor([float('-inf')])
        max_val = torch.Tensor([float('inf')])
        self.register_buffer('min_val', min_val)
        self.register_buffer('max_val', max_val)


class GlobalMovingAverageMinMaxObserver(ObserverBase):
    def __init__(self, global_statistics=None, quant_min=-128, quant_max=127):
        super().__init__(torch.qint32)
        self.global_statistics = global_statistics
        self.qscheme = torch.per_tensor_symmetric
        self.averaging_constant = 0.00001
        self.quant_min = quant_min
        self.quant_max = quant_max

    @torch.jit.export
    def forward(self, x):
        x_detached = x.detach()

        min_val_cur = torch.argmin(x_detached)
        max_val_cur = torch.argmax(x_detached)

        min_val = self.global_statistics.min_val
        max_val = self.global_statistics.max_val

        if min_val == float('-inf') and max_val == float('inf'): 
            min_val = min_val_cur
            max_val = max_val_cur
        else: 
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)

        self.global_statistics.min_val.resize_(min_val.shape)
        self.global_statistics.max_val.resize_(max_val.shape)
        self.global_statistics.min_val.copy_(min_val)
        self.global_statistics.max_val.copy_(max_val)

        return x

    @torch.jit.export
    def calculate_qparams(self):
        
        device = self.global_statistics.min_val.device
        min_val = self.global_statistics.min_val.clone().detach()
        max_val = self.global_statistics.max_val.clone().detach()
        zero_points = torch.Tensor([0], device=device)
        if min_val < 0.0:
            max_val = torch.max(-min_val, max_val)

        quant_factor = (float(self.quant_max + 1  - self.quant_min) / 2)
        # TODO: use max_val
        scales = torch.tensor([1.0]) / quant_factor
        #print(scales)
        #scales = torch.Tensor([1/128], device=device)
        

        return scales, zero_points

def get_trax_qat_qconfig(config):
    bits_bias = config.bw_b
    bits_activation = config.bw_f
    bits_weight = config.bw_f

    global_statistics_act = GlobalObserverStatistics()
    global_statistics_weight = GlobalObserverStatistics()
    global_statistics_bias = GlobalObserverStatistics()

    qconfig = QConfig(
        FakeQuantize.with_args(observer=GlobalMovingAverageMinMaxObserver, quant_min=-2**(bits_activation-1), quant_max=2**(bits_activation-1)-1, global_statistics=global_statistics_act),
        FakeQuantize.with_args(observer=GlobalMovingAverageMinMaxObserver, quant_min=-2**(bits_weight-1), quant_max=2**(bits_weight-1)-1, global_statistics=global_statistics_weight),
        FakeQuantize.with_args(observer=GlobalMovingAverageMinMaxObserver, quant_min=-2**(bits_bias-1), quant_max=2**(bits_bias-1)-1, global_statistics=global_statistics_bias),
    )

    return qconfig


