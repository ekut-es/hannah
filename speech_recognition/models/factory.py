""" A neural network model factory

It allows us to construct quantized and unquantized versions of the same network,
allows to explore implementation alternatives using a common neural network construction
interface.
"""

from dataclasses import dataclass
from typing import Union, Optional

from torch import nn
import torch.quantization as tqant
from . import qat


@dataclass
class BNConfig:
    type: str = "bn"
    eps: float = 1e-05
    momentum: float = 0.1
    affine: bool = True


@dataclass
class ActConfig:
    type: str = "relu"


@dataclass
class ELUConfig(ActConfig):
    type: str = "elu"
    alpha: float = 1.0


@dataclass
class HardtanhConfig(ActConfig):
    type: str = "hardtanh"
    min_val: float = -1.0
    max_val: float = 1.0


@dataclass
class ConvLayerConfig(ActConfig):
    target: str = "conv1d"
    out_channels: int = 32
    kernel_size: int = 3
    stride: int = 0
    padding: bool = True
    dilation: int = 0
    groups: int = 1
    padding_mode: str = "zeros"
    norm: Union[BNConfig] = False
    act: Union[ActConfig] = False


class ModelFactory:
    def __init__(
        self,
        norm: BNConfig = BNConfig(),
        act: ActConfig = ActConfig(),
        qconfig: Optional[tqant.QConfig] = None,
    ) -> None:
        self.norm = norm
        self.act = act
        self.qconfig = qconfig

    def conv1d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 0,
        padding: Union[int, bool] = True,
        dilation: int = 0,
        groups: int = 1,
        padding_mode: str = "zeros",
        norm: Union[BNConfig, bool] = False,
        act: Union[ActConfig, bool] = False,
        qconfig: Union[tqant.QConfig, bool] = False,
    ) -> None:
        if padding is True:
            # Calculate full padding
            padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        if padding is False:
            padding = 0

        if norm is True:
            norm = self.norm

        if act is True:
            act = self.act

        if qconfig is True:
            qconfig = self.qconfig

        if not qconfig:
            layers = nn.Sequential()
            conv_module = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                padding_mode,
            )
            layers.append(conv_module)
            if norm:
                norm_module = self.getattr(norm.type)(**norm)
                layers.append(norm_module)

            if act:
                act_module = self.getattr(act.type)(**act)
                layers.append(act_module)
        elif isinstance(qconfig, tqant.QConfig):
            if norm and act:
                layers = qat.ConvBnReLU1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    padding_mode=padding_mode,
                    eps=norm.eps,
                    momentum=norm.momentum,
                    qconfig=qconfig,
                )
            elif norm:
                layers = qat.ConvBn1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    padding_mode=padding,
                    eps=norm.eps,
                    momentum=norm.momentum,
                    qconfig=qconfig,
                )
            elif act:
                layers = qat.ConvReLU1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    padding_mode=padding_mode,
                    eps=norm.eps,
                    momentum=norm.momentum,
                    qconfig=qconfig,
                )
            else:
                layers = qat.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    padding_mode=padding_mode,
                    qconfig=qconfig,
                )
        else:
            raise Exception(f"Qconfig: {qconfig} is not supported for conv1d")

        return layers
