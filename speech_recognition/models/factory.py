""" A neural network model factory

It allows us to construct quantized and unquantized versions of the same network,
allows to explore implementation alternatives using a common neural network construction
interface.
"""

from dataclasses import dataclass
from speech_recognition.models.act import DummyActivation
from typing import Union, Optional, List

from torch import nn
import torch.quantization as tqant
from . import qat


@dataclass
class ReductionConfig:
    # one of add, concat
    type: str = "add"


@dataclass
class NormConfig:
    # Currently only bn is supported
    type: str


@dataclass
class BNConfig(NormConfig):
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
class MinorBlockConfig:
    target: str
    out_channels: int
    kernel_size: int
    stride: int


@dataclass
class ConvLayerConfig(MinorBlockConfig):
    target: str = "conv1d"
    out_channels: int = 32
    kernel_size: int = 3
    stride: int = 0
    padding: bool = True
    dilation: int = 0
    groups: int = 1
    padding_mode: str = "zeros"
    norm: Union[NormConfig] = False
    act: Union[ActConfig] = False


@dataclass
class MajorBlockConfig:
    out_channels: int = 32
    blocks: List[MinorBlockConfig] = []


@dataclass
class NetworkConfig:
    norm: Optional[NormConfig] = BNConfig()
    act: Optional[ActConfig] = ActConfig()
    qconfig: Optional[tqant.QConfig] = ActConfig()
    blocks: List[MajorBlockConfig] = []


class ModelFactory:
    def __init__(self,) -> None:
        self.norm = None
        self.act = None
        self.qconfig = None

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
                if norm.type == "bn":
                    norm_module = nn.BatchNorm1d(
                        out_channels,
                        eps=norm.eps,
                        momentum=norm.momentum,
                        affine=norm.affine,
                    )
                else:
                    raise Exception(f"Unknown normalization module: {norm}")
                layers.append(norm_module)

            act_module = DummyActivation()
            if act:
                if act.type == "relu":
                    act_module = nn.ReLU()
                elif act.type == "elu":
                    act_module = nn.ELU(alpha=act.alpha)
                elif act.type == "hardtanh":
                    act_module = nn.Hardtanh(min_val=act.min_val, max_val=act.max_val)
                else:
                    raise Exception(f"Unknown activation config {act}")

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

    def residual(self):
        pass

    def parallel(self):
        pass

    def structured(self):
        pass

    def major(self, config: MajorBlockConfig):
        pass

    def network(self, network_config: NetworkConfig):
        self.norm = network_config.norm
        self.act = network_config.act
        self.qconfig = network_config.qconfig

        model = nn.Sequential()
        for block in network_config.blocks:
            self.model.append(self.major(block))

        return model
