#
# Copyright (c) 2024 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
""" A neural network model factory

It allows us to construct quantized and unquantized versions of the same network,
allows to explore implementation alternatives using a common neural network construction
interface.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf
from torch.nn.modules.container import Sequential
from torch.nn.modules.linear import Identity

from hannah.models.factory import pooling

from . import qat
from .act import DummyActivation
from .network import ConvNet
from .reduction import ReductionBlockAdd, ReductionBlockConcat


@dataclass
class NormConfig:
    """ """

    # Currently only bn is supported
    target: str = MISSING


@dataclass
class BNConfig(NormConfig):
    """ """

    target: str = "bn"
    eps: float = 1e-05
    momentum: float = 0.1
    affine: bool = True


@dataclass
class ActConfig:
    """ """

    target: str = "relu"


@dataclass
class ELUConfig(ActConfig):
    """ """

    target: str = "elu"
    alpha: float = 1.0


@dataclass
class HardtanhConfig(ActConfig):
    """ """

    target: str = "hardtanh"
    min_val: float = -1.0
    max_val: float = 1.0


@dataclass
class MinorBlockConfig:
    """ """

    target: str = "conv1d"
    "target Operation"
    parallel: bool = False
    "execute block in parallel with preceding block"
    out_channels: int = 32
    "number of output channels"
    kernel_size: Any = 3  # Union[int, Tuple[int, ...], Tuple[int, ...]]
    "kernel size of this Operation (if applicable)"
    stride: Any = 1  # Union[None, int, Tuple[int, ...], Tuple[int, ...]]
    "stride for this operation use"
    padding: bool = True
    "use padding for this operation (padding will always try to keep input dimensions / stride)"
    dilation: Any = 1  # Union[int, Tuple[int, ...], Tuple[int, ...]]
    "dilation factor to use for this operation"
    groups: int = 1
    "number of groups for this operation"
    norm: Any = False  # Union[NormConfig, bool]
    "normalization to use (true uses networks default configs)"
    act: Any = False  # Union[ActConfig, bool]
    "activation to use (true uses default configs)"
    upsampling: Any = 1.0
    "Upsampling factor for mbconv layers"
    bias: bool = False
    "use bias for this operation"
    out_quant: bool = True
    "use output quantization for this operation"
    kernel_per_layer: int = 1


@dataclass
class MajorBlockConfig:
    """ """

    target: str = "residual"
    blocks: List[MinorBlockConfig] = field(default_factory=list)
    reduction: str = "add"
    stride: Optional[int] = None  # Union[None, int, Tuple[int, ...], Tuple[int, ...]]
    last: bool = False  # Indicates wether this block is the last reduction block


@dataclass
class LinearConfig:
    """ """

    outputs: int = 128
    norm: Any = False  # Union[bool, NormConfig]
    act: Any = False  # Union[bool, ActConfig]
    out_quant: bool = False
    qconfig: Optional[Any] = None


@dataclass
class NetworkConfig:
    """ """

    _target_: str = "hannah.models.factory.create_cnn"
    name: str = MISSING
    norm: Optional[NormConfig] = field(default_factory=BNConfig)
    act: Optional[ActConfig] = field(default_factory=ActConfig)
    qconfig: Optional[Any] = None
    conv: List[MajorBlockConfig] = field(default_factory=list)
    linear: List[LinearConfig] = field(default_factory=list)
    dropout: float = 0.5


class NetworkFactory:
    """ """

    def __init__(
        self,
    ) -> None:
        self.default_norm = None
        self.default_act = None
        self.default_qconfig = None

    def act(self, config: ActConfig) -> nn.Module:
        """

        Args:
          config: ActConfig:
          config: ActConfig:

        Returns:

        """
        act_module: nn.Module
        if config.target == "relu":
            act_module = nn.ReLU()
        elif config.target == "elu":
            act_module = nn.ELU(alpha=config.alpha)  # type: ignore
        elif config.target == "hardtanh":
            act_module = nn.Hardtanh(min_val=config.min_val, max_val=config.max_val)  # type: ignore
        elif config.target == "sigmoid":
            act_module = nn.Sigmoid()
        elif config.target == "tanh":
            act_module = nn.Tanh()
        else:
            raise Exception(f"Unknown activation config {config}")

        return act_module

    def max_pool1d(
        self,
        input_shape: Tuple[int, ...],
        kernel_size: int,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...], bool] = True,
        dilation: Union[int, Tuple[int, ...]] = 1,
        norm: Union[BNConfig, bool] = False,
        act: Union[ActConfig, bool] = False,
        bias: bool = False,
    ) -> Tuple[Tuple[int, ...], nn.Module]:
        in_channels = input_shape[1]
        in_len = input_shape[2]
        output_shape = (
            input_shape[0],
            in_channels,
            self._calc_spatial_dim(in_len, kernel_size, stride, padding, dilation),
        )

        if padding is True:
            # Calculate full padding
            padding = self._padding(kernel_size, stride, dilation)
        if padding is False:
            padding = 0

        return output_shape, nn.MaxPool1d(
            kernel_size, stride=stride, padding=padding, dilation=dilation
        )

    def conv2d(
        self,
        input_shape: Tuple[int, ...],
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...], bool] = True,
        dilation: Union[int, Tuple[int, ...]] = 0,
        groups: int = 1,
        norm: Union[BNConfig, bool] = False,
        act: Union[ActConfig, bool] = False,
        bias: bool = False,
    ) -> Any:
        """

        Args:
          input_shape: Tuple[int:
          int:
          int]:
          out_channels: int:
          kernel_size: Union[int:
          Tuple[int, ...]]: (Default value = 1)
          stride: Union[int:
          padding: Union[int:
          Tuple[int, ...]:
          bool]: (Default value = False)
          dilation: int:  (Default value = 0)
          groups: int:  (Default value = 1)
          norm: Union[BNConfig:
          act: Union[ActConfig:
          bias: bool:  (Default value = False)
          input_shape: Tuple[int:
          out_channels: int:
          kernel_size: Union[int:
          stride: Union[int:
          padding: Union[int:
          dilation: int:  (Default value = 0)
          groups: int:  (Default value = 1)
          norm: Union[BNConfig:
          act: Union[ActConfig:
          bias: bool:  (Default value = False)

        Returns:

        """
        in_channels = input_shape[1]

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        if isinstance(stride, int):
            stride = (stride, stride)

        if padding is True:
            # Calculate full padding
            padding_x = self._padding(kernel_size[0], stride[0], dilation[0])
            padding_y = self._padding(kernel_size[1], stride[1], dilation[1])
            padding = (padding_x, padding_y)
        if padding is False:
            padding = (0, 0)
        if isinstance(padding, int):
            padding = (padding, padding)

        output_shape = (
            input_shape[0],
            out_channels,
            self._calc_spatial_dim(
                input_shape[2], kernel_size[0], stride[0], padding[0], dilation[0]
            ),
            self._calc_spatial_dim(
                input_shape[3], kernel_size[1], stride[1], padding[1], dilation[1]
            ),
        )

        if norm is True:
            norm = self.default_norm

        if act is True:
            act = self.default_act

        qconfig = self.default_qconfig

        if not qconfig:
            layers = []
            conv_module = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
            layers.append(conv_module)
            if norm:
                if norm.target == "bn":
                    norm_module = nn.BatchNorm2d(
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
                act_module = self.act(act)
            layers.append(act_module)

            layers = nn.Sequential(*layers)

        else:
            if norm and act:
                layers = qat.ConvBnReLU2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    dilation=dilation,
                    groups=groups,
                    eps=norm.eps,
                    momentum=norm.momentum,
                    qconfig=qconfig,
                )
            elif norm:
                layers = qat.ConvBn2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    dilation=dilation,
                    groups=groups,
                    eps=norm.eps,
                    momentum=norm.momentum,
                    qconfig=qconfig,
                )
            elif act:
                layers = qat.ConvReLU2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    dilation=dilation,
                    groups=groups,
                    eps=norm.eps,
                    momentum=norm.momentum,
                    qconfig=qconfig,
                )
            else:
                layers = qat.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    bias=bias,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    qconfig=qconfig,
                )

        return output_shape, layers

    def mbconv1d(
        self,
        input_shape: Tuple[int, ...],
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        padding: Union[int, bool] = True,
        bias=False,
        upsampling: float = 1.0,
        groups: int = 1,
        norm: Union[BNConfig, bool] = False,
        act: Union[ActConfig, bool] = False,
    ):
        """

        Args:
          input_shape: Tuple[int:
          int:
          int]:
          out_channels: int:
          kernel_size: int:
          dilation: int:  (Default value = 1)
          stride: int:  (Default value = 1)
          padding: Union[int:
          bool]: (Default value = False)
          bias: (Default value = False)
          upsampling: float:  (Default value = 1.0)
          groups: int:  (Default value = 1)
          norm: Union[BNConfig:
          act: Union[ActConfig:
          input_shape: Tuple[int:
          out_channels: int:
          kernel_size: int:
          dilation: int:  (Default value = 1)
          stride: int:  (Default value = 1)
          padding: Union[int:
          upsampling: float:  (Default value = 1.0)
          groups: int:  (Default value = 1)
          norm: Union[BNConfig:
          act: Union[ActConfig:

        Returns:

        """

        up_channels = int(out_channels * upsampling)

        output_shape, upsample_conv = self.conv1d(
            input_shape, up_channels, 1, norm=norm, act=act
        )
        output_shape, grouped_conv = self.conv1d(
            output_shape,
            up_channels,
            kernel_size,
            stride=stride,
            padding=True,
            bias=bias,
            groups=up_channels,
            norm=False,
            act=False,
            dilation=dilation,
        )
        output_shape, downsample_conv = self.conv1d(
            output_shape, out_channels, 1, act=act, norm=norm
        )

        mbconv = nn.Sequential(upsample_conv, grouped_conv, downsample_conv)

        return output_shape, mbconv

    def conv1d(
        self,
        input_shape: Tuple[int, ...],
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        bias: bool = False,
        padding: Union[int, bool, Tuple[int]] = True,
        dilation: Union[int, Tuple[int]] = 1,
        groups: Union[int, Tuple[int]] = 1,
        norm: Union[BNConfig, bool] = False,
        act: Union[ActConfig, bool] = False,
        out_quant: bool = True,
    ) -> Tuple[Tuple[int, ...], nn.Module]:
        """

        Args:
          input_shape: Tuple[int:
          int:
          int]:
          out_channels: int:
          kernel_size: int:
          stride: int:  (Default value = 1)
          bias: bool:  (Default value = False)
          padding: Union[int:
          bool]: (Default value = False)
          dilation: int:  (Default value = 1)
          groups: int:  (Default value = 1)
          norm: Union[BNConfig:
          act: Union[ActConfig:
          out_quant: bool:  (Default value = True)
          input_shape: Tuple[int:
          out_channels: int:
          kernel_size: int:
          stride: int:  (Default value = 1)
          bias: bool:  (Default value = False)
          padding: Union[int:
          dilation: int:  (Default value = 1)
          groups: int:  (Default value = 1)
          norm: Union[BNConfig:
          act: Union[ActConfig:
          out_quant: bool:  (Default value = True)

        Returns:

        """

        in_channels = input_shape[1]

        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        if isinstance(dilation, tuple):
            dilation = dilation[0]

        if padding is True:
            # Calculate full padding
            padding = self._padding(kernel_size, stride, dilation)

        if isinstance(padding, tuple):
            padding = padding[0]

        if padding is False:
            padding = 0

        if norm is True:
            norm = self.default_norm

        if act is True:
            act = self.default_act

        qconfig = self.default_qconfig

        output_shape = (
            input_shape[0],
            out_channels,
            self._calc_spatial_dim(
                input_shape[2], kernel_size, stride, padding, dilation
            ),
        )

        if not qconfig:
            layers = []
            conv_module = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                dilation=dilation,
                groups=groups,
            )
            layers.append(conv_module)
            if norm:
                if norm.target == "bn":
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
                act_module = self.act(act)

            layers.append(act_module)
            layers = nn.Sequential(*layers)

        else:
            if act and act.target != "relu":
                logging.warning(
                    "Activation function '{act.target}' is not supported for quantized networks, replacing by 'relu'"
                )
            if norm and act:
                layers = qat.ConvBnReLU1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    bias=bias,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    eps=norm.eps,
                    momentum=norm.momentum,
                    qconfig=qconfig,
                    out_quant=out_quant,
                )
            elif norm:
                layers = qat.ConvBn1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    dilation=dilation,
                    groups=groups,
                    eps=norm.eps,
                    momentum=norm.momentum,
                    qconfig=qconfig,
                    out_quant=out_quant,
                )
            elif act:
                layers = qat.ConvReLU1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    dilation=dilation,
                    groups=groups,
                    qconfig=qconfig,
                    out_quant=out_quant,
                )
            else:
                layers = qat.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    dilation=dilation,
                    groups=groups,
                    qconfig=qconfig,
                    out_quant=out_quant,
                )
        return output_shape, layers

    def minor(self, input_shape, config: MinorBlockConfig, major_stride=None):
        """

        Args:
          input_shape:
          config: MinorBlockConfig:
          major_stride: (Default value = None)
          config: MinorBlockConfig:

        Returns:

        """
        assert config.out_channels % config.groups == 0
        assert input_shape[1] % config.groups == 0

        if major_stride is not None:
            config.stride = major_stride

        if config.target == "conv1d":
            return self.conv1d(
                input_shape,
                config.out_channels,
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=config.padding,
                dilation=config.dilation,
                groups=config.groups,
                act=config.act,
                norm=config.norm,
                bias=config.bias,
                out_quant=config.out_quant,
            )

        elif config.target == "mbconv1d":
            return self.mbconv1d(
                input_shape,
                config.out_channels,
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=config.padding,
                bias=config.bias,
                dilation=config.dilation,
                groups=config.groups,
                act=config.act,
                norm=config.norm,
                upsampling=config.upsampling,
            )
        elif config.target == "conv2d":
            return self.conv2d(
                input_shape,
                config.out_channels,
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=config.padding,
                dilation=config.dilation,
                groups=config.groups,
                act=config.act,
                norm=config.norm,
                bias=config.bias,
            )
        elif config.target == "max_pool1d":
            return self.max_pool1d(
                input_shape,
                kernel_size=config.kernel_size,
                padding=config.padding,
                stride=config.stride,
                act=config.act,
                norm=config.norm,
            )

        elif config.target == "max_pool2d":
            return self.max_pool2d(
                input_shape,
                config.out_channels,
                kernel_size=config.kernel_size,
                padding=config.padding,
                stride=config.stride,
                act=config.act,
                norm=config.norm,
            )

        elif config.target == "avg_pool1d":
            raise NotImplementedError(
                "Minor block config avg_pool1d has not been implemented yet"
            )
        elif config.target == "avg_pool2d":
            raise NotImplementedError(
                "Minor block config avg_pool2d has not been implemented yet"
            )
        else:
            raise Exception(f"Unknown minor block config {config}")
        """ Depthwise separable convolution can be splitted into dephtwise convolution first followed by pointwise convolution.
        if config.target == "conv1d":
            #breakpoint()
            depthwise_conv = self.conv1d(
                input_shape,
                out_channels=input_shape[1],#*config.kernel_per_layer, # adjust number of output channels
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=config.padding,
                dilation=config.dilation,
                groups=input_shape[1], # number of input channels
                act=config.act,
                norm=config.norm,
                bias=config.bias,
                out_quant=config.out_quant,
            )
            #print(input_shape[1]*config.kernel_per_layer)
            pointwise_conv = self.conv1d(
                input_shape, #input_shape[1]*config.kernel_per_layer# number of output channels of depthwise convolution
                config.out_channels, # out_channels as for normal convolution
                kernel_size=1, # must be 1, since convolution through every point
                stride=config.stride,
                padding=config.padding,
                dilation=config.dilation,
                groups=config.groups,
                act=config.act,
                norm=config.norm,
                bias=config.bias,
                out_quant=config.out_quant,
            )
            return nn.Sequential(depthwise_conv, pointwise_conv) """

    def _build_chain(
        self,
        input_shape: Tuple[int, ...],
        block_configs: List[MinorBlockConfig],
        major_stride: Optional[Any],
    ) -> List[Tuple[Tuple[int, ...], Sequential]]:
        """

        Args:
          input_shape: Tuple[int:
          int:
          int]:
          block_configs: List[MinorBlockConfig]:
          major_stride: Optional[Any]:
          input_shape: Tuple[int:
          block_configs: List[MinorBlockConfig]:
          major_stride: Optional[Any]:

        Returns:

        """
        block_input_shape = input_shape
        result_chain = []
        for block_config in block_configs:
            result_chain.append(
                self.minor(block_input_shape, block_config, major_stride)
            )
            block_input_shape = result_chain[-1][0]
            if major_stride is not None:
                major_stride = 1

        return result_chain

    def _build_reduction(
        self,
        reduction: str,
        input_shape: Tuple[int, ...],
        *input_chains: List[Tuple[Tuple[int, ...], Sequential]],
        reduction_quant: bool = False,
    ) -> Tuple[Tuple[int, ...], Union[ReductionBlockConcat, Sequential]]:
        """

        Args:
          reduction: str:
          input_shape: Tuple[int:
          int:
          int]:
          *input_chains: List[Tuple[Tuple[int:
          Sequential]]:
          reduction_quant: bool:  (Default value = False)
          reduction: str:
          input_shape: Tuple[int:
          *input_chains: List[Tuple[Tuple[int:
          reduction_quant: bool:  (Default value = False)

        Returns:

        """
        output_shapes = []
        for chain in input_chains:
            output_shapes.append(chain[-1][0] if len(chain) > 0 else input_shape)

        minimum_output_shape = tuple(map(min, zip(*output_shapes)))
        maximum_output_shape = tuple(map(max, zip(*output_shapes)))
        target_output_shape = maximum_output_shape[:2] + minimum_output_shape[2:]

        for output_shape, chain in zip(output_shapes, input_chains):
            if output_shape != target_output_shape:
                logging.info("Adding reduction cell")

                groups = output_shape[1]
                output_channels = output_shape[1]

                if reduction == "add":
                    output_channels = target_output_shape[1]
                    groups = 1  # For now do not use grouped convs for resampling: math.gcd(output_channels, groups)

                stride = tuple(
                    (
                        math.ceil(y / x)
                        for x, y in zip(target_output_shape[2:], output_shape[2:])
                    )
                )

                dimension = len(stride)
                if dimension == 1:
                    downsampler = self.conv1d(
                        input_shape=output_shape,
                        out_channels=output_channels,
                        kernel_size=1,
                        stride=stride,
                        padding=True,
                        dilation=1,
                        groups=groups,
                        norm=True,
                        act=False,
                        bias=False,
                    )
                elif dimension == 2:
                    downsampler = self.conv2d(
                        input_shape=input_shape,
                        out_channels=output_channels,
                        kernel_size=1,
                        stride=stride,
                        padding=True,
                        dilation=1,
                        groups=groups,
                        norm=True,
                        act=False,
                        bias=False,
                    )

                chain.append(downsampler)

        inputs = [nn.Sequential(*[x[1] for x in chain]) for chain in input_chains]
        if reduction == "add":
            reduction = ReductionBlockAdd(*inputs)
            if reduction_quant:
                reduction_quant = self.identity()
                return target_output_shape, nn.Sequential(reduction, reduction_quant)
            else:
                return target_output_shape, nn.Sequential(reduction)
        elif reduction == "concat":
            output_channels = sum((x[1] for x in output_shapes))

            return (
                (target_output_shape[0], output_channels) + target_output_shape[2:],
                ReductionBlockConcat(*inputs),
            )
        else:
            raise Exception(f"Could not find reduction: {reduction}")

    def forward(self, input_shape: Tuple[int, ...], config: MajorBlockConfig):
        """Create a forward neural network block without parallelism

        If parallel is set to [True, False, True, False]

        Input: ------->|---> parallel: False --->  parallel: False ---> | --> output

        Args:
          input_shape: Tuple[int, ...]:
          config: MajorBlockConfig:
          input_shape: Tuple[int, ...]:
          config: MajorBlockConfig:

        Returns:

        """

        main_configs = []

        for block_config in config.blocks:
            main_configs.append(block_config)

        main_configs[-1].out_quant = True

        main_chain = self._build_chain(input_shape, main_configs, config.stride)
        output_shape = main_chain[-1][0]
        major_block = nn.Sequential(*[x[1] for x in main_chain])

        return output_shape, major_block

    def residual(self, input_shape: Tuple[int, ...], config: MajorBlockConfig):
        """Create a neural network block with with residual parallelism

        If parallel is set to [True, False, True, False]
                       |---> parallel: True  --->  parallel: True  ---> |
        Input: ------->|                                                +--->
                       |---> parallel: False --->  parallel: False ---> |

        If the major block does change the output dimensions compared to the input
        and one of the branches does not contain any layers, we infer
        1x1 conv of maximum group size (gcd (input_channels, output_channels)) to do the
        downsampling.

        Args:
          input_shape: Tuple[int, ...]:
          config: MajorBlockConfig:
          input_shape: Tuple[int, ...]:
          config: MajorBlockConfig:

        Returns:

        """

        main_configs = []
        residual_configs = []

        for block_config in config.blocks:
            if block_config.parallel:
                residual_configs.append(block_config)
            else:
                main_configs.append(block_config)

        if len(residual_configs) > len(main_configs):
            residual_configs, main_configs = main_configs, residual_configs

        if len(residual_configs) == 0:
            if config.stride <= 1:
                # If skip connection is empty, use forward block
                return self.forward(input_shape, config)
            else:
                residual_configs.append(
                    MinorBlockConfig(
                        target=main_configs[-1].target,
                        parallel=True,
                        out_channels=main_configs[-1].out_channels,
                        kernel_size=1,
                        padding=True,
                        norm=main_configs[-1].norm,
                        act=True,
                    )
                )

        main_configs[-1].out_quant = False
        main_configs[-1].act = False

        main_chain = self._build_chain(input_shape, main_configs, config.stride)
        residual_chain = self._build_chain(input_shape, residual_configs, config.stride)

        output_shape, major_block = self._build_reduction(
            config.reduction,
            input_shape,
            main_chain,
            residual_chain,
            reduction_quant=True,
        )

        return output_shape, major_block

    def input(self, in_channels: int, config: MajorBlockConfig):
        """Create a neural network block with input parallelism

        If parallel is set to [True, False, True, False]
                        |---> parallel: True  ---> |
                        |---> parallel: True  ---> + -----------------> |
        Input:--------->|                                               +--->
                        |---> parallel: False ---> parallel: False ---> |

        If there are no parallel branches in the network. The major block is
        a standard feed forward layer.

        Args:
          in_channels: int:
          config: MajorBlockConfig:
          in_channels: int:
          config: MajorBlockConfig:

        Returns:

        """

        out_channels = config.out_channels
        block = None
        return out_channels, block

    def full(self, in_channels: int, config: MajorBlockConfig):
        """Create a neural network block with full parallelism

        If parallel is set to [True, False, True, False]
                  |---> parallel: True  ---------------------------------- -|
        Input:--->|                                                         +--->
                  |                           |--> parallel: False --->|    |
                  |---> parallel: False ----> |                        +--->|
                                              |--> parallel: True ---->|

        If there are no parallel blocks the block is a standard feed forward network.

        Args:
          in_channels: int:
          config: MajorBlockConfig:
          in_channels: int:
          config: MajorBlockConfig:

        Returns:

        """
        out_channels = config.out_channels
        block = None
        return out_channels, block

    def major(self, input_shape, config: MajorBlockConfig):
        """

        Args:
          input_shape:
          config: MajorBlockConfig:
          config: MajorBlockConfig:

        Returns:

        """
        if config.target == "residual":
            out_shape, layers = self.residual(input_shape, config)
        elif config.target == "input":
            out_shape, layers = self.input(input_shape, config)
        elif config.target == "parallel":
            out_shape, layers = self.parallel(input_shape, config)
        elif config.target == "forward":
            out_shape, layers = self.forward(input_shape, config)
        else:
            raise Exception("Unknown major block target: {major_block.target}")

        return out_shape, layers

    def linear(self, input_shape, config: LinearConfig):
        """

        Args:
          input_shape:
          config: LinearConfig:
          config: LinearConfig:

        Returns:

        """

        act = config.act
        if act is True:
            act = self.default_act

        norm = config.norm
        if norm is True:
            norm = self.default_norm

        qconfig = self.default_qconfig

        out_shape = (input_shape[0], config.outputs)
        layers: list[nn.Module] = []
        if not qconfig:
            layers.append(nn.Linear(input_shape[1], config.outputs, bias=False))
        else:
            layers.append(
                qat.Linear(
                    input_shape[1],
                    config.outputs,
                    qconfig=qconfig,
                    bias=False,
                    out_quant=config.out_quant,
                )
            )
        if norm:
            layers.append(self.norm(norm))

        if act:
            layers.append(self.act(act))

        layers_sequential = nn.Sequential(*layers)

        return out_shape, layers_sequential

    def identity(self) -> Identity:
        """ """
        qconfig = self.default_qconfig

        if not qconfig:
            identity = nn.Identity()
        else:
            identity = qat.Identity(qconfig=qconfig)

        return identity

    def network(
        self, input_shape, labels: int, network_config: Union[ListConfig, DictConfig]
    ):
        """

        Args:
          input_shape:
          labels: int:
          network_config: NetworkConfig:
          labels: int:
          network_config: NetworkConfig:

        Returns:

        """
        self.default_norm = network_config.norm
        self.default_act = network_config.act
        self.default_qconfig = (
            instantiate(network_config.qconfig) if network_config.qconfig else None
        )

        conv_layers = []
        network_config.conv[-1].last = True
        for block in network_config.conv:
            input_shape, block_model = self.major(input_shape, block)
            conv_layers.append(block_model)
        conv_layers_sequential = nn.Sequential(*conv_layers)

        global_pooling: nn.Module
        if len(input_shape) == 3:
            if self.default_qconfig:
                global_pooling = pooling.ApproximateGlobalAveragePooling1D(
                    input_shape[2], qconfig=self.default_qconfig
                )
            else:
                global_pooling = nn.AvgPool1d(kernel_size=input_shape[2])
        elif len(input_shape) == 4:
            global_pooling = nn.AvgPool2d(kernel_size=input_shape[2:])

        output_shape = (input_shape[0], input_shape[1])

        linear_layers = []
        for linear_config in network_config.linear:
            input_shape, layer = self.linear(input_shape, linear_config)
            linear_layers.append(layer)

        input_shape, last_linear = self.linear(
            input_shape,
            LinearConfig(
                outputs=labels,
                norm=False,
                act=False,
                out_quant=False,
                qconfig=True if self.default_qconfig else False,
            ),
        )

        linear_layers.append(last_linear)

        linear_sequential = nn.Sequential(*linear_layers)

        dropout = nn.Dropout(network_config.dropout)

        model = ConvNet(
            conv_layers_sequential,
            global_pooling,
            linear_sequential,
            dropout,
            self.default_qconfig,
        )

        return output_shape, model

    def _calc_spatial_dim(
        self, in_dim: int, kernel_size: int, stride: int, padding: int, dilation: int
    ) -> int:
        """

        Args:
          in_dim: int:
          kernel_size: int:
          stride: int:
          padding: int:
          dilation: int:
          in_dim: int:
          kernel_size: int:
          stride: int:
          padding: int:
          dilation: int:

        Returns:

        """
        return (in_dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def _padding(self, kernel_size: int, stride: int, _dilation: int) -> int:
        """

        Args:
          kernel_size: int:
          stride: int:
          _dilation: int:
          kernel_size: int:
          stride: int:
          _dilation: int:

        Returns:

        """
        padding = (((kernel_size - 1) * _dilation) + 1) // 2
        return padding


def create_cnn(
    input_shape: Sequence[int],
    labels: int,
    name: str,
    conv: Optional[List[MajorBlockConfig]] = None,
    linear: Optional[List[LinearConfig]] = None,
    norm: Optional[NormConfig] = None,
    act: Optional[ActConfig] = None,
    qconfig: Any = None,
    dropout: float = 0.5,
):
    """

    Args:
      input_shape: Sequence[int]:
      labels: int:
      name: str:
      conv: Optional[List[MajorBlockConfig]]:  (Default value = None)
      linear: Optional[List[LinearConfig]]:  (Default value = None)
      norm: Optional[NormConfig]:  (Default value = None)
      act: Optional[ActConfig]:  (Default value = None)
      qconfig: Any:  (Default value = None)
      dropout: float:  (Default value = 0.5)
      input_shape: Sequence[int]:
      labels: int:
      name: str:
      conv: Optional[List[MajorBlockConfig]]:  (Default value = None)
      linear: Optional[List[LinearConfig]]:  (Default value = None)
      norm: Optional[NormConfig]:  (Default value = None)
      act: Optional[ActConfig]:  (Default value = None)
      qconfig: Any:  (Default value = None)
      dropout: float:  (Default value = 0.5)

    Returns:

    """
    if conv is None:
        conv = []
    if linear is None:
        linear = []
    factory = NetworkFactory()
    schema = OmegaConf.structured(NetworkConfig)

    # FIXME: this does not full validate models at the moment
    #        as Omegaconf does not support Unions atm
    structured_config = OmegaConf.merge(
        schema,
        dict(
            name=name,
            conv=conv,
            linear=linear,
            norm=norm,
            act=act,
            qconfig=qconfig,
            dropout=dropout,
        ),
    )
    output_shape, model = factory.network(input_shape, labels, structured_config)

    return model
