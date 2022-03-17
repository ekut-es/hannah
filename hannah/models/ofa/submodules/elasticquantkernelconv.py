import copy
from typing import List
import torch.nn as nn
import torch.nn.functional as nnf
import logging
import math
import torch
from ..utilities import (
    conv1d_get_padding,
    filter_single_dimensional_weights,
)
from .elasticchannelhelper import SequenceDiscovery
from .elasticBatchnorm import ElasticWidthBatchnorm1d
from .elasticLinear import ElasticPermissiveReLU
from .elasticBase import ElasticBase1d
from ...factory import qat

from torch.nn import init

# pytype: enable=attribute-error
class _ElasticConvBnNd(
    ElasticBase1d, qat._ConvForwardMixin
):  # pytype: disable=module-attr

    _version = 2

    def __init__(
        self,
        # ConvNd args
        in_channels,
        out_channels,
        kernel_sizes,
        dilation_sizes,
        stride=1,
        padding=0,
        transposed=False,
        output_padding=0,
        groups=1,
        bias=False,
        padding_mode="zeros",
        # BatchNormNd args
        eps=1e-05,
        momentum=0.1,
        freeze_bn=False,
        qconfig=None,
        dim=1,
        out_quant=True,
        track_running_stats=True,
    ):
        ElasticBase1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            stride=stride,
            padding=padding,
            dilation_sizes=dilation_sizes,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.freeze_bn = freeze_bn if self.training else True
        self.bn = nn.ModuleList()
        self.bn.append(
            ElasticWidthBatchnorm1d(
                out_channels,
                eps=eps,
                momentum=momentum,
                track_running_stats=track_running_stats,
            )
        )

        self.weight_fake_quant = self.qconfig.weight()
        self.activation_post_process = (
            self.qconfig.activation() if out_quant else nn.Identity
        )
        self.dim = dim

        if hasattr(self.qconfig, "bias"):
            self.bias_fake_quant = self.qconfig.bias()
        else:
            self.bias_fake_quant = self.qconfig.activation()

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        self.reset_bn_parameters()

        # this needs to be called after reset_bn_parameters,
        # as they modify the same state
        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

    def on_warmup_end(self):
        for i in range(len(self.kernel_sizes) - 1):
            self.bn.append(copy.deepcopy(self.bn[0]))

    def reset_running_stats(self):
        for idx in range(len(self.bn)):
            self.bn[idx].reset_running_stats()

    def reset_bn_parameters(self):
        for idx in range(len(self.bn)):
            self.bn[idx].reset_running_stats()
            init.uniform_(self.bn[idx].weight)
            init.zeros_(self.bn[idx].bias)
        # note: below is actully for conv, not BN
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def reset_parameters(self):
        super(_ElasticConvBnNd, self).reset_parameters()

    def update_bn_stats(self):
        self.freeze_bn = False
        for idx in range(len(self.bn) - 1):
            self.bn[idx].training = True
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        for idx in range(len(self.bn) - 1):
            self.bn[idx].training = False
        return self

    @property
    def scale_factor(self):
        running_std = torch.sqrt(
            self.bn[self.target_kernel_index].running_var
            + self.bn[self.target_kernel_index].eps
        )
        scale_factor = self.bn[self.target_kernel_index].weight / running_std

        if all(self.out_channel_filter):
            return scale_factor
        else:
            return filter_single_dimensional_weights(
                scale_factor, self.out_channel_filter
            )

    @property
    def scaled_weight(self):
        scale_factor = self.scale_factor
        weight, bias = self.get_kernel()
        weight_shape = [1] * len(weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(
            weight * scale_factor.reshape(weight_shape)
        )

        return scaled_weight

    def _forward(self, input):
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        self.padding = conv1d_get_padding(self.kernel_sizes[self.target_kernel_index])

        scale_factor = self.scale_factor
        scaled_weight = self.scaled_weight

        # using zero bias here since the bias for original conv
        # will be added later
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias)
        else:
            zero_bias = torch.zeros(self.out_channels, device=scaled_weight.device)
        kernelsize = self.kernel_sizes[self.target_kernel_index]
        self.dilation = self.get_dilation_size()
        self.padding = conv1d_get_padding(kernelsize, self.dilation)
        if not all(self.out_channel_filter):
            zero_bias = filter_single_dimensional_weights(
                zero_bias, self.out_channel_filter
            )
        conv = self._real_conv_forward(input, scaled_weight, zero_bias)
        if self.training:
            conv_orig = conv / scale_factor.reshape(bias_shape)
            if self.bias is not None and all(self.out_channel_filter):
                conv_orig = conv_orig + self.bias.reshape(bias_shape)
            elif self.bias is not None and not all(self.out_channel_filter):
                tmpbias = filter_single_dimensional_weights(
                    self.bias, self.out_channel_filter
                )
                conv_orig = conv_orig + tmpbias.reshape(bias_shape)
            conv = self.bn[self.target_kernel_index](conv_orig)
            # conv = conv - (self.bn.bias - self.bn.running_mean).reshape(bias_shape)
        else:
            bias = zero_bias
            if self.bias is not None:
                _, bias = self.get_kernel()
                bias = filter_single_dimensional_weights(bias, self.out_channel_filter)
            bn_rmean = self.bn[self.target_kernel_index].running_mean
            bn_bias = self.bn[self.target_kernel_index].bias

            if not all(self.out_channel_filter):

                bn_rmean = filter_single_dimensional_weights(
                    bn_rmean, self.out_channel_filter
                )
                bn_bias = filter_single_dimensional_weights(
                    bn_bias, self.out_channel_filter
                )

            bias = self.bias_fake_quant(
                (bias - bn_rmean) * scale_factor + bn_bias
            ).reshape(bias_shape)
            conv = conv + bias

        return conv

    def extra_repr(self):
        # TODO(jerryzh): extend
        return None
        # return super(_ElasticConvBnNd, self).extra_repr()

    def forward(self, input):
        y = self._forward(input)

        return y

    def train(self, mode=True):
        """
        Batchnorm's training behavior is using the self.training flag. Prevent
        changing it if BN is frozen. This makes sure that calling `model.train()`
        on a model with a frozen BN will behave properly.
        """
        self.training = mode
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)
        return self

    # ===== Serialization version history =====
    #
    # Version 1/None
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- gamma : Tensor
    #   |--- beta : Tensor
    #   |--- running_mean : Tensor
    #   |--- running_var : Tensor
    #   |--- num_batches_tracked : Tensor
    #
    # Version 2
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- bn : Module
    #        |--- weight : Tensor (moved from v1.self.gamma)
    #        |--- bias : Tensor (moved from v1.self.beta)
    #        |--- running_mean : Tensor (moved from v1.self.running_mean)
    #        |--- running_var : Tensor (moved from v1.self.running_var)
    #        |--- num_batches_tracked : Tensor (moved from v1.self.num_batches_tracked)
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        if version is None or version == 1:
            # BN related parameters and buffers were moved into the BN module for v2
            v2_to_v1_names = {
                "bn.weight": "gamma",
                "bn.bias": "beta",
                "bn.running_mean": "running_mean",
                "bn.running_var": "running_var",
                "bn.num_batches_tracked": "num_batches_tracked",
            }
            for v2_name, v1_name in v2_to_v1_names.items():
                if prefix + v1_name in state_dict:
                    state_dict[prefix + v2_name] = state_dict[prefix + v1_name]
                    state_dict.pop(prefix + v1_name)
                elif prefix + v2_name in state_dict:
                    # there was a brief period where forward compatibility
                    # for this module was broken (between
                    # https://github.com/pytorch/pytorch/pull/38478
                    # and https://github.com/pytorch/pytorch/pull/38820)
                    # and modules emitted the v2 state_dict format while
                    # specifying that version == 1. This patches the forward
                    # compatibility issue by allowing the v2 style entries to
                    # be used.
                    pass
                elif strict:
                    missing_keys.append(prefix + v2_name)

        super(_ElasticConvBnNd, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict
        Args: `mod` a float module, either produced by torch.quantization utilities
        or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qconfig = mod.qconfig
        conv, bn = mod[0], mod[1]
        qat_convbn = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
            bn.eps,
            bn.momentum,
            False,
            qconfig,
        )
        qat_convbn.weight = conv.weight
        qat_convbn.bias = conv.bias
        qat_convbn.bn.weight = bn.weight
        qat_convbn.bn.bias = bn.bias
        qat_convbn.bn.running_mean = bn.running_mean
        qat_convbn.bn.running_var = bn.running_var
        qat_convbn.bn.num_batches_tracked = bn.num_batches_tracked
        return qat_convbn


class ElasticQuantConv1d(ElasticBase1d, qat._ConvForwardMixin):

    _FLOAT_MODULE = nn.Conv1d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        dilation_sizes: List[int],
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        padding_mode="zeros",
        qconfig=None,
        out_quant=True,
    ):

        # sort available kernel sizes from largest to smallest (descending order)
        kernel_sizes.sort(reverse=True)
        self.kernel_sizes: List[int] = kernel_sizes
        # after sorting kernel sizes, the maximum and minimum size available are the first and last element
        self.max_kernel_size: int = kernel_sizes[0]
        self.min_kernel_size: int = kernel_sizes[-1]
        # initially, the target size is the full kernel
        self.target_kernel_index: int = 0
        self.out_channels: int = out_channels
        padding = conv1d_get_padding(self.kernel_sizes[self.target_kernel_index])

        ElasticBase1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            stride=stride,
            padding=padding,
            dilation_sizes=dilation_sizes,
            groups=groups,
            bias=bias,
        )

        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.out_quant = out_quant
        self.weight_fake_quant = self.qconfig.weight()
        self.activation_post_process = (
            self.qconfig.activation() if out_quant else nn.Identity()
        )
        if hasattr(qconfig, "bias"):
            self.bias_fake_quant = self.qconfig.bias()
        else:
            self.bias_fake_quant = self.qconfig.activation()
        self.dim = 1

        self.in_channel_filter = [True] * self.in_channels
        self.out_channel_filter = [True] * self.out_channels

        # the list of kernel transforms will have one element less than the list of kernel sizes.
        # between every two sequential kernel sizes, there will be a kernel transform
        # the subsequent kernel is determined by applying the same-size center of the previous kernel to the transform
        self.kernel_transforms = nn.ModuleList([])
        for i in range(len(kernel_sizes) - 1):
            # the target size of the kernel transform is the next kernel size in the sequence
            new_kernel_size = kernel_sizes[i + 1]
            # kernel transform is kept minimal by being shared between channels.
            # It is simply a linear transformation from the center of the previous kernel to the new kernel
            # directly applying the kernel to the transform is possible: nn.Linear accepts
            # multi-dimensional input in a way where the last input dim is transformed
            # from in_channels to out_channels for the last output dim
            new_transform_module = nn.Linear(
                new_kernel_size, new_kernel_size, bias=False
            )
            # initialise the transform as the identity matrix to start training
            # from the center of the larger kernel
            new_transform_module.weight.data.copy_(torch.eye(new_kernel_size))
            # transform weights are initially frozen
            new_transform_module.weight.requires_grad = True
            self.kernel_transforms.append(new_transform_module)
        self.set_kernel_size(self.max_kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        # return self.get_basic_conv1d().forward(input)  # for validaing assembled module
        # get the kernel for the current index
        weight, bias = self.get_kernel()
        # get padding for the size of the kernel
        dilation = self.get_dilation_size()
        self.padding = conv1d_get_padding(
            self.kernel_sizes[self.target_kernel_index], dilation
        )
        y = self.activation_post_process(
            self._real_conv_forward(
                input,
                self.weight_fake_quant(weight),
                self.bias_fake_quant(bias) if self.bias is not None else None,
            )
        )
        return y

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_module(self) -> nn.Conv1d:
        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        dilation = self.get_dilation_size()
        padding = conv1d_get_padding(kernel_size, dilation)
        new_conv = qat.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding,
            dilation,
            self.groups,
            bias,
            qconfig=self.qconfig,
            out_quant=self.out_quant,
        )
        new_conv.weight.data = kernel
        if bias is not None:
            new_conv.bias = bias

        # print("\nassembled a basic conv from elastic kernel!")
        return new_conv


class ElasticQuantConvReLu1d(ElasticBase1d, qat._ConvForwardMixin):

    _FLOAT_MODULE = nn.Conv1d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        dilation_sizes: List[int],
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        padding_mode="zeros",
        qconfig=None,
        out_quant=True,
    ):

        # sort available kernel sizes from largest to smallest (descending order)
        kernel_sizes.sort(reverse=True)
        self.kernel_sizes: List[int] = kernel_sizes
        # after sorting kernel sizes, the maximum and minimum size available are the first and last element
        self.max_kernel_size: int = kernel_sizes[0]
        self.min_kernel_size: int = kernel_sizes[-1]
        # initially, the target size is the full kernel
        self.target_kernel_index: int = 0
        self.out_channels: int = out_channels
        padding = conv1d_get_padding(self.kernel_sizes[self.target_kernel_index])
        ElasticBase1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            stride=stride,
            padding=padding,
            dilation_sizes=dilation_sizes,
            groups=groups,
            bias=bias,
        )

        assert qconfig, "qconfig must be provided for QAT module"
        self.relu = ElasticPermissiveReLU()
        self.qconfig = qconfig
        self.out_quant = out_quant
        self.weight_fake_quant = self.qconfig.weight()
        self.activation_post_process = (
            self.qconfig.activation() if out_quant else nn.Identity()
        )
        if hasattr(qconfig, "bias"):
            self.bias_fake_quant = self.qconfig.bias()
        else:
            self.bias_fake_quant = self.qconfig.activation()
        self.dim = 1

        self.in_channel_filter = [True] * self.in_channels
        self.out_channel_filter = [True] * self.out_channels

        # the list of kernel transforms will have one element less than the list of kernel sizes.
        # between every two sequential kernel sizes, there will be a kernel transform
        # the subsequent kernel is determined by applying the same-size center of the previous kernel to the transform
        self.kernel_transforms = nn.ModuleList([])
        for i in range(len(kernel_sizes) - 1):
            # the target size of the kernel transform is the next kernel size in the sequence
            new_kernel_size = kernel_sizes[i + 1]
            # kernel transform is kept minimal by being shared between channels.
            # It is simply a linear transformation from the center of the previous kernel to the new kernel
            # directly applying the kernel to the transform is possible: nn.Linear accepts
            # multi-dimensional input in a way where the last input dim is transformed
            # from in_channels to out_channels for the last output dim
            new_transform_module = nn.Linear(
                new_kernel_size, new_kernel_size, bias=False
            )
            # initialise the transform as the identity matrix to start training
            # from the center of the larger kernel
            new_transform_module.weight.data.copy_(torch.eye(new_kernel_size))
            # transform weights are initially frozen
            new_transform_module.weight.requires_grad = True
            self.kernel_transforms.append(new_transform_module)
        self.set_kernel_size(self.max_kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        # return self.get_basic_conv1d().forward(input)  # for validaing assembled module
        # get the kernel for the current index
        weight, bias = self.get_kernel()
        # get padding for the size of the kernel
        self.dilation = self.get_dilation_size()
        self.padding = conv1d_get_padding(
            self.kernel_sizes[self.target_kernel_index], self.dilation
        )
        y = self.activation_post_process(
            self.relu(
                self._real_conv_forward(
                    input,
                    self.weight_fake_quant(weight),
                    self.bias_fake_quant(bias) if self.bias is not None else None,
                )
            )
        )
        return y

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_module(self) -> nn.Conv1d:
        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        dilation = self.get_dilation_size()
        padding = conv1d_get_padding(kernel_size, dilation)
        new_conv = qat.ConvReLU1d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding,
            dilation,
            self.groups,
            bias,
            qconfig=self.qconfig,
            out_quant=self.out_quant,
        )
        new_conv.weight.data = kernel
        if bias is not None:
            new_conv.bias = bias

        # print("\nassembled a basic conv from elastic kernel!")
        return new_conv


class ElasticQuantConvBn1d(_ElasticConvBnNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        dilation_sizes: List[int],
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        track_running_stats=True,
        qconfig=None,
        out_quant=True,
    ):
        _ElasticConvBnNd.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            stride=stride,
            padding=padding,
            dilation_sizes=dilation_sizes,
            groups=groups,
            bias=bias,
            qconfig=qconfig,
        )
        self.out_quant = out_quant

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        # return self.get_basic_conv1d().forward(input)  # for validaing assembled module
        # get the kernel for the current index
        kernel, bias = self.get_kernel()
        # get padding for the size of the kernel
        dilation = self.get_dilation_size()
        self.padding = conv1d_get_padding(
            self.kernel_sizes[self.target_kernel_index], dilation
        )
        y = super(ElasticQuantConvBn1d, self).forward(input)
        return self.activation_post_process(y)

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_module(self) -> nn.Conv1d:
        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        dilation = self.get_dilation_size()
        padding = conv1d_get_padding(kernel_size, dilation)
        new_conv = qat.ConvBn1d(
            kernel.shape[1],
            kernel.shape[0],
            kernel_size,
            self.stride,
            padding,
            dilation,
            self.groups,
            bias,
            eps=self.bn[self.target_kernel_index].eps,
            momentum=self.bn[self.target_kernel_index].momentum,
            qconfig=self.qconfig,
            out_quant=self.out_quant,
        )
        new_conv.weight.data = kernel
        new_conv.bias = bias
        tmp_bn = self.bn[self.target_kernel_index].get_basic_batchnorm1d()

        new_conv.bn.weight = tmp_bn.weight
        new_conv.bn.bias = tmp_bn.bias
        new_conv.bn.running_var = tmp_bn.running_var
        new_conv.bn.running_mean = tmp_bn.running_mean
        new_conv.bn.num_batches_tracked = tmp_bn.num_batches_tracked

        # print("\nassembled a basic conv from elastic kernel!")
        return new_conv


class ElasticQuantConvBnReLu1d(ElasticQuantConvBn1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        dilation_sizes: List[int],
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        track_running_stats=True,
        qconfig=None,
        out_quant=True,
    ):
        ElasticQuantConvBn1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            stride=stride,
            padding=padding,
            dilation_sizes=dilation_sizes,
            groups=groups,
            bias=bias,
            qconfig=qconfig,
        )
        self.out_quant = out_quant

        self.relu = ElasticPermissiveReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        dilation = self.get_dilation_size()
        self.padding = conv1d_get_padding(
            self.kernel_sizes[self.target_kernel_index], dilation
        )
        y = super(ElasticQuantConvBnReLu1d, self)._forward(input)
        return self.activation_post_process(self.relu(y))

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_module(self) -> nn.Conv1d:
        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        dilation = self.get_dilation_size()
        padding = conv1d_get_padding(kernel_size, dilation)
        new_conv = qat.ConvBnReLU1d(
            kernel.shape[1],
            kernel.shape[0],
            kernel_size,
            self.stride,
            padding,
            dilation,
            self.groups,
            bias,
            eps=self.bn[self.target_kernel_index].eps,
            momentum=self.bn[self.target_kernel_index].momentum,
            qconfig=self.qconfig,
            out_quant=self.out_quant,
        )
        new_conv.weight.data = kernel
        new_conv.bias = bias
        tmp_bn = self.bn[self.target_kernel_index].get_basic_batchnorm1d()

        new_conv.bn.weight = tmp_bn.weight
        new_conv.bn.bias = tmp_bn.bias
        new_conv.bn.running_var = tmp_bn.running_var
        new_conv.bn.running_mean = tmp_bn.running_mean
        new_conv.bn.num_batches_tracked = tmp_bn.num_batches_tracked

        # print("\nassembled a basic conv from elastic kernel!")
        return new_conv
