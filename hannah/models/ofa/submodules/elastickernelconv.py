import copy
from typing import List
import torch.nn as nn
import torch.nn.functional as nnf
import logging
import math
import torch
from ..utilities import (
    conv1d_get_padding,
    filter_primary_module_weights,
    filter_single_dimensional_weights,
    # set_weight_maybe_bias_grad,
    sub_filter_start_end,
)
from .elasticchannelhelper import SequenceDiscovery
from .elasticwidthmodules import ElasticWidthBatchnorm1d, ElasticPermissiveReLU
from ...factory.qat import (
    _EConvNd,
    _ConvBnNd,
    ConvBn1d,
    ConvBnReLU1d,
    _ConvForwardMixin,
)
from torch.nn import init

# pytype: enable=attribute-error
class _ElasticConvBnNd(
    nn.modules.conv._ConvNd, _ConvForwardMixin
):  # pytype: disable=module-attr

    _version = 2

    def __init__(
        self,
        # ConvNd args
        in_channels,
        out_channels,
        kernel_sizes,
        stride=1,
        padding=0,
        dilation=1,
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
        track_running_stats=False,
    ):
        # sort available kernel sizes from largest to smallest (descending order)
        kernel_sizes.sort(reverse=True)
        self.kernel_sizes: List[int] = kernel_sizes
        # after sorting kernel sizes, the maximum and minimum size available are the first and last element
        self.max_kernel_size: int = kernel_sizes[0]
        self.min_kernel_size: int = kernel_sizes[-1]
        # initially, the target size is the full kernel
        self.target_kernel_index: int = 0
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        # print(self.out_channels)

        nn.modules.conv._ConvNd.__init__(  # pytype: disable=module-attr
            self,
            in_channels,
            out_channels,
            (self.max_kernel_size,),
            stride,
            (padding,),
            dilation,
            transposed,
            output_padding,
            groups,
            False,
            padding_mode,
        )
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.freeze_bn = freeze_bn if self.training else True
        self.bn = ElasticWidthBatchnorm1d(out_channels, track_running_stats)

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

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        self.bn.reset_running_stats()
        init.uniform_(self.bn.weight)
        init.zeros_(self.bn.bias)
        # note: below is actully for conv, not BN
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def reset_parameters(self):
        super(_ElasticConvBnNd, self).reset_parameters()

    def update_bn_stats(self):
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.training = False
        return self

    @property
    def scale_factor(self):
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std

        return scale_factor

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

        scale_factor = self.scale_factor
        scaled_weight = self.scaled_weight
        # using zero bias here since the bias for original conv
        # will be added later
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias)
        else:
            zero_bias = torch.zeros(self.out_channels, device=scaled_weight.device)
        conv = self._real_conv_forward(input, scaled_weight, zero_bias)
        if self.training:
            conv_orig = conv / scale_factor.reshape(bias_shape)
            if self.bias is not None:
                conv_orig = conv_orig + self.bias.reshape(bias_shape)
            conv = self.bn(conv_orig)
            # conv = conv - (self.bn.bias - self.bn.running_mean).reshape(bias_shape)
        else:
            bias = zero_bias
            if self.bias is not None:
                _, bias = self.get_kernel()
            bias = self.bias_fake_quant(
                (bias - self.bn.running_mean) * scale_factor + self.bn.bias
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

    def set_kernel_size(self, new_kernel_size):
        # previous_kernel_size = self.kernel_sizes[self.target_kernel_index]
        if (
            new_kernel_size < self.min_kernel_size
            or new_kernel_size > self.max_kernel_size
        ):
            logging.warn(
                f"requested elastic kernel size ({new_kernel_size}) outside of min/max range: ({self.max_kernel_size}, {self.min_kernel_size}). clamping."
            )
            if new_kernel_size < self.min_kernel_size:
                new_kernel_size = self.min_kernel_size
            else:
                new_kernel_size = self.max_kernel_size

        self.target_kernel_index = 0
        try:
            index = self.kernel_sizes.index(new_kernel_size)
            self.target_kernel_index = index
        except ValueError:
            logging.warn(
                f"requested elastic kernel size {new_kernel_size} is not an available kernel size. Defaulting to full size ({self.max_kernel_size})"
            )

        # if self.kernel_sizes[self.target_kernel_index] != previous_kernel_size:
        # print(f"\nkernel size was changed: {previous_kernel_size} -> {self.kernel_sizes[self.target_kernel_index]}")

    # the initial kernel size is the first element of the list of available sizes
    # set the kernel back to its initial size
    def reset_kernel_size(self):
        self.set_kernel_size(self.kernel_sizes[0])

    # step current kernel size down by one index, if possible.
    # return True if the size limit was not reached
    def step_down_kernel_size(self):
        next_kernel_index = self.target_kernel_index + 1
        if next_kernel_index < len(self.kernel_sizes):
            self.set_kernel_size(self.kernel_sizes[next_kernel_index])
            # print(f"stepped down kernel size of a module! Index is now {self.target_kernel_index}")
            return True
        else:
            logging.debug(
                f"unable to step down kernel size, no available index after current: {self.target_kernel_index} with size: {self.kernel_sizes[self.target_kernel_index]}"
            )
            return False

    def pick_kernel_index(self, target_kernel_index: int):
        if (target_kernel_index < 0) or (target_kernel_index >= len(self.kernel_sizes)):
            logging.warn(
                f"selected kernel index {target_kernel_index} is out of range: 0 .. {len(self.kernel_sizes)}. Setting to last index."
            )
            target_kernel_index = len(self.kernel_sizes) - 1
        self.set_kernel_size(self.kernel_sizes[target_kernel_index])

    def get_available_kernel_steps(self):
        return len(self.kernel_sizes)

    def get_full_width_kernel(self):
        current_kernel_index = 0
        current_kernel = self.weight

        logging.debug("Target kernel index: %s", str(self.target_kernel_index))

        # step through kernels until the target index is reached.
        while current_kernel_index < self.target_kernel_index:
            if current_kernel_index >= len(self.kernel_sizes):
                logging.warn(
                    f"kernel size index {current_kernel_index} is out of range. Elastic kernel acquisition stopping at last available kernel"
                )
                break
            # find start, end pos of the kernel center for the given next kernel size
            start, end = sub_filter_start_end(
                self.kernel_sizes[current_kernel_index],
                self.kernel_sizes[current_kernel_index + 1],
            )
            # extract the kernel center of the correct size
            kernel_center = current_kernel[:, :, start:end]
            # apply the kernel transformation to the next kernel. the n-th transformation
            # is applied to the n-th kernel, yielding the (n+1)-th kernel
            next_kernel = self.kernel_transforms[current_kernel_index](kernel_center)
            # the kernel has now advanced through the available sizes by one
            current_kernel = next_kernel
            current_kernel_index += 1

        return current_kernel

    def get_kernel(self):
        full_kernel = self.get_full_width_kernel()
        new_kernel = None
        if all(self.in_channel_filter) and all(self.out_channel_filter):
            # if no channel filtering is required, the full kernel can be kept
            new_kernel = full_kernel
        else:
            # if channels need to be filtered, apply filters to the kernel
            new_kernel = filter_primary_module_weights(
                full_kernel, self.in_channel_filter, self.out_channel_filter
            )
        # if the module has a bias parameter, also apply the output filtering to it.
        if self.bias is None:
            return new_kernel, None
        else:
            if all(self.out_channel_filter):
                # if out_channels are unfiltered, the output bias does not need filtering.
                return new_kernel, self.bias
            else:
                new_bias = filter_single_dimensional_weights(
                    self.bias, self.out_channel_filter
                )
                return new_kernel, new_bias


class ElasticBase1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        nn.Module.__init__(self)
        # sort available kernel sizes from largest to smallest (descending order)
        kernel_sizes.sort(reverse=True)
        self.kernel_sizes: List[int] = kernel_sizes
        # after sorting kernel sizes, the maximum and minimum size available are the first and last element
        self.max_kernel_size: int = kernel_sizes[0]
        self.min_kernel_size: int = kernel_sizes[-1]
        # initially, the target size is the full kernel
        self.target_kernel_index: int = 0
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        # print(self.out_channels)

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

    def set_kernel_size(self, new_kernel_size):
        # previous_kernel_size = self.kernel_sizes[self.target_kernel_index]
        if (
            new_kernel_size < self.min_kernel_size
            or new_kernel_size > self.max_kernel_size
        ):
            logging.warn(
                f"requested elastic kernel size ({new_kernel_size}) outside of min/max range: ({self.max_kernel_size}, {self.min_kernel_size}). clamping."
            )
            if new_kernel_size < self.min_kernel_size:
                new_kernel_size = self.min_kernel_size
            else:
                new_kernel_size = self.max_kernel_size

        self.target_kernel_index = 0
        try:
            index = self.kernel_sizes.index(new_kernel_size)
            self.target_kernel_index = index
        except ValueError:
            logging.warn(
                f"requested elastic kernel size {new_kernel_size} is not an available kernel size. Defaulting to full size ({self.max_kernel_size})"
            )

        # if self.kernel_sizes[self.target_kernel_index] != previous_kernel_size:
        # print(f"\nkernel size was changed: {previous_kernel_size} -> {self.kernel_sizes[self.target_kernel_index]}")

    # the initial kernel size is the first element of the list of available sizes
    # set the kernel back to its initial size
    def reset_kernel_size(self):
        self.set_kernel_size(self.kernel_sizes[0])

    # step current kernel size down by one index, if possible.
    # return True if the size limit was not reached
    def step_down_kernel_size(self):
        next_kernel_index = self.target_kernel_index + 1
        if next_kernel_index < len(self.kernel_sizes):
            self.set_kernel_size(self.kernel_sizes[next_kernel_index])
            # print(f"stepped down kernel size of a module! Index is now {self.target_kernel_index}")
            return True
        else:
            logging.debug(
                f"unable to step down kernel size, no available index after current: {self.target_kernel_index} with size: {self.kernel_sizes[self.target_kernel_index]}"
            )
            return False

    def pick_kernel_index(self, target_kernel_index: int):
        if (target_kernel_index < 0) or (target_kernel_index >= len(self.kernel_sizes)):
            logging.warn(
                f"selected kernel index {target_kernel_index} is out of range: 0 .. {len(self.kernel_sizes)}. Setting to last index."
            )
            target_kernel_index = len(self.kernel_sizes) - 1
        self.set_kernel_size(self.kernel_sizes[target_kernel_index])

    def get_available_kernel_steps(self):
        return len(self.kernel_sizes)

    def get_full_width_kernel(self):
        current_kernel_index = 0
        current_kernel = self.weight

        logging.debug("Target kernel index: %s", str(self.target_kernel_index))

        # step through kernels until the target index is reached.
        while current_kernel_index < self.target_kernel_index:
            if current_kernel_index >= len(self.kernel_sizes):
                logging.warn(
                    f"kernel size index {current_kernel_index} is out of range. Elastic kernel acquisition stopping at last available kernel"
                )
                break
            # find start, end pos of the kernel center for the given next kernel size
            start, end = sub_filter_start_end(
                self.kernel_sizes[current_kernel_index],
                self.kernel_sizes[current_kernel_index + 1],
            )
            # extract the kernel center of the correct size
            kernel_center = current_kernel[:, :, start:end]
            # apply the kernel transformation to the next kernel. the n-th transformation
            # is applied to the n-th kernel, yielding the (n+1)-th kernel
            next_kernel = self.kernel_transforms[current_kernel_index](kernel_center)
            # the kernel has now advanced through the available sizes by one
            current_kernel = next_kernel
            current_kernel_index += 1

        return current_kernel

    def get_kernel(self):
        full_kernel = self.get_full_width_kernel()
        new_kernel = None
        if all(self.in_channel_filter) and all(self.out_channel_filter):
            # if no channel filtering is required, the full kernel can be kept
            new_kernel = full_kernel
        else:
            # if channels need to be filtered, apply filters to the kernel
            new_kernel = filter_primary_module_weights(
                full_kernel, self.in_channel_filter, self.out_channel_filter
            )
        # if the module has a bias parameter, also apply the output filtering to it.
        if self.bias is None:
            return new_kernel, None
        else:
            if all(self.out_channel_filter):
                # if out_channels are unfiltered, the output bias does not need filtering.
                return new_kernel, self.bias
            else:
                new_bias = filter_single_dimensional_weights(
                    self.bias, self.out_channel_filter
                )
                return new_kernel, new_bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_conv1d(self) -> nn.Conv1d:
        return None

    # return a safe copy of a conv1d equivalent to this module in the current state
    def assemble_basic_conv1d(self) -> nn.Conv1d:
        return None

    def set_out_channel_filter(self, out_channel_filter):
        pass


class ElasticConv1d(ElasticBase1d, nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
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

        # print(self.out_channels)
        ElasticBase1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_sizes=kernel_sizes,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        nn.Conv1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=self.max_kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

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
        kernel, bias = self.get_kernel()
        # get padding for the size of the kernel
        padding = conv1d_get_padding(self.kernel_sizes[self.target_kernel_index])
        return nnf.conv1d(input, kernel, bias, self.stride, padding, self.dilation)

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_conv1d(self) -> nn.Conv1d:
        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        padding = conv1d_get_padding(kernel_size)
        new_conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            bias=False,
        )
        new_conv.weight.data = kernel
        if bias is not None:
            new_conv.bias = bias

        # print("\nassembled a basic conv from elastic kernel!")
        return self

    # return a safe copy of a conv1d equivalent to this module in the current state
    def assemble_basic_conv1d(self) -> nn.Conv1d:
        return copy.deepcopy(self.get_basic_conv1d())

    def set_out_channel_filter(self, out_channel_filter):
        if out_channel_filter is not None:
            self.out_channel_filter = out_channel_filter


class ElasticQuantConv1d(ElasticBase1d, nn.Conv1d, _ConvForwardMixin):

    _FLOAT_MODULE = nn.Conv1d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
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
            out_channels=self.out_channels,
            kernel_sizes=kernel_sizes,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # print(self.out_channels)
        nn.Conv1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.max_kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
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
        padding = conv1d_get_padding(self.kernel_sizes[self.target_kernel_index])

        y = self.activation_post_process(
            self._real_conv_forward(
                input,
                self.weight_fake_quant(weight),
                self.bias_fake_quant(bias) if self.bias is not None else None,
                padding=padding,
            )
        )
        return y

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_conv1d(self) -> nn.Conv1d:
        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        padding = conv1d_get_padding(kernel_size)
        new_conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            bias=False,
        )
        new_conv.weight.data = kernel
        if bias is not None:
            new_conv.bias = bias

        # print("\nassembled a basic conv from elastic kernel!")
        return self

    # return a safe copy of a conv1d equivalent to this module in the current state
    def assemble_basic_conv1d(self) -> nn.Conv1d:
        return copy.deepcopy(self.get_basic_conv1d())

    def set_out_channel_filter(self, out_channel_filter):
        if out_channel_filter is not None:
            self.out_channel_filter = out_channel_filter


class ElasticQuantConvBn1d(_ElasticConvBnNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        track_running_stats=False,
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

        # print(self.out_channels)
        _ElasticConvBnNd.__init__(
            self,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_sizes=kernel_sizes,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            qconfig=qconfig,
        )
        self.bn = nn.BatchNorm1d(out_channels, track_running_stats)
        self.in_channel_filter = [True] * self.in_channels
        self.out_channel_filter = [True] * self.out_channels
        self.qconfig = qconfig
        self.out_quant = out_quant
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

    def set_out_channel_filter(self, out_channel_filter):
        if out_channel_filter is not None:
            self.out_channel_filter = out_channel_filter
            self.bn.channel_filter = out_channel_filter

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        # return self.get_basic_conv1d().forward(input)  # for validaing assembled module
        # get the kernel for the current index
        kernel, bias = self.get_kernel()
        # get padding for the size of the kernel
        padding = conv1d_get_padding(self.kernel_sizes[self.target_kernel_index])
        y = super(ElasticQuantConvBn1d, self)._forward(input)
        return self.activation_post_process(y)

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_conv1d(self) -> nn.Conv1d:
        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        padding = conv1d_get_padding(kernel_size)
        new_conv = ConvBn1d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding,
            self.dilation,
            self.groups,
            bias,
            eps=self.bn.eps,
            momentum=self.bn.momentum,
            qconfig=self.qconfig,
            out_quant=self.out_quant,
        )
        new_conv.weight.data = kernel
        new_conv.bias = bias

        new_conv.bn.weight = self.bn.weight
        new_conv.bn.bias = self.bn.bias
        new_conv.bn.running_var = self.bn.running_var
        new_conv.bn.running_mean = self.bn.running_mean

        # print("\nassembled a basic conv from elastic kernel!")
        return self

    # return a safe copy of a conv1d equivalent to this module in the current state
    def assemble_basic_conv1d(self) -> nn.Conv1d:
        return copy.deepcopy(self.get_basic_conv1d())

    def assemble_basic_batchnorm1d(self):
        return self.bn.assemble_basic_batchnorm1d()


class ElasticQuantConvBnReLu1d(ElasticQuantConvBn1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        track_running_stats=False,
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

        # print(self.out_channels)
        ElasticQuantConvBn1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_sizes=kernel_sizes,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            qconfig=qconfig,
        )
        self.bn = nn.BatchNorm1d(out_channels, track_running_stats)
        self.in_channel_filter = [True] * self.in_channels
        self.out_channel_filter = [True] * self.out_channels
        self.qconfig = qconfig
        self.out_quant = out_quant
        self.relu = ElasticPermissiveReLU()
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

    def set_out_channel_filter(self, out_channel_filter):
        if out_channel_filter is not None:
            self.out_channel_filter = out_channel_filter
            self.bn.channel_filter = out_channel_filter

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        # return self.get_basic_conv1d().forward(input)  # for validaing assembled module
        # get the kernel for the current index
        kernel, bias = self.get_kernel()
        # get padding for the size of the kernel
        padding = conv1d_get_padding(self.kernel_sizes[self.target_kernel_index])
        y = super(ElasticQuantConvBnReLu1d, self)._forward(input)
        return self.activation_post_process(y)

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_conv1d(self) -> nn.Conv1d:
        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        padding = conv1d_get_padding(kernel_size)
        new_conv = ConvBn1d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding,
            self.dilation,
            self.groups,
            bias,
            eps=self.bn.eps,
            momentum=self.bn.momentum,
            qconfig=self.qconfig,
            out_quant=self.out_quant,
        )
        new_conv.weight.data = kernel
        new_conv.bias = bias

        new_conv.bn.weight = self.bn.weight
        new_conv.bn.bias = self.bn.bias
        new_conv.bn.running_var = self.bn.running_var
        new_conv.bn.running_mean = self.bn.running_mean

        # print("\nassembled a basic conv from elastic kernel!")
        return self

    # return a safe copy of a conv1d equivalent to this module in the current state
    def assemble_basic_conv1d(self) -> nn.Conv1d:
        return copy.deepcopy(self.get_basic_conv1d())

    def assemble_basic_batchnorm1d(self):
        return self.bn.assemble_basic_batchnorm1d()


class ElasticConvBn1d(ElasticBase1d, nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        track_running_stats=False,
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
        # print(self.out_channels)
        padding = conv1d_get_padding(self.kernel_sizes[self.target_kernel_index])
        ElasticBase1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_sizes=kernel_sizes,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        nn.Conv1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=self.max_kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = ElasticWidthBatchnorm1d(out_channels, track_running_stats)
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

    def set_out_channel_filter(self, out_channel_filter):
        if out_channel_filter is not None:
            self.out_channel_filter = out_channel_filter
            self.bn.channel_filter = out_channel_filter

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        # return self.get_basic_conv1d().forward(input)  # for validaing assembled module
        # get the kernel for the current index
        kernel, bias = self.get_kernel()
        # get padding for the size of the kernel
        padding = conv1d_get_padding(self.kernel_sizes[self.target_kernel_index])
        return self.bn(
            nnf.conv1d(input, kernel, bias, self.stride, padding, self.dilation)
        )

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_conv1d(self) -> nn.Conv1d:
        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        padding = conv1d_get_padding(kernel_size)
        new_conv = BConvBn1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            bias=False,
        )
        new_conv.weight.data = kernel
        new_conv.bias = bias

        # print("\nassembled a basic conv from elastic kernel!")
        return self

    # return a safe copy of a conv1d equivalent to this module in the current state
    def assemble_basic_conv1d(self) -> nn.Conv1d:
        return copy.deepcopy(self.get_basic_conv1d())

    def assemble_basic_batchnorm1d(self):
        return self.bn.assemble_basic_batchnorm1d()


class ElasticConvBnReLu1d(ElasticConvBn1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        track_running_stats=False,
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

        # print(self.out_channels)
        ElasticConvBn1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_sizes=kernel_sizes,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.relu = ElasticPermissiveReLU()

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
        # kernel, bias = self.get_kernel()
        # get padding for the size of the kernel
        # padding = conv1d_get_padding(self.kernel_sizes[self.target_kernel_index])
        # t = nnf.conv1d(input, kernel, bias, self.stride, padding, self.dilation)
        return self.relu(super(ElasticConvBnReLu1d, self).forward(input))

    # return a normal conv1d equivalent to this module in the current state
    def get_basic_conv1d(self) -> nn.Conv1d:
        kernel, bias = self.get_kernel()
        kernel_size = self.kernel_sizes[self.target_kernel_index]
        padding = conv1d_get_padding(kernel_size)
        new_conv = BConvBnReLu1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            bias=False,
        )
        new_conv.weight.data = kernel
        new_conv.bias = bias

        # print("\nassembled a basic conv from elastic kernel!")
        return self


class BConvBn1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        track_running_stats=False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels, track_running_stats=track_running_stats)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        return self.bn(super(BConvBn1d, self).forward(input))


class BConvBnReLu1d(BConvBn1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        track_running_stats=False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size,),
            stride=(stride,),
            padding=padding,
            dilation=(dilation,),
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels, track_running_stats=track_running_stats)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        return self.relu(super(BConvBnReLu1d, self).forward(input))
