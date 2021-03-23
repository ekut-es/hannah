"""Implementations of torch.nn.intrinsics qat with an optional
   quantize bias parameter.

    Qconfigs can support an optional bias quantization funciton which should be returned by
    `qconfig.bias()` else biases will be quantized with `qconfig.activation()`
"""

import math

from typing import Callable, Dict, Any

import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.utils import _single, _pair
from torch.nn.parameter import Parameter

from . import quantized as q

_BN_CLASS_MAP = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}


class _ConvForwardMixin:
    def _real_conv_forward(self, input, weight, bias):
        if self.dim == 1:
            return F.conv1d(
                input,
                weight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        elif self.dim == 2:
            return F.conv2d(
                input,
                weight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        elif self.dim == 3:
            return F.conv3d(
                input,
                weight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )


class _ConvBnNd(nn.modules.conv._ConvNd, _ConvForwardMixin):

    _version = 2

    def __init__(
        self,
        # ConvNd args
        in_channels,
        out_channels,
        kernel_size,
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
        dim=2,
    ):
        nn.modules.conv._ConvNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
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
        self.bn = _BN_CLASS_MAP[dim](out_channels, eps, momentum, True, True)
        self.weight_fake_quant = self.qconfig.weight()
        self.activation_post_process = self.qconfig.activation()
        self.dim = dim

        if hasattr(self.qconfig, "bias"):
            self.bias_fake_quant = self.qconfig.bias()
        else:
            self.bias_fake_quant = self.qconfig.activation()

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
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
        super(_ConvBnNd, self).reset_parameters()

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
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(
            self.weight * scale_factor.reshape(weight_shape)
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
                bias = self.bias
            bias = self.bias_fake_quant(
                (bias - self.bn.running_mean) * scale_factor + self.bn.bias
            ).reshape(bias_shape)
            conv = conv + bias

        return conv

    def extra_repr(self):
        # TODO(jerryzh): extend
        return super(_ConvBnNd, self).extra_repr()

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

        super(_ConvBnNd, self)._load_from_state_dict(
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


class ConvBn1d(_ConvBnNd):
    r"""
    A ConvBn1d module is a module fused from Conv1d and BatchNorm1d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.
    We combined the interface of :class:`torch.nn.Conv1d` and
    :class:`torch.nn.BatchNorm1d`.
    Similar to :class:`torch.nn.Conv1d`, with FakeQuantize modules initialized
    to default.
    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = nni.ConvBn1d

    def __init__(
        self,
        # Conv1d args
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=None,
        padding_mode="zeros",
        # BatchNorm1d args
        # num_features: out_channels
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        qconfig=None,
    ):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)

        _ConvBnNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            transposed=False,
            output_padding=(0,),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            eps=eps,
            momentum=momentum,
            freeze_bn=freeze_bn,
            qconfig=qconfig,
            dim=1,
        )

        self.activation_post_process = qconfig.activation()

    def forward(self, input):
        y = super(ConvBn1d, self).forward(input)
        return self.activation_post_process(y)


class ConvBnReLU1d(ConvBn1d):
    r"""
    A ConvBnReLU1d module is a module fused from Conv1d, BatchNorm1d and ReLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.
    We combined the interface of :class:`torch.nn.Conv1d` and
    :class:`torch.nn.BatchNorm1d` and :class:`torch.nn.ReLU`.
    Similar to `torch.nn.Conv1d`, with FakeQuantize modules initialized to
    default.
    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = nni.ConvBnReLU1d

    def __init__(
        self,
        # Conv1d args
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=None,
        padding_mode="zeros",
        # BatchNorm1d args
        # num_features: out_channels
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        qconfig=None,
    ):

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            eps=eps,
            momentum=momentum,
            freeze_bn=freeze_bn,
            qconfig=qconfig,
        )
        self.activation_post_process = qconfig.activation()

    def forward(self, input):
        # print(f"ConvBnRelu1d {self.stride}")
        # print(input.shape)
        y = self.activation_post_process(F.relu(ConvBn1d._forward(self, input)))
        # print(y.shape)

        return y

    @classmethod
    def from_float(cls, mod):
        return super(ConvBnReLU1d, cls).from_float(mod)


class ConvBn2d(_ConvBnNd):
    r"""
    A ConvBn2d module is a module fused from Conv2d and BatchNorm2d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.
    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d`.
    Similar to :class:`torch.nn.Conv2d`, with FakeQuantize modules initialized
    to default.
    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = nni.ConvBn2d

    def __init__(
        self,
        # ConvNd args
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=None,
        padding_mode="zeros",
        # BatchNorm2d args
        # num_features: out_channels
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        qconfig=None,
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        _ConvBnNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            transposed=False,
            output_padding=(0, 0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            eps=eps,
            momentum=momentum,
            freeze_bn=freeze_bn,
            qconfig=qconfig,
            dim=2,
        )
        self.activation_post_process = qconfig.activation()

    def forward(self, input):
        return self.activation_post_process(super(ConvBn2d, self)._forward(input))


class ConvBnReLU2d(ConvBn2d):
    r"""
    A ConvBnReLU2d module is a module fused from Conv2d, BatchNorm2d and ReLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.
    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d` and :class:`torch.nn.ReLU`.
    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.
    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = nni.ConvBnReLU2d

    def __init__(
        self,
        # Conv2d args
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=None,
        padding_mode="zeros",
        # BatchNorm2d args
        # num_features: out_channels
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        qconfig=None,
    ):
        super(ConvBnReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            qconfig,
        )
        self.activation_post_process = qconfig.activation()

    def forward(self, input):
        return self.activation_post_process(
            F.relu(super(ConvBnReLU2d, self)._forward(input))
        )

    @classmethod
    def from_float(cls, mod):
        return super(ConvBnReLU2d, cls).from_float(mod)


class ConvReLU2d(nn.Conv2d, _ConvForwardMixin):
    r"""A ConvReLU2d module is a fused module of Conv2d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.
    We combined the interface of :class:`~torch.nn.Conv2d` and
    :class:`~torch.nn.BatchNorm2d`.
    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = nni.ConvReLU2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        assert qconfig, "qconfig must be provided for QAT module"
        self.dim = 2
        self.qconfig = qconfig
        self.weight_fake_quant = self.qconfig.weight()
        self.activation_post_process = self.qconfig.activation()
        if hasattr(qconfig, "bias"):
            self.bias_fake_quant = self.qconfig.bias()
        else:
            self.bias_fake_quant = self.qconfig.activation()

    def forward(self, input):
        return self.activation_post_process(
            F.relu(
                self._real_conv_forward(
                    input,
                    self.weight_fake_quant(self.weight),
                    self.bias_fake_quant(self.bias),
                )
            )
        )

    @classmethod
    def from_float(cls, mod):
        return super(ConvReLU2d, cls).from_float(mod)


class ConvReLU1d(nn.Conv1d, _ConvForwardMixin):
    r"""A ConvReLU1d module is fused module of Conv1d and ReLU, attached with
     FakeQuantize modules for quantization aware training"""

    _FLOAT_MODULE = nn.Conv1d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvReLU1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        assert qconfig, "qconfig must be provided for QAT module"
        self.dim = 1
        self.qconfig = qconfig
        self.weight_fake_quant = self.qconfig.weight()
        self.activation_post_process = self.qconfig.activation()
        if hasattr(qconfig, "bias"):
            self.bias_fake_quant = self.qconfig.bias()
        else:
            self.bias_fake_quant = self.qconfig.activation()

    def forward(self, input):
        output = self._real_conv_forward(
            input,
            self.weight_fake_quant(self.weight),
            self.bias_fake_quant(self.bias) if self.bias else None,
        )
        output = F.relu(output)
        return self.activation_post_process(output)


class Conv1d(nn.Conv1d, _ConvForwardMixin):
    r"""A Conv1d module is a Conv1d module , attached with
    FakeQuantize modules for weight for
    quantization aware training.
    Attributes:
        weight_fake_quant: fake quant module for weight
        bias_fake_quant: fake quant module for bias
        activation_post_process: fake_quant_module for activations
    """
    _FLOAT_MODULE = nn.Conv1d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
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
        self.activation_post_process = self.qconfig.activation()
        if hasattr(qconfig, "bias"):
            self.bias_fake_quant = self.qconfig.bias()
        else:
            self.bias_fake_quant = self.qconfig.activation()
        self.dim = 1

    def forward(self, input):
        # print(f"Conv1D {self.stride}")
        # print(input.shape)
        y = self.activation_post_process(
            self._real_conv_forward(
                input,
                self.weight_fake_quant(self.weight),
                self.bias_fake_quant(self.bias) if self.bias else None,
            )
        )
        return y

    @classmethod
    def from_float(cls, mod):
        return super(ConvReLU2d, cls).from_float(mod)


class Conv2d(nn.Conv2d, _ConvForwardMixin):
    r"""A Conv2d module is a Conv2d module , attached with
    FakeQuantize modules for weight for
    quantization aware training.
    Attributes:
        weight_fake_quant: fake quant module for weight
        bias_fake_quant: fake quant module for bias
        activation_post_process: fake_quant_module for activations
    """
    _FLOAT_MODULE = nn.Conv2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
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
        self.activation_post_process = self.qconfig.activation()
        if hasattr(qconfig, "bias"):
            self.bias_fake_quant = self.qconfig.bias()
        else:
            self.bias_fake_quant = self.qconfig.activation()
        self.dim = 2

    def forward(self, input):
        y = self.activation_post_process(
            self._real_conv_forward(
                input,
                self.weight_fake_quant(self.weight),
                self.bias_fake_quant(self.bias),
            )
        )

        return y

    @classmethod
    def from_float(cls, mod):
        return super(Conv2d, cls).from_float(mod)


class Linear(nn.Linear):
    r"""
    A linear module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """
    _FLOAT_MODULE = nn.Linear

    def __init__(self, in_features, out_features, bias=True, qconfig=None):
        super().__init__(in_features, out_features, bias)
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight()
        if hasattr(qconfig, "bias"):
            self.bias_fake_quant = qconfig.bias()
        else:
            self.bias_fake_quant = qconfig.activation()

        self.activation_post_process = qconfig.activation()

    @property
    def scaled_weight(self):
        return self.weight_fake_quant(self.weight)

    def forward(self, input):
        return self.activation_post_process(
            F.linear(
                input,
                self.weight_fake_quant(self.weight),
                self.bias_fake_quant(self.bias) if self.bias is not None else self.bias,
            )
        )

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            " qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        if type(mod) == LinearReLU:
            mod = mod[0]

        qconfig = mod.qconfig
        qat_linear = cls(
            mod.in_features,
            mod.out_features,
            bias=mod.bias is not None,
            qconfig=qconfig,
        )
        qat_linear.weight = mod.weight
        qat_linear.bias = mod.bias
        return qat_linear


def update_bn_stats(mod):
    if type(mod) in set([ConvBnReLU1d, ConvBnReLU2d, ConvBn1d, ConvBn2d]):
        mod.update_bn_stats()


def freeze_bn_stats(mod):
    if type(mod) in set([ConvBnReLU1d, ConvBnReLU2d, ConvBn1d, ConvBn2d]):
        mod.freeze_bn_stats()


# Default map for swapping float module to qat modules
QAT_MODULE_MAPPINGS: Dict[Callable, Any] = {
    Conv1d: q.Conv1d,
    Conv2d: q.Conv2d,
    Linear: q.Linear,
    # Intrinsic modules:
    ConvBn1d: q.Conv1d,
    ConvBn2d: q.Conv2d,
    ConvBnReLU1d: q.ConvReLU1d,
    ConvBnReLU2d: q.ConvReLU2d,
    ConvReLU1d: q.ConvReLU1d,
    ConvReLU2d: q.ConvReLU2d,
    torch.quantization.stubs.QuantStub: nn.Identity,
    torch.quantization.stubs.DeQuantStub: nn.Identity,
}
