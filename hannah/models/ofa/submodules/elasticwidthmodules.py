import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as nnf

from ..utilities import filter_primary_module_weights, filter_single_dimensional_weights
from .elasticchannelhelper import SequenceDiscovery
from torch.nn.modules.batchnorm import _BatchNorm
from ...factory import qat

# from .elastickernelconv import ElasticKernelConv1d
# from hannah.models.ofa.submodules.elasticchannelhelper import ElasticChannelHelper


class ElasticBatchnorm1d(nn.BatchNorm1d):
    def __init__(
        self,
        num_features,
        track_running_stats=False,
    ):
        super().__init__(
            num_features=num_features, track_running_stats=track_running_stats
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, SequenceDiscovery):
            return x.discover(self)

        return super().forward(x)

    def assemble_basic_module(self) -> nn.BatchNorm1d:
        return copy.deepcopy(super())


class ElasticWidthBatchnorm1d(nn.BatchNorm1d):
    def __init__(
        self,
        num_features,
        track_running_stats=False,
    ):
        super().__init__(
            num_features=num_features, track_running_stats=track_running_stats
        )
        self.channel_filter = [True] * num_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        if self.track_running_stats:
            logging.warn(
                "ElasticWidthBatchnorm with tracked running stats currently not fully implemented!"
            )
            # num_batches_tracked and exponential averaging are currently not implemented.

        running_mean = self.running_mean
        running_var = self.running_var
        weight = self.weight
        bias = self.bias
        training = self.training
        momentum = self.momentum
        eps = self.eps

        if all(self.channel_filter):
            # if the channels are unfiltered, the full batchnorm can be used
            return nnf.batch_norm(
                input=input,
                running_mean=running_mean,
                running_var=running_var,
                weight=weight,
                bias=bias,
                training=training or not self.track_running_stats,
                momentum=momentum,
                eps=eps,
            )

        else:
            new_running_mean = None
            new_running_var = None
            if self.track_running_stats:
                new_running_mean = filter_single_dimensional_weights(
                    running_mean, self.channel_filter
                )
                new_running_var = filter_single_dimensional_weights(
                    running_var, self.channel_filter
                )
            new_weight = filter_single_dimensional_weights(weight, self.channel_filter)
            new_bias = filter_single_dimensional_weights(bias, self.channel_filter)

            return nnf.batch_norm(
                input=input,
                running_mean=new_running_mean,
                running_var=new_running_var,
                weight=new_weight,
                bias=new_bias,
                training=training or not self.track_running_stats,
                momentum=momentum,
                eps=eps,
            )

    def get_basic_batchnorm1d(self):
        # filter_single_dimensional_weights checks for None-input, no need to do it here.
        new_running_mean = filter_single_dimensional_weights(
            self.running_mean, self.channel_filter
        )
        new_running_var = filter_single_dimensional_weights(
            self.running_var, self.channel_filter
        )
        new_weight = make_parameter(
            filter_single_dimensional_weights(self.weight, self.channel_filter)
        )
        new_bias = make_parameter(
            filter_single_dimensional_weights(self.bias, self.channel_filter)
        )
        new_bn = nn.BatchNorm1d(
            num_features=self.num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        )
        new_bn.running_mean = new_running_mean
        new_bn.running_var = new_running_var
        new_bn.weight = new_weight
        new_bn.bias = new_bias
        new_bn.training = self.training
        return new_bn

    def assemble_basic_module(self) -> nn.BatchNorm1d:
        return copy.deepcopy(self).get_basic_batchnorm1d()


class ElasticWidthLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.in_channel_filter = [True] * self.in_features
        self.out_channel_filter = [True] * self.out_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        if all(self.in_channel_filter) and all(self.out_channel_filter):
            # if no channel filtering is required, simply use the full linear
            return nnf.linear(input, self.weight, self.bias)
        else:
            # if channels need to be filtered, apply filters.
            new_weight = filter_primary_module_weights(
                self.weight, self.in_channel_filter, self.out_channel_filter
            )
            # if the module has a bias parameter, also apply the output filtering to it.
            # filter_single_dimensional_weights checks for None-input, so no check is done here.
            new_bias = filter_single_dimensional_weights(
                self.bias, self.out_channel_filter
            )
            return nnf.linear(input, new_weight, new_bias)

    def get_basic_linear(self):
        weight = self.weight
        bias = self.bias
        # weight and bias of this linear will be overwritten
        new_linear = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
        )
        if all(self.in_channel_filter) and all(self.out_channel_filter):
            new_linear.weight = weight
            new_linear.bias = bias
            return new_linear
        else:
            new_weight = make_parameter(
                filter_primary_module_weights(
                    self.weight, self.in_channel_filter, self.out_channel_filter
                )
            )
            new_bias = make_parameter(
                filter_single_dimensional_weights(self.bias, self.out_channel_filter)
            )
            new_linear.weight = new_weight
            new_linear.bias = new_bias
            return new_linear

    def assemble_basic_module(self):
        return copy.deepcopy(self).get_basic_linear()


class ElasticQuantWidthLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        out_quant=True,
        qconfig=None,
    ) -> None:

        super().__init__(in_features, out_features, bias=bias)

        self.in_channel_filter = [True] * self.in_features
        self.out_channel_filter = [True] * self.out_features

        assert qconfig, "qconfig must be provided for QAT module"
        self.out_quant = out_quant
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight()
        if hasattr(qconfig, "bias"):
            self.bias_fake_quant = qconfig.bias()
        else:
            self.bias_fake_quant = qconfig.activation()

        self.activation_post_process = (
            qconfig.activation() if out_quant else nn.Identity()
        )

    @property
    def filtered_weight(self):
        if all(self.in_channel_filter) and all(self.out_channel_filter):
            return self.weight
        else:

            return filter_primary_module_weights(
                self.weight, self.in_channel_filter, self.out_channel_filter
            )

    @property
    def filtered_bias(self):
        if all(self.in_channel_filter) and all(self.out_channel_filter):
            return self.bias
        else:
            return filter_single_dimensional_weights(self.bias, self.out_channel_filter)

    @property
    def scaled_weight(self):
        return self.weight_fake_quant(self.filtered_weight)

    @property
    def scaled_bias(self):
        return (
            self.bias_fake_quant(self.filtered_bias)
            if self.bias is not None
            else self.bias
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        return self.activation_post_process(
            nnf.linear(input, self.scaled_weight, self.scaled_bias)
        )

    def get_basic_linear(self):
        weight = self.weight
        bias = self.bias
        # weight and bias of this linear will be overwritten
        new_linear = qat.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias is not None,
            out_quant=self.out_quant,
            qconfig=self.qconfig,
        )
        if all(self.in_channel_filter) and all(self.out_channel_filter):
            new_linear.weight = weight
            new_linear.bias = bias
            return new_linear
        else:
            new_weight = make_parameter(
                filter_primary_module_weights(
                    self.weight, self.in_channel_filter, self.out_channel_filter
                )
            )
            new_bias = make_parameter(
                filter_single_dimensional_weights(self.bias, self.out_channel_filter)
            )
            new_linear.weight = new_weight
            new_linear.bias = new_bias
            return new_linear

    def assemble_basic_module(self):
        return copy.deepcopy(self).get_basic_linear()

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
        # TODO kombi noch einfügen
        # if type(mod) == LinearReLU:
        #    mod = mod[0]

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


# just a ReLu, which can forward a SequenceDiscovery
class ElasticPermissiveReLU(nn.ReLU):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if isinstance(x, SequenceDiscovery):
            return x
        else:
            return super().forward(x)

    def assemble_basic_module(self):
        return nn.ReLU()


def make_parameter(t: torch.Tensor) -> nn.Parameter:
    if t is None:
        return t
    if isinstance(t, nn.Parameter):
        return t
    elif isinstance(t, torch.Tensor):
        return nn.parameter.Parameter(t)
    else:
        logging.error(f"Could not create parameter from input of type '{type(t)}'.")
        return None
