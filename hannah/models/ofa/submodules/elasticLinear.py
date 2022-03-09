import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as nnf

from .elasticBase import _Elastic
from ..utilities import (
    filter_primary_module_weights,
    filter_single_dimensional_weights,
    make_parameter,
)
from .elasticchannelhelper import SequenceDiscovery
from ...factory import qat


class ElasticWidthLinear(nn.Linear, _Elastic):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        nn.Linear.__init__(in_features, out_features, bias=bias)
        _Elastic.__init__([True] * in_features, [True] * out_features)

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

    def get_basic_module(self):
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


class ElasticQuantWidthLinear(nn.Linear, _Elastic):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        out_quant=True,
        qconfig=None,
    ) -> None:

        super().__init__(in_features, out_features, bias=bias)
        _Elastic.__init__(self, [True] * in_features, [True] * out_features)

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

    def get_basic_module(self):
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
