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
        affine=True,
        momentum=0.1,
        eps=1e-5,
    ):

        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.channel_filter = [True] * num_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, SequenceDiscovery):
            return input.discover(self)

        """if self.track_running_stats:
            logging.warn(
                "ElasticWidthBatchnorm with tracked running stats currently not fully implemented!"
            )
            # num_batches_tracked and exponential averaging are currently not implemented.
        """
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
