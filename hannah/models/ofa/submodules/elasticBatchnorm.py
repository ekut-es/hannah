import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from ..utilities import filter_single_dimensional_weights, make_parameter


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
        """if self.track_running_stats:
        logging.warn(
            "ElasticWidthBatchnorm with tracked running stats currently not fully implemented!"
        )
        # num_batches_tracked and exponential averaging are currently not implemented.
        """
        running_mean = filter_single_dimensional_weights(
            self.running_mean, self.channel_filter
        )
        running_var = filter_single_dimensional_weights(
            self.running_var, self.channel_filter
        )
        weight = filter_single_dimensional_weights(self.weight, self.channel_filter)
        bias = filter_single_dimensional_weights(self.bias, self.channel_filter)
        training = self.training
        momentum = self.momentum
        eps = self.eps

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

    def get_basic_batchnorm1d(self):
        # filter_single_dimensional_weights checks for None-input, no need to do it here.
        running_mean = filter_single_dimensional_weights(
            self.running_mean, self.channel_filter
        )
        running_var = filter_single_dimensional_weights(
            self.running_var, self.channel_filter
        )
        weight = make_parameter(
            filter_single_dimensional_weights(self.weight, self.channel_filter)
        )
        bias = make_parameter(
            filter_single_dimensional_weights(self.bias, self.channel_filter)
        )
        new_bn = nn.BatchNorm1d(
            num_features=self.num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        )
        new_bn.running_mean = running_mean
        new_bn.running_var = running_var
        new_bn.weight = weight
        new_bn.bias = bias
        new_bn.training = self.training
        return new_bn

    def assemble_basic_module(self) -> nn.BatchNorm1d:
        return copy.deepcopy(self.get_basic_batchnorm1d())
