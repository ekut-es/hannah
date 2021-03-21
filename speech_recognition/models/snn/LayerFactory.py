from .SNNLayers import (
    SpikingDenseLayer,
    SpikingConv1DLayer,
    ReadoutLayer,
    SurrogateHeaviside,
)
import torch.nn as nn


def build1DConvolution(
    type,
    in_channels,
    out_channels,
    kernel_size=3,
    dilation=1,
    spike_fn=SurrogateHeaviside.apply,
    stride=1,
    padding=0,
    w_init_mean=0.0,
    w_init_std=0.15,
    recurrent: bool = False,
    lateral_connections: bool = False,
    flatten_output: bool = False,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = "zeros",
):
    if type == "SNN":
        return SpikingConv1DLayer(
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            spike_fn,
            stride=stride,
            w_init_mean=w_init_mean,
            w_init_std=w_init_std,
            recurrent=recurrent,
            lateral_connections=lateral_connections,
            flatten_output=flatten_output,
        )
    elif type == "NN":
        return nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
    else:
        print("Error wrong type Parameter")


def buildLinearLayer(
    type,
    input_shape,
    output_shape,
    w_init_mean=0.0,
    w_init_std=0.15,
    eps=1e-8,
    spike_fn=SurrogateHeaviside.apply,
    time_reduction="mean",
    readout=False,
    recurrent=False,
    lateral_connections=False,
    bias=False,
):
    if type == "SNN" and readout:
        return ReadoutLayer(
            input_shape=input_shape,
            output_shape=output_shape,
            w_init_mean=w_init_mean,
            w_init_std=w_init_std,
            eps=eps,
            time_reduction=time_reduction,
        )

    elif type == "SNN" and not readout:
        return SpikingDenseLayer(
            input_shape=input_shape,
            output_shape=output_shape,
            spike_fn=spike_fn,
            w_init_mean=w_init_mean,
            w_init_std=w_init_std,
            eps=eps,
            recurrent=recurrent,
            lateral_connections=lateral_connections,
        )
    elif type == "NN":
        return nn.Linear(in_features=input_shape, out_features=output_shape, bias=bias)
    else:
        print("Error wrong type Parameter")
