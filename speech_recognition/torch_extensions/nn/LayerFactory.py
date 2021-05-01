from .SNNLayers import (
    SpikingDenseLayer,
    Spiking1DLayer,
    ReadoutLayer,
    SurrogateHeaviside,
    EmptyLayer,
    Surrogate_BP_Function,
    BatchNormalizationThroughTime1D,
)
import torch.nn as nn


def create_spike_fn(spike_fn_name="SHeaviside"):
    if spike_fn_name == "SHeaviside":
        return SurrogateHeaviside.apply
    elif spike_fn_name == "SBPHeaviside":
        return Surrogate_BP_Function.apply
    else:
        return None


def build1DConvolution(
    type,
    in_channels,
    out_channels,
    kernel_size=3,
    dilation=1,
    spike_fn=None,
    stride=1,
    padding=0,
    flatten_output: bool = False,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = "zeros",
    timesteps: int = 0,
    batchnorm=None,
    activation=None,
    alpha=0.75,
    beta=0.75,
    gamma=0.75,
    neuron_type="eLIF",
):
    conv = nn.Conv1d(
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
    bn = build1DBatchNorm(out_channels, type=batchnorm, timesteps=timesteps)

    if batchnorm is None and activation is None and spike_fn is None:
        return nn.Sequential(conv)
    elif batchnorm is None and (activation is not None or spike_fn is not None):
        if type == "SNN":
            return nn.Sequential(
                conv,
                Spiking1DLayer(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    spike_fn=spike_fn,
                    stride=stride,
                    flatten_output=flatten_output,
                    convolution_layer=conv,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    neuron_type=neuron_type,
                ),
            )
        elif type == "NN":
            return nn.Sequential(conv, activation)
    elif batchnorm is not None and activation is None and spike_fn is None:
        return nn.Sequential(conv, bn)
    elif batchnorm is not None and (activation is not None or spike_fn is not None):
        if type == "SNN":
            return nn.Sequential(
                conv,
                bn,
                Spiking1DLayer(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    spike_fn=spike_fn,
                    stride=stride,
                    flatten_output=flatten_output,
                    convolution_layer=conv,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    neuron_type=neuron_type,
                ),
            )
        elif type == "NN":
            return nn.Sequential(conv, bn, activation)
    else:
        print("Error wrong type Parameter")


def buildLinearLayer(
    type,
    input_shape,
    output_shape,
    w_init_mean=0.0,
    w_init_std=0.15,
    eps=1e-8,
    spike_fn=None,
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


def build1DBatchNorm(out_channels, type=None, timesteps: int = 0):
    if type == "BN":
        return nn.BatchNorm1d(out_channels)
    elif type in ["BNTTv1", "BNTTv2"]:
        return BatchNormalizationThroughTime1D(
            channels=out_channels, timesteps=timesteps, variant=type
        )
    else:
        return None


# else:
#    print("Error wrong type Parameter")
