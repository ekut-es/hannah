from .SNNLayers import (
    SpikingDenseLayer,
    Spiking1DeLIFLayer,
    Spiking1DLIFLayer,
    Spiking1DeALIFLayer,
    Spiking1DALIFLayer,
    Spiking1DS2NetLayer,
    ReadoutLayer,
    ReadouteLIFLayer,
    ReadoutLIFLayer,
    SurrogateHeaviside,
    EmptyLayer,
    Surrogate_BP_Function,
    BatchNormalizationThroughTime1D,
    ActivationLayer,
)
import torch.nn as nn


def create_spike_fn(spike_fn_name="SHeaviside"):
    if spike_fn_name == "SHeaviside":
        return SurrogateHeaviside.apply
    elif spike_fn_name == "SBPHeaviside":
        return Surrogate_BP_Function.apply
    else:
        return None


def get1DNeuronLayer(
    channels,
    spike_fn,
    flatten_output,
    convolution_layer,
    alpha,
    beta,
    gamma,
    roh,
    neuron_type,
):
    if neuron_type == "s2net":
        return Spiking1DS2NetLayer(
            channels=channels,
            spike_fn=spike_fn,
            flatten_output=flatten_output,
            convolution_layer=convolution_layer,
        )
    if neuron_type == "eLIF":
        return Spiking1DeLIFLayer(
            channels=channels,
            spike_fn=spike_fn,
            flatten_output=flatten_output,
            beta=beta,
        )
    elif neuron_type == "LIF":
        return Spiking1DLIFLayer(
            channels=channels,
            spike_fn=spike_fn,
            flatten_output=flatten_output,
            alpha=alpha,
            beta=beta,
        )
    elif neuron_type == "eALIF":
        return Spiking1DeALIFLayer(
            channels=channels,
            spike_fn=spike_fn,
            flatten_output=flatten_output,
            beta=beta,
            gamma=gamma,
            roh=roh,
        )
    elif neuron_type == "ALIF":
        return Spiking1DALIFLayer(
            channels=channels,
            spike_fn=spike_fn,
            flatten_output=flatten_output,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            rho=roh,
        )


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
    rho=0.75,
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
                get1DNeuronLayer(
                    channels=out_channels,
                    spike_fn=spike_fn,
                    flatten_output=flatten_output,
                    convolution_layer=conv,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    roh=rho,
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
                get1DNeuronLayer(
                    channels=out_channels,
                    spike_fn=spike_fn,
                    flatten_output=flatten_output,
                    convolution_layer=conv,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    roh=rho,
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
    alpha=1,
    beta=1,
    neurontype="eLIF",
):
    if type == "SNN" and readout:
        return get_readoutLayer(
            neurontype=neurontype,
            input_shape=input_shape,
            output_shape=output_shape,
            w_init_mean=w_init_mean,
            w_init_std=w_init_std,
            eps=eps,
            time_reduction=time_reduction,
            alpha=alpha,
            beta=beta,
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


def get_readoutLayer(
    neurontype,
    input_shape,
    output_shape,
    w_init_mean,
    w_init_std,
    eps,
    time_reduction,
    alpha,
    beta,
):
    if neurontype in ["eLIF", "eALIF"]:
        return ReadouteLIFLayer(
            input_shape,
            output_shape,
            w_init_mean,
            w_init_std,
            time_reduction=time_reduction,
            beta=beta,
        )
    elif neurontype in ["LIF", "ALIF"]:
        return ReadoutLIFLayer(
            input_shape,
            output_shape,
            w_init_mean,
            w_init_std,
            time_reduction=time_reduction,
            alpha=alpha,
            beta=beta,
        )
    else:
        return ReadoutLayer(
            input_shape=input_shape,
            output_shape=output_shape,
            w_init_mean=w_init_mean,
            w_init_std=w_init_std,
            eps=eps,
            time_reduction=time_reduction,
        )


def build1DBatchNorm(out_channels, type=None, timesteps: int = 0):
    if type == "BN":
        return nn.BatchNorm1d(out_channels)
    elif type in ["BNTTv1", "BNTTv2"]:
        return BatchNormalizationThroughTime1D(
            channels=out_channels, timesteps=timesteps, variant=type
        )
    else:
        raise ValueError(
            "Wrong variant for Batchnorm. allowed: BN/BNTTv1/BNTTv2 You used: "
            + str(type)
        )


def buildActivationLayer(spike_fn_name="SHeaviside"):
    return ActivationLayer(create_spike_fn(spike_fn_name))


# else:
#    print("Error wrong type Parameter")
