import torch.nn

from .SNNLayers import TimeTransposeLayer, Spiking1DS2NetLayer
from .SNNReadoutLayers import (
    ReadoutLayer,
    ReadoutMeanLayer,
    ReadoutSpikeTimeLayer,
    ReadoutCountLayer,
    ReadoutFirstSpikeLayer,
)
from .SNNBatchNormThroughTime import BatchNormalizationThroughTime1D
from .SNNActivationLayer import (
    Spiking1DIFLayer,
    Spiking1DeLIFLayer,
    Spiking1DLIFLayer,
    Spiking1DeALIFLayer,
    Spiking1DALIFLayer,
    ActivationLayer,
    Surrogate_BP_Function,
    SurrogateHeaviside,
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
    rho=0.75,
    neuron_type="eLIF",
    trainable_parameter=False,
    negative_mempot=True,
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
                    rho=rho,
                    neuron_type=neuron_type,
                    trainable_parameter=trainable_parameter,
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
                    rho=rho,
                    neuron_type=neuron_type,
                    trainable_parameter=trainable_parameter,
                ),
            )
        elif type == "NN":
            return nn.Sequential(conv, bn, activation)
    else:
        print("Error wrong type Parameter")


def get1DNeuronLayer(
    channels,
    spike_fn,
    flatten_output=False,
    convolution_layer=None,
    alpha=0,
    beta=0,
    gamma=0,
    rho=0,
    neuron_type="IF",
    time_position=2,
    trainable_parameter=False,
    negative_mempot=True,
):
    if neuron_type == "s2net":
        return Spiking1DS2NetLayer(
            channels=channels,
            spike_fn=spike_fn,
            flatten_output=flatten_output,
            convolution_layer=convolution_layer,
            negative_mempot=negative_mempot,
        )
    elif neuron_type == "IF":
        return Spiking1DIFLayer(
            channels=channels,
            spike_fn=spike_fn,
            flatten_output=flatten_output,
            time_position=time_position,
            negative_mempot=negative_mempot,
        )
    elif neuron_type == "eLIF":
        return Spiking1DeLIFLayer(
            channels=channels,
            spike_fn=spike_fn,
            flatten_output=flatten_output,
            beta=beta,
            time_position=time_position,
            trainable_parameter=trainable_parameter,
            negative_mempot=negative_mempot,
        )
    elif neuron_type == "LIF":
        return Spiking1DLIFLayer(
            channels=channels,
            spike_fn=spike_fn,
            flatten_output=flatten_output,
            alpha=alpha,
            beta=beta,
            time_position=time_position,
            trainable_parameter=trainable_parameter,
            negative_mempot=negative_mempot,
        )
    elif neuron_type == "eALIF":
        return Spiking1DeALIFLayer(
            channels=channels,
            spike_fn=spike_fn,
            flatten_output=flatten_output,
            beta=beta,
            gamma=gamma,
            rho=rho,
            time_position=time_position,
            trainable_parameter=trainable_parameter,
            negative_mempot=negative_mempot,
        )
    elif neuron_type == "ALIF":
        return Spiking1DALIFLayer(
            channels=channels,
            spike_fn=spike_fn,
            flatten_output=flatten_output,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            rho=rho,
            time_position=time_position,
            trainable_parameter=trainable_parameter,
            negative_mempot=negative_mempot,
        )


def buildLinearLayer(
    conv_type,
    input_shape,
    output_shape,
    spike_fn=None,
    bias=False,
    alpha=1,
    beta=1,
    gamma=1,
    rho=1,
    neuron_type="eLIF",
    trainable_parameter=False,
    time_position=1,
):
    if conv_type == "SNN":
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=input_shape, out_features=output_shape),
            get1DNeuronLayer(
                channels=output_shape,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                rho=rho,
                neuron_type=neuron_type,
                spike_fn=spike_fn,
                time_position=time_position,
                trainable_parameter=trainable_parameter,
            ),
        )
    elif conv_type == "NN":
        return nn.Linear(in_features=input_shape, out_features=output_shape, bias=bias)
    else:
        print("Error wrong type Parameter")


def buildReadoutLayer(
    readout_type,
    input_shape,
    output_shape,
    alpha,
    beta,
    gamma,
    rho,
    eps=1e-8,
    w_init_mean=0.0,
    w_init_std=0.15,
    spike_fn=None,
    neuron_type="eLIF",
    trainable_parameter=False,
    time_position=1,
):
    if readout_type == "s2net":
        return ReadoutLayer(
            input_shape=input_shape,
            output_shape=output_shape,
            w_init_mean=w_init_mean,
            w_init_std=w_init_std,
            eps=eps,
        )
    elif neuron_type in ["IF", "eLIF", "eALIF", "LIF", "ALIF"] and readout_type in [
        "count",
        "timing",
        "mean",
    ]:

        linear = buildLinearLayer(
            conv_type="SNN",
            input_shape=input_shape,
            output_shape=output_shape,
            spike_fn=spike_fn,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            rho=rho,
            neuron_type=neuron_type,
            time_position=time_position,
            trainable_parameter=trainable_parameter,
        )
        if readout_type == "count":
            return torch.nn.Sequential(linear, ReadoutCountLayer())
        elif readout_type == "timing":
            return torch.nn.Sequential(linear, ReadoutSpikeTimeLayer())
        elif readout_type == "mean":
            return torch.nn.Sequential(
                linear,
                ReadoutMeanLayer(
                    output_shape=output_shape, trainable_parameter=trainable_parameter
                ),
            )
    else:
        print("Error in buildReadoutLayer!!!!!!!!!")


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
