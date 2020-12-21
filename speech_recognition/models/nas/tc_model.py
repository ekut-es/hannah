import logging
import torch.nn as nn
import torch
from torch.autograd import Variable
from ..utils import SerializableModule, next_power_of2
from hydra.utils import instantiate

msglogger = logging.getLogger()


def create_act(act, clipping_value):
    if act == "relu":
        return nn.ReLU()
    else:
        return nn.Hardtanh(0.0, clipping_value)


class ApproximateGlobalAveragePooling1D(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.size = size
        self.divisor = next_power_of2(size)

    def forward(self, x):
        x = torch.sum(x, dim=2, keepdim=True)
        x = x / self.divisor

        return x


# TODO: implement case: block type input and minor type parallel
class MajorBlock(nn.Module):
    def __init__(
        self,
        output_channels,
        stride,
        branch,
        minor_blocks,
        input_channels=None,
        act_layer=None,
    ):
        super().__init__()

        assert act_layer is not None

        # module lists for branches of MajorBlock
        self.main_modules = nn.ModuleList()
        self.parallel_modules = nn.ModuleList()

        self.is_residual_block = False
        if branch == "residual":
            self.is_residual_block = True

        # config for iteration of minors
        n_parallels = sum(block["parallel"] for block in minor_blocks)
        n_mains = len(minor_blocks) - n_parallels
        count_main = 1
        count_parallel = 1

        stride_main = stride
        stride_parallel = stride

        input_channels_main = input_channels
        input_channels_parallel = input_channels

        kwargs_main = {
            "batch_norm": True,
            "act_layer": act_layer,
        }

        # BUILD MajorBlock of MinorBlocks
        for minor_block in minor_blocks:

            is_parallel = minor_block["parallel"]
            size = minor_block["size"]

            # parallel MinorBlocks
            if is_parallel:

                if count_parallel > 1:
                    input_channels_parallel = output_channels
                    stride_parallel = 1

                module = MinorBlock(
                    input_channels_parallel,
                    output_channels,
                    size,
                    stride_parallel,
                    batch_norm=True,
                    act_layer=act_layer,
                )

                count_parallel += 1
                self.parallel_modules.append(module)

            # main MinorBlocks
            else:
                if count_main > 1:
                    input_channels_main = output_channels
                    stride_main = 1
                    kwargs_main

                if not self.is_residual_block:
                    kwargs_main = {
                        "batch_norm": False,
                        "act_layer": None,
                    }
                elif count_main == n_mains:
                    kwargs_main = {
                        "batch_norm": True,
                        "act_layer": None,
                    }

                module = MinorBlock(
                    input_channels_main,
                    output_channels,
                    size,
                    stride_main,
                    **kwargs_main
                )

                self.main_modules.append(module)
                count_main += 1

        # only append activation layer if residual block
        if self.is_residual_block:
            self.act_layer = act_layer

    def forward(self, x):

        main_feed = x
        parallel_feed = x

        # always feed through main modules
        for layer in self.main_modules:
            main_feed = layer(main_feed)

        if self.is_residual_block:

            for layer in self.parallel_modules:
                parallel_feed = layer(parallel_feed)

            act_input = main_feed + parallel_feed

            output = self.act_layer(act_input)

        output = main_feed

        return output


# 1D-Conv + (optional: activation) + (optional: batch normalization) + (optional: dropout)
class MinorBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        size,
        stride,
        batch_norm=True,
        act_layer=None,
    ):
        super().__init__()
        layers = []

        layers.append(
            nn.Conv1d(
                input_channels,
                output_channels,
                size,
                stride,
                # padding=dilation * pad_x,
                # dilation=dilation,
                bias=False,
            )
        )

        if batch_norm:
            layers.append(nn.BatchNorm1d(output_channels))

        if act_layer is not None:
            layers.append(act_layer)

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        y = self.sequential(x)

        return y


class TCCandidateModel(SerializableModule):
    def __init__(self, config):
        super().__init__()

        # config properties
        n_labels = config["n_labels"]
        width = config["width"]
        height = config["height"]
        dropout_prob = config["dropout_prob"]
        width_multiplier = config["width_multiplier"]
        activation_fnc = config["activation_function"]
        is_fully_convolutional = config["is_fully_convolutional"]
        has_inputlayer = config["has_inputlayer"]
        dilation_factor = config["dilation_factor"]
        clipping_value = config["clipping_value"]
        small = config["small"]
        act_layer = create_act(activation_fnc, clipping_value)

        # TODO: make following properties configurable
        bottleneck = 0
        channel_division = 4
        separable = 0
        # TODO: channel division not implemented

        # config architecture
        major_blocks = config["major_blocks"]

        # model
        self.modules = []

        # BUILD MODEL BODY
        input_channels = height

        for major_block in major_blocks:
            # first major block cannot be parallel
            current_module = instantiate(
                major_block, input_channels=input_channels, act_layer=act_layer
            )
            self.modules.append(current_module)
            input_channels = major_block["output_channels"]

        # ------------------

        # BUILD AND APPEND pooling, droput and fully connect

        # dummy input to forward once through the model for configuring
        x = Variable(torch.zeros(1, height, width))

        # iterate over the layers of the main branch to get dummy output
        print("!!! TCCandidateModel layers:")
        for layer in self.modules:
            print(layer)
            x = layer(x)
        print("------------------------------")

        # average pooling
        shape = x.shape
        average_pooling = ApproximateGlobalAveragePooling1D(x.shape[2])
        self.modules.append(average_pooling)
        x = average_pooling(x)

        # dropout
        self.dropout = nn.Dropout(dropout_prob)

        # fully connect
        if not self.fully_convolutional:
            x = x.view(1, -1)
        shape = x.shape
        if self.fully_convolutional:
            self.fc = nn.Conv1d(shape[1], n_labels, 1, bias=False)
        else:
            self.fc = nn.Linear(shape[1], n_labels, bias=False)

        print("Model created.")

    def forward(self, x):
        for layer in self.modules:
            x = layer(x)

        x = self.dropout(x)
        if not self.fully_convolutional:
            x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
