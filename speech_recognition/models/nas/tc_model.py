import logging
import torch.nn as nn
import torch
from torch.autograd import Variable
from ..utils import SerializableModule, next_power_of2
from hydra.utils import instantiate
from collections import OrderedDict

msglogger = logging.getLogger()


class ApproximateGlobalAveragePooling1D(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.size = size
        self.divisor = next_power_of2(size)

    def forward(self, x):
        x = torch.sum(x, dim=2, keepdim=True)
        x = x / self.divisor

        return x


# max. two MajorBlocks in parallel
class TCResidualBlock(nn.Module):
    pass


# Sequence of MicroBlocks
# + (optional: activation) + (optional: batch normalization) + (optional: dropout)
class MajorBlock(nn.Module):
    def __init__(
        self, output_channels, stride, branch, minor_blocks, input_channels=None
    ):
        super().__init__()
        # Example of using Sequential with OrderedDict
        # model = nn.Sequential(OrderedDict([
        #           ('conv1', nn.Conv2d(1,20,5)),
        #           ('relu1', nn.ReLU()),
        #           ('conv2', nn.Conv2d(20,64,5)),
        #           ('relu2', nn.ReLU())
        #         ]))
        self.main_modules = []
        self.parallel_modules = []

        # TODO: implement case: block type input and minor type parallel
        for n, minor_block in enumerate(minor_blocks):
            print(f"!!! minor_block = {minor_block}")

            if minor_block["parallel"]:
                self.parallel_modules.append(
                    instantiate(minor_block, input_channels=input_channels)
                )
            else:
                self.main_modules.append(
                    instantiate(minor_block, input_channels=input_channels)
                )

    def forward(self, x):
        y = self.main(x)
        if self.parallel > 1:
            x = self.parallel(x)

        res = self.act(y + x)

        return res


# 1D-Conv + (optional: activation) + (optional: batch normalization) + (optional: dropout)
class MinorBlock(nn.Module):
    pass


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

        # TODO: make following properties configurable
        bottleneck = 0
        channel_division = 4
        separable = 0
        # TODO: channel division not implemented

        # config architecture
        major_blocks = config["major_blocks"]

        # model
        self.modules = nn.ModuleList()

        # BUILD MODEL BODY
        input_channels = height

        for major_block in major_blocks:
            # first major block cannot be parallel
            current_module = instantiate(major_block, input_channels=input_channels)
            self.modules.append(current_module)

        # ------------------

        # BUILD AND APPEND pooling, droput and fully connect

        # dummy input to forward once through the model for configuring
        x = Variable(torch.zeros(1, height, width))

        # iterate over the layers of the main branch to get dummy output
        for layer in self.modules:
            x = layer(x)

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
