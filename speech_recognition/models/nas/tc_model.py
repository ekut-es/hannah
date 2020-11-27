import logging
import torch.nn as nn
import torch
from torch.autograd import Variable
from ..utils import SerializableModule, next_power_of2

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
    pass


# 1D-Conv + (optional: activation) + (optional: batch normalization) + (optional: dropout)
class MicroBlock(nn.Module):
    pass


class TCCandidateModel(SerializableModule):
    def __init__(self, config):
        super().__init__()

        # get from config
        n_labels = config["n_labels"]
        width = config["width"]
        height = config["height"]
        dropout_prob = config["dropout_prob"]
        width_multiplier = config["width_multiplier"]
        k_MicroBlocks = config["k_MicroBlocks"]
        i_MajorBlocks = config["i_MajorBlocks"]
        m_TCResidualBlocks = config["m_TCResidualBlocks"]

        # model
        self.layers = nn.ModuleList()

        # BUILD MODEL BODY
        input_channels = height

        # BUILD AND APPEND pooling, droput and fully connect

        # dummy input to forward once through the model for configuring
        x = Variable(torch.zeros(1, height, width))

        # iterate over the layers of the main branch to get dummy output
        for layer in self.layers:
            x = layer(x)

        # average pooling
        shape = x.shape
        average_pooling = ApproximateGlobalAveragePooling1D(x.shape[2])
        self.layers.append(average_pooling)
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
        for layer in self.layers:
            x = layer(x)

        x = self.dropout(x)
        if not self.fully_convolutional:
            x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
