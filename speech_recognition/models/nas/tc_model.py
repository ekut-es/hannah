import logging
from numpy.core.shape_base import _block_setup
import torch.nn as nn
import torch
from torch.autograd import Variable
from ..utils import SerializableModule, next_power_of2
from hydra.utils import instantiate

msglogger = logging.getLogger()


def create_act(act, clipping_value):
    if act == "relu":
        return nn.ReLU()
    elif act == "hardtanh":
        return nn.Hardtanh(0.0, clipping_value)
    else:
        raise ("Unknown activation function: %s", act)


class DummyActivation(nn.Identity):
    """Dummy class that instantiated to mark a missing activation.

       This can be used to mark requantization of activations for convolutional layers without
       activation functions.
    """

    pass


class ApproximateGlobalAveragePooling1D(nn.Module):
    """A global average pooling layer, that divides by the next power of 2 instead of true number of elements"""

    def __init__(self, size):
        super().__init__()

        self.size = size
        self.divisor = next_power_of2(size)

    def forward(self, x):
        x = torch.sum(x, dim=2, keepdim=True)
        x = x / self.divisor

        return x


class MajorBlock(nn.Module):
    def __init__(
        self,
        output_channels,
        stride,
        branch,
        minor_blocks,
        activation_fn="relu",
        clipping_value=6,
        input_channels=None,
        dilation_factor=None,
    ):
        super().__init__()

        # module lists for branches of MajorBlock
        self.main_modules = nn.ModuleList()
        self.parallel_modules = nn.ModuleList()

        self.is_residual_block = False
        self.is_input_block = False
        self.has_parallel = False

        if branch == "residual":
            self.is_residual_block = True
        elif branch == "input":
            self.is_input_block = True

        n_parallels = sum(block["parallel"] for block in minor_blocks)
        n_seq = sum(not block["parallel"] for block in minor_blocks)

        # If all blocks are set to parallel:
        # Make all blocks sequential
        if n_seq == 0:
            n_parallels = 0
            for block in minor_blocks:
                block["parallel"] = False

        if n_parallels > 0:
            self.has_parallel = True

        # helpers for iteration of minors
        count_main = 1
        count_parallel = 1

        stride_main = stride
        stride_parallel = stride

        input_channels_main = input_channels
        input_channels_parallel = input_channels

        # BUILD MajorBlock of MinorBlocks
        for minor_block in minor_blocks:

            # standard minor block config is fully loaded
            kwargs_main = {
                "dilation_factor": dilation_factor,
                "act_layer": create_act(activation_fn, clipping_value),
            }

            kwargs_parallel = {
                "dilation_factor": dilation_factor,
                "act_layer": create_act(activation_fn, clipping_value),
            }

            # get minor block config
            size = minor_block["size"]
            padding = minor_block["padding"]
            batchnorm = minor_block["batchnorm"]
            activation = minor_block["activation"]
            is_parallel = minor_block["parallel"]

            if not activation:
                kwargs_main.update({"act_layer": None})
                kwargs_parallel.update({"act_layer": None})

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
                    padding,
                    batchnorm,
                    **kwargs_parallel,
                )

                count_parallel += 1
                self.parallel_modules.append(module)

            # main MinorBlocks
            else:

                if count_main > 1:
                    input_channels_main = output_channels
                    stride_main = 1

                module = MinorBlock(
                    input_channels_main,
                    output_channels,
                    size,
                    stride_main,
                    padding,
                    batchnorm,
                    **kwargs_main,
                )

                self.main_modules.append(module)
                count_main += 1

            # only if block has parallel minor blocks it needs a final activation layer for the outs of the branches
            if self.has_parallel:
                self.act_layer = create_act(activation_fn, clipping_value)

    def forward(self, x):

        main_feed = x
        parallel_feed = x

        # always feed through main modules
        for layer in self.main_modules:
            main_feed = layer(main_feed)

        act_input = main_feed

        if self.is_residual_block:
            #                |---> parallel: True  --->  parallel: True  ---> |
            # Residual:  --->|                                                +--->
            #                |---> parallel: False --->  parallel: False ---> |
            if self.has_parallel:
                for layer in self.parallel_modules:
                    parallel_feed = layer(parallel_feed)

                act_input = main_feed + parallel_feed

        if self.is_input_block:
            #                 |---> parallel: True  --->  |  ---> |
            #                 |---> parallel: True  --->  |  ---> + ----------|
            # Input:      --->|                                               +--->
            #                 |---> parallel: False ---> parallel: False      |
            if self.parallel_modules:
                parallel_outs = []
                for layer in self.parallel_modules:
                    parallel_outs.append(layer(parallel_feed))

                parallel_outs_sum = sum(parallel_outs)
                act_input = main_feed + parallel_outs_sum

        if not self.has_parallel:
            output = main_feed
        else:
            output = self.act_layer(act_input)

        return output


# 1D-Conv + (optional: activation) + (optional: batch normalization)
class MinorBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        size,
        stride,
        padding,
        batch_norm,
        dilation_factor=None,
        act_layer=None,
    ):
        super().__init__()
        layers = nn.ModuleList()

        if padding:
            assert dilation_factor is not None
            pad_x = size // 2
            padding = dilation_factor * pad_x

        layers.append(
            nn.Conv1d(
                input_channels,
                output_channels,
                size,
                stride,
                padding=padding,
                dilation=dilation_factor,
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
        activation_fn = config["activation_function"]
        dropout_prob = config["dropout_prob"]
        clipping_value = config["clipping_value"]
        dilation_factor = config["dilation_factor"]
        major_blocks = config["major_blocks"]

        # model
        self.modules_list = nn.ModuleList()

        # BUILD MODEL BODY
        input_channels = height

        for major_block in major_blocks:
            # first major block cannot be parallel
            current_module = instantiate(
                # TODO positional arguments not working
                major_block,
                input_channels=input_channels,
                activation_fn=activation_fn,
                clipping_value=clipping_value,
                dilation_factor=dilation_factor,
            )
            self.modules_list.append(current_module)
            # input channels for next major block are output channels of current
            input_channels = major_block["output_channels"]

        # GET OUTPUT SHAPE

        # dummy input to forward once through the model for configuring
        x = Variable(torch.zeros(1, height, width))
        self.eval()

        # iterate over the layers of the main branch to get dummy output
        print("!!! TCCandidateModel layers:")
        for layer in self.modules_list:
            print(layer)
            x = layer(x)
        print("------------------------------")

        # APPEND average pooling
        shape = x.shape
        average_pooling = ApproximateGlobalAveragePooling1D(x.shape[2])
        self.modules_list.append(average_pooling)
        x = average_pooling(x)

        # APPEND dropout
        self.dropout = nn.Dropout(dropout_prob)

        # APPEND fully connect
        x = x.view(1, -1)
        shape = x.shape
        self.fc = nn.Linear(shape[1], n_labels, bias=False)

        print("Model created.")

    def forward(self, x):
        for layer in self.modules_list:
            x = layer(x)

        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
