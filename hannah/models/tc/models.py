import logging
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.autograd import Variable

msglogger = logging.getLogger(__name__)

import numpy as np
import pwlf

from ..utils import next_power_of2


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


class TCResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        size,
        stride,
        dilation,
        clipping_value,
        bottleneck,
        channel_division,
        separable,
        small,
        act,
    ):
        super().__init__()
        self.stride = stride
        self.clipping_value = clipping_value
        if stride > 1:
            # No dilation needed: 1x1 kernel
            act = create_act(act, clipping_value)
            self.downsample = nn.Sequential(
                nn.Conv1d(input_channels, output_channels, 1, stride, bias=False),
                nn.BatchNorm1d(output_channels),
                act,
            )

        pad_x = size // 2

        if bottleneck:
            groups = output_channels // channel_division if separable else 1
            act = create_act(act, clipping_value)
            self.convs = nn.Sequential(
                nn.Conv1d(
                    input_channels,
                    output_channels // channel_division,
                    1,
                    stride=1,
                    dilation=dilation,
                    bias=False,
                ),
                nn.Conv1d(
                    output_channels // channel_division,
                    output_channels // channel_division,
                    size,
                    stride=stride,
                    padding=dilation * pad_x,
                    dilation=dilation,
                    bias=False,
                    groups=groups,
                ),
                nn.Conv1d(
                    output_channels // channel_division,
                    output_channels,
                    1,
                    stride=1,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm1d(output_channels),
                act,
                nn.Conv1d(
                    output_channels,
                    output_channels // channel_division,
                    1,
                    stride=1,
                    dilation=dilation,
                    bias=False,
                ),
                nn.Conv1d(
                    output_channels // channel_division,
                    output_channels // channel_division,
                    size,
                    1,
                    padding=dilation * pad_x,
                    dilation=dilation,
                    bias=False,
                    groups=groups,
                ),
                nn.Conv1d(
                    output_channels // channel_division,
                    output_channels,
                    1,
                    stride=1,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm1d(output_channels),
            )
        elif small:
            act = create_act(act, clipping_value)
            self.convs = nn.Sequential(
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    size,
                    stride,
                    padding=dilation * pad_x,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm1d(output_channels),
                act,
                nn.BatchNorm1d(output_channels),
            )
        else:
            act = create_act(act, clipping_value)
            self.convs = nn.Sequential(
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    size,
                    stride,
                    padding=dilation * pad_x,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm1d(output_channels),
                act,
                nn.Conv1d(
                    output_channels,
                    output_channels,
                    size,
                    1,
                    padding=dilation * pad_x,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm1d(output_channels),
                # distiller.quantization.SymmetricClippedLinearQuantization(num_bits=20, clip_val=2.0**5-1.0/(2.0**14),min_val=-2.0**5)
            )

        self.act = create_act(act, clipping_value)

    def forward(self, x):
        y = self.convs(x)
        if self.stride > 1:
            x = self.downsample(x)

        res = self.act(y + x)

        return res


class TCResNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_labels = config["n_labels"]
        width = config["width"]
        height = config["height"]
        dropout_prob = config["dropout_prob"]
        width_multiplier = config["width_multiplier"]
        self.fully_convolutional = config["fully_convolutional"]
        dilation = config["dilation"]
        clipping_value = config["clipping_value"]
        bottleneck = config["bottleneck"]
        channel_division = config["channel_division"]
        separable = config["separable"]
        small = config["small"]
        use_inputlayer = config["inputlayer"]
        act = config.get("act", "relu")

        self.layers = nn.ModuleList()
        self.feat = None

        input_channels = height

        x = Variable(torch.zeros(1, height, width))

        count = 1
        while "conv{}_size".format(count) in config:
            output_channels_name = "conv{}_output_channels".format(count)
            size_name = "conv{}_size".format(count)
            stride_name = "conv{}_stride".format(count)
            dropout_name = "conv{}_dropout".format(count)
            bn_name = "conv{}_bn".format(count)
            allowActivation_name = "conv{}_activation".format(count)

            output_channels = int(config[output_channels_name] * width_multiplier)
            size = config[size_name]
            stride = config[stride_name]
            dropout = config.get(dropout_name, False)
            bn = config.get(bn_name, False)
            allowact = config.get(allowActivation_name, False)

            # Change first convolution to bottleneck layer.
            if bottleneck[0] == 1:
                channel_division_local = channel_division[0]
                # Change bottleneck layer to separable convolution
                groups = (
                    output_channels // channel_division_local if separable[0] else 1
                )

                conv1 = nn.Conv1d(
                    input_channels,
                    output_channels // channel_division_local,
                    1,
                    1,
                    bias=False,
                )
                conv2 = nn.Conv1d(
                    output_channels // channel_division_local,
                    output_channels // channel_division_local,
                    size,
                    stride,
                    bias=False,
                    groups=groups,
                )
                conv3 = nn.Conv1d(
                    output_channels // channel_division_local,
                    output_channels,
                    1,
                    1,
                    bias=False,
                )

                if dropout:
                    drop_layer = nn.Dropout(dropout_prob)
                    self.layers.append(conv1)
                    self.layers.append(drop_layer)
                    self.layers.append(conv2)
                    self.layers.append(drop_layer)
                    self.layers.append(conv3)
                    self.layers.append(drop_layer)
                else:
                    self.layers.append(conv1)
                    self.layers.append(conv2)
                    self.layers.append(conv3)

            elif use_inputlayer:
                conv = nn.Conv1d(
                    input_channels, output_channels, size, stride, bias=False
                )
                input_channels = output_channels
                self.layers.append(conv)

                if bn:
                    bn = nn.BatchNorm1d(output_channels)
                    self.layers.append(bn)
                if allowact:
                    tmpact = nn.ReLU()
                    self.layers.append(tmpact)

                if dropout:
                    drop_layer = nn.Dropout(dropout_prob)
                    self.layers.append(drop_layer)

                # self.layers.append(distiller.quantization.SymmetricClippedLinearQuantization(num_bits=8, clip_val=0.9921875))
            count += 1

        count = 1
        while "block{}_conv_size".format(count) in config:
            output_channels_name = "block{}_output_channels".format(count)
            size_name = "block{}_conv_size".format(count)
            stride_name = "block{}_stride".format(count)

            output_channels = int(config[output_channels_name] * width_multiplier)
            size = config[size_name]
            stride = config[stride_name]

            block = TCResidualBlock(
                input_channels,
                output_channels,
                size,
                stride,
                dilation**count,
                clipping_value,
                bottleneck[1],
                channel_division[1],
                separable[1],
                small,
                act,
            )
            self.layers.append(block)

            input_channels = output_channels
            count += 1

        for layer in self.layers:
            x = layer(x)

        shape = x.shape
        average_pooling = ApproximateGlobalAveragePooling1D(
            x.shape[2]
        )  # nn.AvgPool1d((shape[2]))
        self.layers.append(average_pooling)

        x = average_pooling(x)

        if not self.fully_convolutional:
            x = x.view(1, -1)

        shape = x.shape

        self.dropout = nn.Dropout(dropout_prob)

        if self.fully_convolutional:
            self.fc = nn.Conv1d(shape[1], n_labels, 1, bias=False)
        else:
            self.fc = nn.Linear(shape[1], n_labels, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        self.feat = x
        if not self.fully_convolutional:
            self.feat = x = x.view(x.size(0), -1)

        x = self.dropout(x)
        if not self.fully_convolutional:
            x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


class ExitWrapperBlock(nn.Module):
    def __init__(
        self,
        wrapped_block: nn.Module,
        exit_branch: nn.Module,
        threshold: float,
        lossweight: float,
    ):

        super().__init__()

        self.wrapped_block = wrapped_block
        self.threshold = threshold
        self.lossweight = lossweight
        self.exit_branch = exit_branch
        self.exit_result = torch.Tensor()

    def forward(self, x):
        x = self.wrapped_block(x)

        x_exit = self.exit_branch(x)
        x_exit = torch.squeeze(x_exit)
        self.exit_result = x_exit

        return x


class BranchyTCResNetModel(TCResNetModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        dropout_prob = config["dropout_prob"]
        n_labels = config["n_labels"]

        self.n_pieces = config.get("exit_n_pieces", 3)
        self.taylor_degree = config.get("exit_taylor_degree", 3)

        self.n_bits = config.get("exit_bits", 20)
        self.f_bits = config.get("exit_f_bits", 14)

        self.exit_max = 2 ** (self.n_bits - self.f_bits - 1) - 1 / (2**self.f_bits)
        self.exit_min = -(2 ** (self.n_bits - self.f_bits - 1))
        self.exit_divider = 2 ** (self.f_bits)

        self.earlyexit_thresholds = config["earlyexit_thresholds"]
        self.earlyexit_lossweights = config["earlyexit_lossweights"]

        assert len(self.earlyexit_thresholds) == len(self.earlyexit_lossweights)
        assert sum(self.earlyexit_lossweights) < 1.0

        # Generate piecewisefunction
        x = np.linspace(-2.0, 2.0)
        y = np.exp(x)
        my_pwlf = pwlf.PiecewiseLinFit(x, y)
        self.piecewise_func = my_pwlf
        my_pwlf.fit(self.n_pieces)

        msglogger.info("Initial piecewise paramters:")
        msglogger.info("Slopes: {}".format(my_pwlf.slopes))
        msglogger.info("Intercepts: {}".format(my_pwlf.intercepts))
        msglogger.info("Breaks: {}".format(my_pwlf.fit_breaks))
        msglogger.info("Beta: {}".format(my_pwlf.beta))

        y_pred = my_pwlf.predict(x)

        # plt.plot(x, y, 'r')
        # plt.plot(x, y_pred, 'b')
        # plt.show()
        # sys.exit(-1)

        exit_candidate = TCResidualBlock

        new_layers = nn.ModuleList()

        x = Variable(torch.zeros(1, config["height"], config["width"]))

        exit_count = 0
        for layer in self.layers:
            x = layer.forward(x)
            if isinstance(layer, exit_candidate) and exit_count < len(
                self.earlyexit_thresholds
            ):
                # Simplified exit branch
                exit_branch = nn.Sequential(
                    nn.Conv1d(x.shape[1], n_labels, 1, bias=False),
                    nn.BatchNorm1d(n_labels),
                    nn.ReLU(),
                    ApproximateGlobalAveragePooling1D(x.shape[2]),
                    nn.Dropout(dropout_prob),
                    nn.Conv1d(n_labels, n_labels, 1, bias=False),
                    # nn.BatchNorm1d(n_labels),
                    # nn.ReLU()
                )

                exit_wrapper = ExitWrapperBlock(
                    layer,
                    exit_branch,
                    self.earlyexit_thresholds[exit_count],
                    self.earlyexit_lossweights[exit_count],
                )

                new_layers.append(exit_wrapper)
                exit_count += 1
            else:
                new_layers.append(layer)

        self.exit_count = exit_count
        self.exits_taken = [0] * (exit_count + 1)

        self.layers = new_layers

        self.test = False

        # Piecewise linear intermediate values
        self.x = []
        self.y = []

    def on_val(self):
        self.reset_stats()
        self.x = []
        self.y = []

    def on_val_end(self):
        self.print_stats()

        x = np.concatenate(self.x)
        y = np.concatenate(self.y)

        msglogger.info("Piecewise Parameters")
        msglogger.info("Slopes: {}".format(self.piecewise_func.slopes))
        msglogger.info("Intercepts: {}".format(self.piecewise_func.intercepts))
        msglogger.info("Breaks: {}".format(self.piecewise_func.fit_breaks))
        msglogger.info("Beta: {}".format(self.piecewise_func.beta))

    def on_test(self):
        self.reset_stats()
        self.test = True

    def on_test_end(self):
        self.print_stats()
        self.test = False

    def reset_stats(self):
        self.exits_taken = [0] * (self.exit_count + 1)

    def print_stats(self):
        msglogger.info("")
        msglogger.info("Early exit statistics")
        for num, taken in enumerate(self.exits_taken):
            msglogger.info("Exit {} taken: {}".format(num, taken))

    def _estimate_losses_real(self, thresholded_result, estimated_labels):
        estimated_losses = torch.nn.functional.cross_entropy(
            thresholded_result, estimated_labels, reduce=False
        )

        return estimated_losses

    def _estimate_losses_taylor(self, thresholded_result, estimated_labels):
        expected_result = torch.zeros(
            thresholded_result.shape, device=thresholded_result.device
        )
        for row, column in enumerate(estimated_labels):
            for column2 in range(expected_result.shape[1]):
                expected_result[row, column2] = thresholded_result[row, column]

        diff = thresholded_result - expected_result
        estimated_losses = torch.sum(
            torch.clamp(
                1 + diff + torch.pow(diff, 2) / 2 + torch.pow(diff, 3) / 6, 0, 64
            ),
            dim=1,
        )

        return torch.log(estimated_losses)

    def _estimate_losses_taylor_approximate(self, thresholded_result, estimated_labels):
        expected_result = torch.zeros(
            thresholded_result.shape, device=thresholded_result.device
        )
        for row, column in enumerate(estimated_labels):
            for column2 in range(expected_result.shape[1]):
                expected_result[row, column2] = thresholded_result[row, column]

        diff = thresholded_result - expected_result
        estimated_losses = torch.sum(
            torch.clamp(
                1 + diff + torch.pow(diff, 2) / 2 + torch.pow(diff, 3) / 8, 0, 64
            ),
            dim=1,
        )

        return torch.log(estimated_losses)

    def _estimate_losses_sum(self, thresholded_result, estimated_labels):
        expected_result = torch.zeros(
            thresholded_result.shape, device=thresholded_result.device
        )
        for row, column in enumerate(estimated_labels):
            for column2 in range(expected_result.shape[1]):
                expected_result[row, column2] = thresholded_result[row, column]

        diff = thresholded_result - expected_result
        estimated_losses = torch.sum(diff, dim=1)

        return estimated_losses

    def forward(self, x):
        x = super().forward(x)
        if self.training:
            results = []
            for layer in self.layers:
                if isinstance(layer, ExitWrapperBlock):
                    results.append(layer.exit_result)

            results.append(x)

            return results

        # Forward in eval mode returns only first exit with estimated loss < thresholds
        exit_number = 0

        zeros = torch.zeros(x.shape, device=x.device)
        ones = torch.ones(x.shape, device=x.device)

        current_mask = torch.ones(x.shape, device=x.device)
        global_result = torch.zeros(x.shape, device=x.device)

        batch_taken = [0] * (self.exit_count)

        for layer in self.layers:
            if isinstance(layer, ExitWrapperBlock):
                threshold = layer.threshold
                result = layer.exit_result
                result = result.view(global_result.shape)
                estimated_labels = result.argmax(dim=1)
                thresholded_result = torch.clamp(result, -32.0, 31.9999389611)

                estimated_losses_real = self._estimate_losses_real(
                    thresholded_result, estimated_labels
                )
                estimated_losses_taylor = self._estimate_losses_taylor(
                    thresholded_result, estimated_labels
                )
                estimated_losses_taylor_approximate = (
                    self._estimate_losses_taylor_approximate(
                        thresholded_result, estimated_labels
                    )
                )
                estimated_losses_sum = self._estimate_losses_sum(
                    thresholded_result, estimated_labels
                )

                #    print("real:", estimated_losses_real)
                #    print("taylor:", estimated_losses_taylor)
                #    print("taylor_approx:", estimated_losses_taylor_approximate)

                estimated_losses = estimated_losses_sum

                self.update_piecewise_data(thresholded_result)

                batch_taken[exit_number] = torch.sum(
                    estimated_losses < threshold
                ).item()

                estimated_losses = estimated_losses.reshape(-1, 1).expand(
                    global_result.shape
                )

                masked_result = torch.where(estimated_losses < threshold, result, zeros)
                masked_result = torch.where(current_mask > 0, masked_result, zeros)
                current_mask = torch.where(
                    estimated_losses < threshold, zeros, current_mask
                )

                global_result += masked_result

                exit_number += 1

        global_result += torch.where(current_mask > 0, x, zeros)
        batch_taken.append(x.shape[0])
        for i, taken in enumerate(batch_taken):
            self.exits_taken[i] += taken

        return global_result

    def get_loss_function(self):
        multipliers = list(self.earlyexit_lossweights)
        multipliers.append(1.0 - sum(multipliers))
        criterion = nn.CrossEntropyLoss()

        def loss_function(scores, labels):
            if isinstance(scores, list):
                loss = torch.zeros([1], device=scores[0].device)
                for multiplier, current_scores in zip(multipliers, scores):
                    current_scores = current_scores.view(current_scores.size(0), -1)
                    current_loss = multiplier * criterion(current_scores, labels)
                    loss += current_loss
                return loss
            else:
                scores = scores.view(scores.size(0), -1)
                return criterion(scores, labels)

        return loss_function
