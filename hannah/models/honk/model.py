import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def truncated_normal(tensor, std_dev=0.01):
    tensor.zero_()
    tensor.normal_(std=std_dev)
    while torch.sum(torch.abs(tensor) > 2 * std_dev) > 0:
        t = tensor[torch.abs(tensor) > 2 * std_dev]
        t.zero_()
        tensor[torch.abs(tensor) > 2 * std_dev] = torch.normal(t, std=std_dev)


class SpeechResModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_labels = config["n_labels"]
        n_maps = config["n_feature_maps"]
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        if "res_pool" in config:
            self.pool = nn.AvgPool2d(config["res_pool"])

        self.n_layers = n_layers = config["n_layers"]
        dilation = config["use_dilation"]
        if dilation:
            self.convs = [
                nn.Conv2d(
                    n_maps,
                    n_maps,
                    (3, 3),
                    padding=int(2 ** (i // 3)),
                    dilation=int(2 ** (i // 3)),
                    bias=False,
                )
                for i in range(n_layers)
            ]
        else:
            self.convs = [
                nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1, bias=False)
                for _ in range(n_layers)
            ]
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)
        self.output = nn.Linear(n_maps, n_labels)

    def forward(self, x):
        x = x.unsqueeze(1)
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        x = x.view(x.size(0), x.size(1), -1)  # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return self.output(x)


class SpeechModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_labels = config["n_labels"]
        n_featmaps1 = config["n_feature_maps1"]

        conv1_size = config["conv1_size"]  # (time, frequency)
        conv1_pool = config["conv1_pool"]
        conv1_stride = tuple(config["conv1_stride"])
        dropout_prob = config["dropout_prob"]
        width = config["width"]
        height = config["height"]

        self.conv1 = nn.Conv2d(1, n_featmaps1, conv1_size, stride=conv1_stride)
        tf_variant = config.get("tf_variant")
        self.tf_variant = tf_variant
        if tf_variant:
            truncated_normal(self.conv1.weight.data)
            self.conv1.bias.data.zero_()
        self.pool1 = nn.MaxPool2d(conv1_pool)

        x = Variable(torch.zeros(1, 1, height, width), volatile=True)
        x = self.pool1(self.conv1(x))
        conv_net_size = x.view(1, -1).size(1)
        last_size = conv_net_size

        if "conv2_size" in config:
            conv2_size = config["conv2_size"]
            conv2_pool = config["conv2_pool"]
            conv2_stride = tuple(config["conv2_stride"])
            n_featmaps2 = config["n_feature_maps2"]
            self.conv2 = nn.Conv2d(
                n_featmaps1, n_featmaps2, conv2_size, stride=conv2_stride
            )
            if tf_variant:
                truncated_normal(self.conv2.weight.data)
                self.conv2.bias.data.zero_()
            self.pool2 = nn.MaxPool2d(conv2_pool)
            x = self.pool2(self.conv2(x))
            conv_net_size = x.view(1, -1).size(1)
            last_size = conv_net_size
        if not tf_variant:
            self.lin = nn.Linear(conv_net_size, 32)

        if "dnn1_size" in config:
            dnn1_size = config["dnn1_size"]
            last_size = dnn1_size
            if tf_variant:
                self.dnn1 = nn.Linear(conv_net_size, dnn1_size)
                truncated_normal(self.dnn1.weight.data)
                self.dnn1.bias.data.zero_()
            else:
                self.dnn1 = nn.Linear(32, dnn1_size)
            if "dnn2_size" in config:
                dnn2_size = config["dnn2_size"]
                last_size = dnn2_size
                self.dnn2 = nn.Linear(dnn1_size, dnn2_size)
                if tf_variant:
                    truncated_normal(self.dnn2.weight.data)
                    self.dnn2.bias.data.zero_()
        self.output = nn.Linear(last_size, n_labels)
        if tf_variant:
            truncated_normal(self.output.weight.data)
            self.output.bias.data.zero_()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.relu(self.conv1(x.unsqueeze(1)))  # shape: (batch, channels, i1, o1)
        x = self.dropout(x)
        x = self.pool1(x)
        if hasattr(self, "conv2"):
            x = F.relu(self.conv2(x))  # shape: (batch, o1, i2, o2)
            x = self.dropout(x)
            x = self.pool2(x)
        x = x.view(x.size(0), -1)  # shape: (batch, o3)
        if hasattr(self, "lin"):
            x = self.lin(x)
        if hasattr(self, "dnn1"):
            x = self.dnn1(x)
            if not self.tf_variant:
                x = F.relu(x)
            x = self.dropout(x)
        if hasattr(self, "dnn2"):
            x = self.dnn2(x)
            x = self.dropout(x)
        return self.output(x)
