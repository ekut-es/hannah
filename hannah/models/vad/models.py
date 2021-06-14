import torch.nn as nn
from ..utils import ConfigType, SerializableModule
import torch.nn.functional as F


class BottleneckVad(nn.Module):
    def __init__(
        self,
        conv1_features,
        conv1_size,
        conv2_features,
        conv2_size,
        conv3_features,
        conv3_size,
        fc_size,
        stride,
        batch_norm,
    ):
        super().__init__()
        self.batch_norm = batch_norm
        self.norm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, conv1_features, conv1_size, stride)
        self.norm2 = nn.BatchNorm2d(conv1_features)
        self.conv2a = nn.Conv2d(conv1_features, 4, 1)  # Bottleneck layer
        self.conv2b = nn.Conv2d(4, 4, conv2_size)  # Bottleneck layer
        self.conv2c = nn.Conv2d(4, conv2_features, 1)  # Bottleneck layer
        self.norm3 = nn.BatchNorm2d(conv2_features)
        self.conv3 = nn.Conv2d(conv2_features, conv3_features, conv3_size, stride)
        self.fc1 = nn.Linear(fc_size, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        if self.batch_norm:
            x = self.norm1(x)
        x = F.leaky_relu(self.conv1(x))
        if self.batch_norm:
            x = self.norm2(x)
        x = F.leaky_relu(self.conv2a(x))
        x = F.leaky_relu(self.conv2b(x))
        x = F.leaky_relu(self.conv2c(x))
        if self.batch_norm:
            x = self.norm3(x)
        x = F.leaky_relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class SmallVad(nn.Module):
    def __init__(self, conv1_features, conv1_size, fc_size, stride, batch_norm):
        super().__init__()
        self.batch_norm = batch_norm
        self.norm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, conv1_features, conv1_size, stride=stride)
        self.fc1 = nn.Linear(fc_size, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        if self.batch_norm:
            x = self.norm1(x)
        x = F.leaky_relu(self.conv1(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class SimpleVad(nn.Module):
    def __init__(
        self,
        conv1_features,
        conv1_size,
        conv2_features,
        conv2_size,
        conv3_features,
        conv3_size,
        fc_size,
        stride,
        batch_norm,
    ):
        super().__init__()
        self.batch_norm = batch_norm
        self.norm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, conv1_features, conv1_size, stride=stride)
        self.norm2 = nn.BatchNorm2d(conv1_features)
        self.conv2 = nn.Conv2d(
            conv1_features, conv2_features, conv2_size, stride=stride
        )
        self.norm3 = nn.BatchNorm2d(conv2_features)
        self.conv3 = nn.Conv2d(
            conv2_features, conv3_features, conv3_size, stride=stride
        )
        self.fc1 = nn.Linear(fc_size, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        if self.batch_norm:
            x = self.norm1(x)
        x = F.leaky_relu(self.conv1(x))
        if self.batch_norm:
            x = self.norm2(x)
        x = F.leaky_relu(self.conv2(x))
        if self.batch_norm:
            x = self.norm3(x)
        x = F.leaky_relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class BottleneckVadModel(SerializableModule):
    def __init__(self, config):
        super().__init__()

        conv1_features = config["conv1_features"]
        conv1_size = config["conv1_size"]
        conv2_features = config["conv2_features"]
        conv2_size = config["conv2_size"]
        conv3_features = config["conv3_features"]
        conv3_size = config["conv3_size"]
        fc_size = config["fc_size"]
        stride = config["stride"]
        batch_norm = config["batch_norm"]

        self.net = BottleneckVad(
            conv1_features,
            conv1_size,
            conv2_features,
            conv2_size,
            conv3_features,
            conv3_size,
            fc_size,
            stride,
            batch_norm,
        )

    def forward(self, x):
        x = self.net.forward(x)
        return x


class SimpleVadModel(SerializableModule):
    def __init__(self, config):
        super().__init__()

        conv1_features = config["conv1_features"]
        conv1_size = config["conv1_size"]
        conv2_features = config["conv2_features"]
        conv2_size = config["conv2_size"]
        conv3_features = config["conv3_features"]
        conv3_size = config["conv3_size"]
        fc_size = config["fc_size"]
        stride = config["stride"]
        batch_norm = config["batch_norm"]

        self.net = SimpleVad(
            conv1_features,
            conv1_size,
            conv2_features,
            conv2_size,
            conv3_features,
            conv3_size,
            fc_size,
            stride,
            batch_norm,
        )

    def forward(self, x):
        x = self.net.forward(x)
        return x


class SmallVadModel(SerializableModule):
    def __init__(self, config):
        super().__init__()

        conv1_features = config["conv1_features"]
        conv1_size = config["conv1_size"]
        fc_size = config["fc_size"]
        stride = config["stride"]
        batch_norm = config["batch_norm"]

        self.net = SmallVad(conv1_features, conv1_size, fc_size, stride, batch_norm)

    def forward(self, x):
        x = self.net.forward(x)
        return x


configs = {
    ConfigType.SIMPLE_VAD.value: dict(
        conv1_features=3,
        conv1_size=2,
        conv2_features=3,
        conv2_size=2,
        conv3_features=3,
        conv3_size=2,
        fc_size=11286,
        stride=2,
        batch_norm=False,
    ),
    ConfigType.BOTTLENECK_VAD.value: dict(
        conv1_features=3,
        conv1_size=2,
        conv2_features=3,
        conv2_size=2,
        conv3_features=3,
        conv3_size=2,
        fc_size=11286,
        stride=2,
        batch_norm=False,
    ),
    ConfigType.SMALL_VAD.value: dict(
        conv1_features=3, conv1_size=2, fc_size=11700, stride=2, batch_norm=False
    ),
}
