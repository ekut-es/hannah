import torch.nn as nn
from ..utils import ConfigType, SerializableModule
import torch.nn.functional as F


class BottleneckVad(nn.Module):

    def __init__(self):
            super().__init__()
            self.norm1 = nn.BatchNorm2d(1)
            self.conv1 = nn.Conv2d(1, 32, 3)
            self.norm2 = nn.BatchNorm2d(32)
            self.conv2a = nn.Conv2d(32,4,1) # Bottleneck layer
            self.conv2b = nn.Conv2d(4,4,3)  # Bottleneck layer
            self.conv2c = nn.Conv2d(4,16,1) # Bottleneck layer
            self.norm3 = nn.BatchNorm2d(16)
            self.conv3 = nn.Conv2d(16, 10, 3)
            self.fc1 = nn.Linear(55872, 2)

    def forward(self, x):
            x = x.unsqueeze(1)
            x = self.norm1(x)
            x = F.leaky_relu(self.conv1(x))
            x = self.norm2(x)
            x = F.leaky_relu(self.conv2a(x))
            x = F.leaky_relu(self.conv2b(x))
            x = F.leaky_relu(self.conv2c(x))
            x = self.norm3(x)
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

    def __init__(self):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.fc1 = nn.Linear(37620, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.norm1(x)
        x = F.leaky_relu(self.conv1(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class SimpleVad(nn.Module):

    def __init__(self):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.norm2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, 3)
        self.norm3 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 10, 3)
        self.fc1 = nn.Linear(55872, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.norm1(x)
        x = F.leaky_relu(self.conv1(x))
        x = self.norm2(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.norm3(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class BottleneckVadModel(SerializableModule):
    def __init__(self, config):
        super().__init__()

        self.net = BottleneckVad()

    def forward(self, x):
        x = self.net.forward(x)
        return x

class SimpleVadModel(SerializableModule):

    def __init__(self, config):
        super().__init__()

        self.net = SimpleVad()

    def forward(self, x):
        x = self.net.forward(x)
        return x


class SmallVadModel(SerializableModule):

    def __init__(self, config):
        super().__init__()

        self.net = SmallVad()

    def forward(self, x):
        x = self.net.forward(x)
        return x

configs= {
     ConfigType.SIMPLE_VAD.value: dict(),
     ConfigType.BOTTLENECK_VAD.value: dict(),
     ConfigType.SMALL_VAD.value: dict()
}
