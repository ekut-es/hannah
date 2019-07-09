import torch.nn as nn
from ..utils import ConfigType, SerializableModule
import torch.nn.functional as F


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
        # Max pooling over a (2, 2) window
        x = x.unsqueeze(1)
        x = self.norm1(x)
        x = F.leaky_relu(self.conv1(x))
        x = self.norm2(x)
        # If the size is a square you can only specify a single number
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

class SimpleVadModel(SerializableModule):

    def __init__(self, config):
        super().__init__()

        self.net = SimpleVad()

    def forward(self, x):
        x = self.net.forward(x)
        return x

configs= {
     ConfigType.SIMPLE_VAD.value: dict()
}
