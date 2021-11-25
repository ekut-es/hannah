import timm
from torch import nn


class TimmModel(nn.Module):
    def __init__(self, name, num_classes=10, pretrained=False, **kwargs):
        super().__init__()
        self.name = name
        self.model = timm.create_model(
            name, num_classes=num_classes, pretrained=pretrained
        )
        self.model.conv_stem = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)

    def forward(self, x):
        return self.model(x)
