import timm
from torch import nn


class TimmModel(nn.Module):
    def __init__(self, input_shape: tuple, labels : int, name: str, pretrained : bool =False, **kwargs):
        super().__init__()
        self.name = name
        self.model = timm.create_model(
            name, num_classes=labels, pretrained=pretrained
        )
        self.input_shape = input_shape
        self.model.conv_stem = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)

    def forward(self, x):
        return self.model(x)
