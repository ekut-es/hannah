import timm
from torch import nn


class TimmModel(nn.Module):
    def __init__(self, name, num_classes=10, pretrained=False, **kwargs):
        super().__init__()
        self.name = name
        self.model = timm.create_model(
            name, num_classes=num_classes, pretrained=pretrained
        )

    def forward(self, x):
        return self.model(x)
