import timm
from torch import nn


class TimmModel(nn.Module):
    def __init__(
        self,
        input_shape: tuple,
        labels: int,
        name: str,
        pretrained: bool = False,
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.model = timm.create_model(
            name, num_classes=labels, pretrained=pretrained, **kwargs
        )
        self.input_shape = input_shape

    def forward(self, x):
        return self.model(x)
