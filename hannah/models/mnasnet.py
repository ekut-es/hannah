from torchvision.models import mnasnet0_5
from torchvision.models.mnasnet import MNASNet as TorchMNAS
from torch import nn


# class MNASNET(nn.Module):
#     def __init__(self, name, input_shape, labels, width=0.5) -> None:
#         self.name = name
#         self.model = None
#         super().__init__()
#         if width == 0.5:
#             self.model = mnasnet0_5(num_classes=labels)
#         else:
#             raise Exception(f"width {width} is not a valid choice. Use one of  [0.5].")

#     def forward(self, x):
#         return self.model(x)


class MNASNET(TorchMNAS):
    def __init__(self, name, input_shape, width, labels=10):
        super().__init__(alpha=width, num_classes=labels, dropout=0.1)
