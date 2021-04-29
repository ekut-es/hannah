import torchvision
import torch


class FasterRCNN(torch.nn.Module):
    def __init__(
        self,
        name="faster-rcnn-resnet50",
        labels=list(),
        input_shape=(1, 3, 375, 1242),
        **kwargs
    ):
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(**kwargs)

    def forward(self, x, y=None):
        return self.model(x, y)
