import torchvision
import torch
from PIL import Image


class FasterRCNN(torch.nn.Module):
    def __init__(
        self,
        name="faster-rcnn-resnet50",
        labels=list(),
        input_shape=(1, 3, 375, 1242),
        **kwargs,
    ):
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(**kwargs)

    def forward(self, x, y=None):
        return self.model(x, y)


class YoloV5s:
    # Model

    def __init__(self):
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)

        # Images
        for f in ["zidane.jpg", "bus.jpg"]:  # download 2 images
            print(f"Downloading {f}...")
            torch.hub.download_url_to_file(
                "https://github.com/ultralytics/yolov5/releases/download/v1.0/" + f, f
            )
        img1 = Image.open("zidane.jpg")  # PIL image
        imgs = [img1]  # batch of images

        # Inference
        results = model(imgs, size=640)  # includes NMS

        # Results
        results.print()
        results.save()  # or .show()

        results.xyxy[0]  # img1 predictions (tensor)
        results.pandas().xyxy[0]  # img1 predictions (pandas)
