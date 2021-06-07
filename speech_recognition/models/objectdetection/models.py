import torchvision
import torch
import torch.nn.functional as F

from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .loss import ComputeLoss

from speech_recognition.datasets.Kitti import KittiCOCO


class FasterRCNN(torch.nn.Module):
    def __init__(
        self,
        name="faster-rcnn-resnet50",
        labels=list(),
        input_shape=(1, 3, 375, 1242),
        num_classes=91,
        **kwargs,
    ):
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(**kwargs)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x, y=None):
        return self.model(x, y)

    def transformOutput(self, cocoGt, output, y):
        retval = []

        for boxes, labels, scores, y_img in zip(
            (out["boxes"] for out in output),
            (out["labels"] for out in output),
            (out["scores"] for out in output),
            y,
        ):
            for box, label, score in zip(boxes, labels, scores):
                img_dict = dict()
                x1 = box[0].item()
                y1 = box[1].item()
                img_dict["image_id"] = cocoGt.getImgId(y_img["filename"])
                img_dict["category_id"] = label.item()
                img_dict["bbox"] = [x1, y1, box[2].item() - x1, box[3].item() - y1]
                img_dict["score"] = score.item()
                retval.append(img_dict)

        if len(retval) == 0:
            return COCO()
        breakpoint()
        return cocoGt.loadRes(retval)


class UltralyticsYolo(torch.nn.Module):
    def __init__(
        self,
        name="yolov5s",
        num_classes=80,
        pretrained=True,
        autoshape=True,
        force_reload=False,
        gr=1,
        hyp=dict(),
        *args,
        **kwargs,
    ):

        super().__init__()

        # Model
        self.model = torch.hub.load(
            "ultralytics/yolov5" if name.startswith("yolov5") else "ultralytics/yolov3",
            name,
            classes=num_classes,
            pretrained=pretrained,
            autoshape=autoshape,
            force_reload=force_reload,
        )
        self.model.hyp = hyp
        self.model.gr = gr
        for name, param in self.model.named_parameters():
            param.requires_grad = True

    def _transformAnns(self, x, y):
        retval = []

        for x_elem, y_elem in zip(x, y):
            img_ann = []
            img_widh = x_elem.shape[2]
            img_height = x_elem.shape[1]

            for box, label in zip(
                (boxes for boxes in y_elem["boxes"]),
                (labels for labels in y_elem["labels"]),
            ):
                box_width = box[2] - box[0]
                box_height = box[3] - box[1]
                img_ann.append(
                    torch.tensor(
                        [
                            0,
                            label,
                            (box[0] + (box_width / 2)) / img_widh,
                            (box[1] + (box_height / 2)) / img_height,
                            box_width / img_widh,
                            box_height / img_height,
                        ]
                    )
                )
            retval.append(torch.stack(img_ann))
        return retval

    def forward(self, x, y=None):
        pad = (3, 3, 4, 5)

        if isinstance(x, (tuple, list)):
            retval = list()
            for x_elem in x:
                pad = (0, 1248 - x_elem.size()[2], 0, 384 - x_elem.size()[1])
                retval.append(self.model(F.pad(x_elem.unsqueeze(0), pad, "constant")))

            if self.training:
                ret_loss = []

                loss = ComputeLoss(self.model)
                y = self._transformAnns(x, y)

                for x_elem, y_elem in zip(retval, y):
                    ret_loss.append(loss(x_elem, y_elem.to(x_elem[0].device))[0])
                retval = dict(
                    zip((i for i in range(len(ret_loss))), (ret for ret in ret_loss))
                )

            return retval
        else:
            x = F.pad(x, pad, "constant")
            return self.model(x)

    def train(self, mode=True):
        super().train(mode)
        self.model.nms(not mode)

    def transformOutput(self, cocoGt, output, y):
        retval = []

        for out, y_img in zip(output, y):
            for ann in out[0].data:
                x1 = ann[0].item()
                y1 = ann[1].item()
                x2 = ann[2].item()
                y2 = ann[3].item()
                confidence = ann[4].item()
                label = ann[5].item()

                img_dict = dict()
                img_dict["image_id"] = cocoGt.getImgId(y_img["filename"])
                img_dict["category_id"] = label
                img_dict["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                img_dict["score"] = confidence
                retval.append(img_dict)

        if len(retval) == 0:
            return COCO()
        return cocoGt.loadRes(retval)
