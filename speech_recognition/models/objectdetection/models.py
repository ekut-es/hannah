import torchvision
import torch
import torch.nn.functional as F

from pycocotools.coco import COCO

from .loss import ComputeLoss

from speech_recognition.datasets.Kitti import KittiCOCO


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
        return cocoGt.loadRes(retval)


class YoloV5s(torch.nn.Module):
    def __init__(
        self,
        name="yolo-v5-s",
        num_classes=80,
        pretrained=True,
        autoshape=True,
        force_reload=False,
        *args,
        **kwargs,
    ):

        super().__init__()

        # Model
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "yolov5s",
            classes=num_classes,
            pretrained=pretrained,
            autoshape=autoshape,
            force_reload=force_reload,
        )

    def _transformAnns(self, y):
        retval = []

        for elem in y:
            img_ann = []
            for box, label in zip(
                (boxes for boxes in elem["boxes"]),
                (labels for labels in elem["labels"]),
            ):
                img_ann.append(
                    torch.tensor(
                        [0, label, box[0], box[1], box[2] - box[0], box[3] - box[1]]
                    )
                )
            retval.append(torch.stack(img_ann))
        return retval

    def forward(self, x, y=None):
        pad = (3, 3, 4, 5)

        # if self...training -> nur loss zurückgeben (zuvor berechnen)
        # wie loss berechnen im YOLO Repro schauen (idealerweise ComputeLoss klasse verwenden, zu Not kopieren)

        if isinstance(x, (tuple, list)):
            retval = list()
            for x_elem in x:
                pad = (0, 1248 - x_elem.size()[2], 0, 384 - x_elem.size()[1])
                retval.append(self.model(F.pad(x_elem.unsqueeze(0), pad, "constant")))

            if self.training:
                ret_loss = []
                loss = ComputeLoss(self.model)
                y = self._transformAnns(y)

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
