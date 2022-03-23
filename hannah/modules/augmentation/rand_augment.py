import math
from typing import List, Dict, Optional, Tuple, Dict, Any
from torchvision.transforms import functional as F, InterpolationMode
import torch, torch.nn
from torch import Tensor


def _apply_op(
    img: Tensor,
    op_name: str,
    magnitude: float,
    interpolation: InterpolationMode,
    fill: Optional[List[float]],
):
    if op_name == "ShearX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(magnitude), 0.0],
            interpolation=interpolation,
            fill=fill,
        )
    elif op_name == "ShearY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(magnitude)],
            interpolation=interpolation,
            fill=fill,
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img


class RandAugment(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
        config: Dict[str, Any] = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.config = config

    def _augmentation_space(
        self, num_bins: int, image_size: List[int]
    ) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: [(magnitudes, signed), prob]
            "Identity": [
                (torch.tensor(0.0), False),
                self.config.augmentations.rand_augment.probs.Identity,
            ],
            "ShearX": [
                (torch.linspace(0.0, 0.3, num_bins), True),
                self.config.augmentations.rand_augment.probs.ShearX,
            ],
            "ShearY": [
                (torch.linspace(0.0, 0.3, num_bins), True),
                self.config.augmentations.rand_augment.probs.ShearY,
            ],
            "TranslateX": [
                (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
                self.config.augmentations.rand_augment.probs.TranslateX,
            ],
            "TranslateY": [
                (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
                self.config.augmentations.rand_augment.probs.TranslateY,
            ],
            "Rotate": [
                (torch.linspace(0.0, 30.0, num_bins), True),
                self.config.augmentations.rand_augment.probs.Rotate,
            ],
            "Brightness": [
                (torch.linspace(0.0, 0.9, num_bins), True),
                self.config.augmentations.rand_augment.probs.Brightness,
            ],
            "Contrast": [
                (torch.linspace(0.0, 0.9, num_bins), True),
                self.config.augmentations.rand_augment.probs.Contrast,
            ],
            "Sharpness": [
                (torch.linspace(0.0, 0.9, num_bins), True),
                self.config.augmentations.rand_augment.probs.Sharpness,
            ],
            "Posterize": [
                (
                    8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(),
                    False,
                ),
                self.config.augmentations.rand_augment.probs.Posterize,
            ],
            "Solarize": [
                (torch.linspace(255.0, 0.0, num_bins), False),
                self.config.augmentations.rand_augment.probs.Solarize,
            ],
            "AutoContrast": [
                (torch.tensor(0.0), False),
                self.config.augmentations.rand_augment.probs.AutoContrast,
            ],
            "Equalize": [
                (torch.tensor(0.0), False),
                self.config.augmentations.rand_augment.probs.Equalize,
            ],
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]
        for _ in range(self.num_ops):
            op_meta = self._augmentation_space(
                self.num_magnitude_bins, F.get_image_size(img)
            )
            probs = torch.tensor([p[1] for p in list(op_meta.values())])
            op_index = torch.multinomial(probs, 1)
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name][0]
            magnitude = (
                float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            )
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(
                img, op_name, magnitude, interpolation=self.interpolation, fill=fill
            )
        return img
