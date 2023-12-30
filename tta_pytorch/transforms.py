from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .base import _BaseTransform
from .utils import make_int_pair


class Rescale(_BaseTransform):
    def __init__(
        self,
        scales: List[Union[int, float]],
        image_mode: Optional[str] = "bilinear",
        image_align_corners: Optional[bool] = False,
        mask_mode: Optional[str] = "bilinear",
        mask_align_corners: Optional[bool] = False,
    ):
        """Multi-Scale Rescale Transformation."""
        self.params = scales
        self.original_size = None
        self.image_mode = image_mode
        self.image_align_corners = image_align_corners
        self.mask_mode = mask_mode
        self.mask_align_corners = mask_align_corners

    def resize_image(self, image, size):
        return F.interpolate(
            image, size=size, mode=self.image_mode, align_corners=self.image_align_corners
        )

    def resize_mask(self, mask, size):
        return F.interpolate(
            mask, size=size, mode=self.mask_mode, align_corners=self.mask_align_corners
        )

    def do_image(self, image: torch.Tensor, param: float):
        h, w = image.shape[-2:]
        self.original_size = h, w
        return self.resize_image(image, size=(int(h * param), int(w * param)))

    def undo_image(self, image: torch.Tensor, param: float):
        return self.resize_image(image, size=self.original_size)

    def undo_mask(self, mask: torch.Tensor, param: float):
        return self.resize_mask(mask, size=self.original_size)

    def undo_label(self, label: torch.Tensor, param: float):
        return label


class Resize(Rescale):
    def __init__(
        self,
        sizes: List[Tuple[int]],
        image_mode: Optional[str] = "bilinear",
        image_align_corners: Optional[bool] = False,
        mask_mode: Optional[str] = "bilinear",
        mask_align_corners: Optional[bool] = False,
    ):
        """Multi-Scale Resize Transformation."""
        super().__init__(
            scales=None,
            image_mode=image_mode,
            image_align_corners=image_align_corners,
            mask_mode=mask_mode,
            mask_align_corners=mask_align_corners,
        )
        self.params = make_int_pair(sizes)

    def do_image(self, image: torch.Tensor, param: Union[Tuple[int, int], int]):
        h, w = image.shape[-2:]
        self.original_size = h, w
        return self.resize_image(image, size=param)


class Rotate(_BaseTransform):
    def __init__(
        self,
        angles: List[Union[int, float]],
        image_mode: Optional[str] = "bilinear",
        mask_mode: Optional[str] = "bilinear",
    ):
        """Rotate Transformation."""
        self.params = angles
        self.image_mode = InterpolationMode(image_mode)
        self.mask_mode = InterpolationMode(mask_mode)

    def rotate_image(self, image, angle):
        return TF.rotate(image, angle=angle, interpolation=self.image_mode, fill=0)

    def rotate_mask(self, mask, angle):
        return TF.rotate(mask, angle=angle, interpolation=self.mask_mode, fill=0)

    def do_image(self, image: torch.Tensor, param: float):
        return self.rotate_image(image, angle=param)

    def undo_image(self, image: torch.Tensor, param: float):
        return self.rotate_image(image, angle=-param)

    def undo_mask(self, mask: torch.Tensor, param: float):
        return self.rotate_mask(mask, angle=-param)

    def undo_label(self, label: torch.Tensor, param: float):
        return label


_FLIP_MODES = Enum("_FLIP_MODES", ["HFLIP", "VFLIP", "IDENTITY"])


class Flip(_BaseTransform):
    def __init__(self) -> None:
        """Wrapped horizontal, vertical and identity transformations."""
        self.params = [_FLIP_MODES.HFLIP, _FLIP_MODES.VFLIP, _FLIP_MODES.IDENTITY]

    def flip(self, image, mode: _FLIP_MODES):
        if mode is _FLIP_MODES.IDENTITY:
            return image
        elif mode is _FLIP_MODES.HFLIP:
            return TF.hflip(image)
        elif mode is _FLIP_MODES.VFLIP:
            return TF.vflip(image)
        else:
            raise ValueError(f"Invalid Mode: {mode}")

    def do_image(self, image: torch.Tensor, param: _FLIP_MODES):
        return self.flip(image, mode=param)

    def undo_image(self, image: torch.Tensor, param: _FLIP_MODES):
        return self.flip(image, mode=param)

    def undo_mask(self, mask: torch.Tensor, param: _FLIP_MODES):
        return self.flip(mask, mode=param)

    def undo_label(self, label: torch.Tensor, param: _FLIP_MODES):
        return label


class HFlip(Flip):
    def __init__(self) -> None:
        self.params = [_FLIP_MODES.HFLIP, _FLIP_MODES.IDENTITY]


class VFlip(Flip):
    def __init__(self) -> None:
        self.params = [_FLIP_MODES.VFLIP, _FLIP_MODES.IDENTITY]
