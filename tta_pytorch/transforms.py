import abc
from functools import partial
from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
from torchvision.transforms import functional as TF


class _BaseTransform:
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def do(self, param, *args, **kwargs):
        pass

    @abc.abstractmethod
    def undo(self, param, *args, **kwargs):
        pass

    def __len__(self):
        return len(self.params)

    def __add__(self, other):
        return self.num_paths + other.num_paths


class Rescale(_BaseTransform):
    def __init__(
        self,
        scales: List[Union[int, float]],
        image_mode: Optional[str] = "bilinear",
        image_align_corners: Optional[bool] = False,
        *,
        mask_name: Optional[str] = None,
        mask_mode: Optional[str] = "bilinear",
        mask_align_corners: Optional[bool] = False,
    ):
        """Multi-Scale Rescale Transformation.

        Args:
            scales (List[Union[int, float]]): _description_
            image_mode (Optional[str], optional): _description_. Defaults to "bilinear".
            image_align_corners (Optional[bool], optional): _description_. Defaults to False.
            mask_name (Optional[str], optional): _description_. Defaults to None.
            mask_mode (Optional[str], optional): _description_. Defaults to "bilinear".
            mask_align_corners (Optional[bool], optional): _description_. Defaults to False.
        """
        super().__init__()
        self.params = scales
        self.original_size = None

        self.image_mode = image_mode
        self.image_func = partial(
            F.interpolate,
            mode=image_mode,
            align_corners=image_align_corners,
        )

        self.mask_name = mask_name or ""
        if mask_name:
            self.mask_func = partial(
                F.interpolate,
                mode=mask_mode,
                align_corners=mask_align_corners,
            )

    def do(self, param: float, **data):
        for name, tensor in data.items():
            h, w = tensor.shape[-2:]
            tgt_hw = (int(h * param), int(w * param))
            if name == self.mask_name:
                data[name] = self.mask_func(tensor, size=tgt_hw)
            else:
                data[name] = self.image_func(tensor, size=tgt_hw)
        self.original_size = h, w
        return data

    def undo(self, param: float, **data):
        for name, tensor in data.items():
            if name == self.mask_name:
                data[name] = self.mask_func(tensor, size=self.original_size)
            else:
                data[name] = self.image_func(tensor, size=self.original_size)
        self.original_size = None
        return data


class Resize(_BaseTransform):
    def __init__(
        self,
        sizes: List[Tuple[int]],
        image_mode: Optional[str] = "bilinear",
        image_align_corners: Optional[bool] = False,
        *,
        mask_name: Optional[str] = None,
        mask_mode: Optional[str] = "bilinear",
        mask_align_corners: Optional[bool] = False,
    ):
        super().__init__()
        paired_sizes = []
        for s in sizes:
            assert len(s) in [1, 2], sizes
            if len(s) == 1:
                s = (s, s)
            paired_sizes.append(s)
        self.params = paired_sizes
        self.original_size = None

        self.image_mode = image_mode
        self.image_func = partial(
            F.interpolate,
            mode=image_mode,
            align_corners=image_align_corners,
        )

        self.mask_name = mask_name or ""
        if mask_name:
            self.mask_func = partial(
                F.interpolate,
                mode=mask_mode,
                align_corners=mask_align_corners,
            )

    def do(self, param: float, **data):
        for name, tensor in data.items():
            h, w = tensor.shape[-2:]
            if name == self.mask_name:
                data[name] = self.mask_func(tensor, size=param)
            else:
                data[name] = self.image_func(tensor, size=param)
        self.original_size = h, w
        return data

    def undo(self, param: float, **data):
        for name, tensor in data.items():
            if name == self.mask_name:
                data[name] = self.mask_func(tensor, size=self.original_size)
            else:
                data[name] = self.image_func(tensor, size=self.original_size)
        self.original_size = None
        return data


class HFlip(_BaseTransform):
    def __init__(self) -> None:
        super().__init__()
        self.params = [False, True]

    def do(self, param: bool, **data):
        if param:
            data = {k: TF.hflip(v) for k, v in data.items()}
        return data

    def undo(self, param: bool, **data):
        if param:
            data = {k: TF.hflip(v) for k, v in data.items()}
        return data


class VFlip(_BaseTransform):
    def __init__(self) -> None:
        super().__init__()
        self.params = [False, True]

    def do(self, param: bool, **data):
        if param:
            data = {k: TF.vflip(v) for k, v in data.items()}
        return data

    def undo(self, param: bool, **data):
        if param:
            data = {k: TF.vflip(v) for k, v in data.items()}
        return data
