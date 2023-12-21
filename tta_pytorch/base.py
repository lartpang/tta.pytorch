import abc
from enum import Enum, unique
from typing import Any

import torch


@unique
class TYPES(Enum):
    IMAGE = "image"
    MASK = "mask"
    LABEL = "label"


class _BaseTransform:
    @abc.abstractmethod
    def do_image(self, image: torch.Tensor, param: Any):
        pass

    @abc.abstractmethod
    def undo_image(self, image: torch.Tensor, param: Any):
        pass

    @abc.abstractmethod
    def undo_mask(self, mask: torch.Tensor, param: Any):
        pass

    @abc.abstractmethod
    def undo_label(self, label: torch.Tensor, param: Any):
        pass
