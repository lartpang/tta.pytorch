import itertools
from functools import partial, wraps
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transforms import _BaseTransform


class Merger:
    # Modified from https://github.com/qubvel/ttach/blob/94e579e59a21cbdfbb4f5790502e648008ecf64e/ttach/base.py#L120
    support_modes = ("mean", "gmean", "sum", "max", "min", "tsharpen")

    def __init__(self, mode: str = "mean"):
        if mode not in self.support_modes:
            raise ValueError(f"Not correct merge type `{mode}`.")

        self.output = {}
        self.mode = mode
        self.num = 0

    def reset(self):
        self.output = {}
        self.num = 0

    def append(self, **data):
        for name, tensor in data.items():
            if tensor is None:
                # remove the kv pair in merged output with the value None
                continue

            if self.mode == "tsharpen":
                tensor = tensor**0.5

            if not self.output:
                self.output[name] = tensor
            elif self.mode in ["mean", "sum", "tsharpen"]:
                self.output[name] = self.output[name] + tensor
            elif self.mode == "gmean":
                self.output[name] = self.output[name] * tensor
            elif self.mode == "max":
                self.output[name] = F.max(self.output[name], tensor)
            elif self.mode == "min":
                self.output[name] = F.min(self.output[name], tensor)
            else:
                raise ValueError(f"Not correct merge mode `{self.mode}`.")

    @property
    def result(self) -> torch.Tensor:
        if self.mode in ["sum", "max", "min"]:
            result = self.output
        elif self.mode in ["mean", "tsharpen"]:
            result = {k: v / self.num for k, v in self.output.items()}
        elif self.mode in ["gmean"]:
            result = {k: v ** (1 / self.num) for k, v in self.output.items()}
        else:
            raise ValueError(f"Not correct merge mode `{self.mode}`.")
        return result


class Chain:
    def __init__(
        self,
        path: List[Tuple[int, _BaseTransform]],
        transforms: List[_BaseTransform],
        verbose: Optional[bool] = False,
    ) -> None:
        self.path = path
        self.num_paths = len(self.path)
        self.transforms = transforms
        self.verbose = verbose

    def do(self, **data):
        if self.verbose:
            shapes = {k: v.shape for k, v in data.items()}
            print(f">>> Input: {shapes}")

        for node_idx in range(self.num_paths):
            idx, p = self.path[node_idx]
            t = self.transforms[idx]
            data = t.do(param=p, **data)
            if self.verbose:
                t_name = t.__class__.__name__
                shapes = {k: v.shape for k, v in data.items()}
                print(f">>> Node {node_idx}: {t_name}.do({p}) for {shapes}")

        return data

    def undo(self, **data):
        for node_idx in range(self.num_paths - 1, -1, -1):
            idx, p = self.path[node_idx]
            t = self.transforms[idx]
            data = t.undo(param=p, **data)
            if self.verbose:
                t_name = t.__class__.__name__
                shapes = {k: v.shape for k, v in data.items()}
                print(f"<<< Node {node_idx}: {t_name}.undo({p}) for {shapes}")

        return data


class Compose:
    def __init__(
        self,
        transforms: List[_BaseTransform],
        verbose: Optional[bool] = False,
    ):
        self.paths = self.flatten(transforms)
        self.verbose = verbose
        self.chain = partial(Chain, transforms=transforms, verbose=verbose)

    @staticmethod
    def flatten(transforms):
        trans_params = []
        for idx, trans in enumerate(transforms):
            _params = [(idx, p) for p in trans.params]
            trans_params.append(_params)
        return itertools.product(*trans_params)

    def __iter__(self) -> _BaseTransform:
        for idx, path in enumerate(self.paths):
            if self.verbose:
                print(f"- TTA Path: {idx}")
            yield self.chain(path=path)

    def __len__(self) -> int:
        return len(self.paths)

    def decorate(self, func: nn.Module, merge_mode="mean"):
        tta_merger = Merger(mode=merge_mode)

        @wraps(func)
        def inner(**data):
            tta_merger.reset()
            for t in self:
                do_data = t.do(**data)
                do_outputs = func(**do_data)
                # undo all augmentation operations and merge all outputs
                undo_outputs = t.undo(**do_outputs)
                tta_merger.append(**undo_outputs)
            return tta_merger.result

        return inner
