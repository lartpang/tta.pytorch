import itertools
from functools import partial, wraps
from typing import List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TYPES, _BaseTransform


class Merger:
    # Modified from https://github.com/qubvel/ttach/blob/94e579e59a21cbdfbb4f5790502e648008ecf64e/ttach/base.py#L120
    SUPPORT_MODES = ("mean", "gmean", "sum", "max", "min", "tsharpen")

    def __init__(self, mode: str = "mean"):
        if mode not in self.SUPPORT_MODES:
            raise ValueError(f"Not correct merge type `{mode}`.")

        self.mode = mode
        self.output = {}
        self.num = 0

    def reset(self):
        self.output = {}
        self.num = 0

    def append(self, **data):
        for name, tensor in data.items():
            print(name, tensor.shape)
            if tensor is None:
                # remove the kv pair in merged output with the value None
                continue

            if self.mode == "tsharpen":
                tensor = tensor**0.5

            if name not in self.output:
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

    def do_all(self, inputs: List[torch.Tensor], input_types: List[TYPES]):
        assert len(inputs) == len(input_types)
        assert all([isinstance(t, TYPES) for t in input_types])

        if self.verbose:
            shapes = {t.name: x.shape for x, t in zip(inputs, input_types)}
            print(f">>> Inputs: {shapes}")

        for node_idx in range(self.num_paths):
            idx, p = self.path[node_idx]
            t = self.transforms[idx]
            for i, i_type in enumerate(input_types):
                do_func = getattr(t, f"do_{i_type.value}")
                inputs[i] = do_func(inputs[i], param=p)

                if self.verbose:
                    t_name = t.__class__.__name__
                    print(
                        f">>> Node{node_idx}: {t_name}.do_{i_type.value}({p}) for {inputs[i].shape}"
                    )
        return inputs

    def undo_all(self, outputs: List[torch.Tensor], output_types: List[TYPES]):
        assert len(outputs) == len(output_types)
        assert all([isinstance(t, TYPES) for t in output_types])

        for node_idx in range(self.num_paths - 1, -1, -1):
            idx, p = self.path[node_idx]
            t = self.transforms[idx]
            for i, o_type in enumerate(output_types):
                undo_func = getattr(t, f"undo_{o_type.value}")
                outputs[i] = undo_func(outputs[i], param=p)

                if self.verbose:
                    t_name = t.__class__.__name__
                    print(
                        f">>> Node{node_idx}: {t_name}.do_{o_type.value}({p}) for {outputs[i].shape}"
                    )
        return outputs

    def do_image(self, image):
        if self.verbose:
            print(f">>> Image Input: {image.shape}")

        for node_idx in range(self.num_paths):
            idx, p = self.path[node_idx]
            t = self.transforms[idx]
            image = t.do_image(image=image, param=p)
            if self.verbose:
                t_name = t.__class__.__name__
                print(f">>> Node{node_idx}: {t_name}.do_image({p}) for {image.shape}")

        return image

    def undo_image(self, image):
        for node_idx in range(self.num_paths - 1, -1, -1):
            idx, p = self.path[node_idx]
            t = self.transforms[idx]
            image = t.undo_image(image=image, param=p)
            if self.verbose:
                t_name = t.__class__.__name__
                print(
                    f"<<< Image {node_idx}: {t_name}.undo_image({p}) for {image.shape}"
                )

        return image

    def undo_mask(self, mask):
        for node_idx in range(self.num_paths - 1, -1, -1):
            idx, p = self.path[node_idx]
            t = self.transforms[idx]
            mask = t.undo_mask(mask=mask, param=p)
            if self.verbose:
                t_name = t.__class__.__name__
                print(f"<<< Mask {node_idx}: {t_name}.undo_mask({p}) for {mask.shape}")

        return mask

    def undo_label(self, label):
        for node_idx in range(self.num_paths - 1, -1, -1):
            idx, p = self.path[node_idx]
            t = self.transforms[idx]
            label = t.undo_label(label=label, param=p)
            if self.verbose:
                t_name = t.__class__.__name__
                print(
                    f"<<< Label {node_idx}: {t_name}.undo_label({p}) for {label.shape}"
                )

        return label


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
        # Use lists to facilitate repeated calls later.
        return list(itertools.product(*trans_params))

    def __iter__(self) -> _BaseTransform:
        for idx, path in enumerate(self.paths):
            if self.verbose:
                print(f"- TTA Path: {idx}")
            yield self.chain(path=path)

    def __len__(self) -> int:
        return len(self.paths)

    def decorate(
        self,
        func: nn.Module,
        input_infos: Mapping[str, TYPES],
        output_infos: Mapping[str, TYPES],
        merge_mode="mean",
    ):
        if not all([isinstance(t, TYPES) for t in input_infos.values()]):
            raise ValueError(f"all types in {input_infos} must be one of TYPES!")
        if not all([isinstance(t, TYPES) for t in output_infos.values()]):
            raise ValueError(f"all types in {output_infos} must be one of TYPES!")

        tta_merger = Merger(mode=merge_mode)

        @wraps(func)
        def inner(**inputs):
            tta_merger.reset()
            for t in self:
                # do all augmentation operations for all inputs
                do_inputs = {}
                for input_name, i_type in input_infos.items():
                    do_func = getattr(t, f"do_{i_type.value}")
                    do_inputs[input_name] = do_func(inputs[input_name])

                # do something
                outputs = func(**do_inputs)
                assert isinstance(outputs, Mapping), "Only support the mapping type."

                # undo all augmentation operations and merge all outputs
                undo_outputs = {}
                for output_name, o_type in output_infos.items():
                    undo_func = getattr(t, f"undo_{o_type.value}")
                    undo_outputs[output_name] = undo_func(outputs[output_name])

                tta_merger.append(**undo_outputs)
            return tta_merger.result

        return inner
