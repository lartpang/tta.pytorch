import sys
from functools import partial
from typing import List

import torch

sys.path.append("../")
from tta_pytorch import TYPES, Compose, Flip, HFlip, Merger, Rescale, Resize, VFlip

tta_trans = Compose(
    [
        Rescale(
            scales=[0.5, 1.25],
            image_mode="bilinear",
            image_align_corners=False,
            mask_mode="bilinear",
            mask_align_corners=False,
        ),
        Resize(
            sizes=[128],
            image_mode="bilinear",
            image_align_corners=False,
            mask_mode="bilinear",
            mask_align_corners=False,
        ),
        Flip(),
        HFlip(),
    ],
    verbose=True,
)


def base_usage():
    image = torch.randn(3, 1, 50, 50, dtype=torch.float32)
    tta_results = Merger()
    for trans in tta_trans:
        aug_image = trans.do_image(image)
        deaug_image = trans.undo_image(aug_image)
        tta_results.append(mask=deaug_image)
    print({k: v.shape for k, v in tta_results.result.items()})


def enhanced_usage1():
    image = torch.randn(3, 1, 50, 50, dtype=torch.float32)
    tta_results = Merger()
    for trans in tta_trans:
        aug_images: List[torch.Tensor] = trans.do_all(inputs=[image], input_types=[TYPES.IMAGE])
        deaug_images = trans.undo_all(outputs=aug_images, output_types=[TYPES.MASK])
        tta_results.append(mask=deaug_images[0])
    print({k: v.shape for k, v in tta_results.result.items()})


def enhanced_usage2():
    @partial(
        tta_trans.decorate,
        input_infos={"image": TYPES.IMAGE},
        output_infos={"mask": TYPES.MASK, "label": TYPES.LABEL},
    )
    def do_something(image=None):
        h, w = image.shape[-2:]
        mask = torch.randn(3, 20, h, w, dtype=torch.float32)
        label = torch.randn(3, 1000, dtype=torch.float32)
        return {"mask": mask, "label": label}

    image = torch.randn(3, 1, 50, 50, dtype=torch.float32)
    tta_results = do_something(image=image)
    print({k: v.shape for k, v in tta_results.items()})


if __name__ == "__main__":
    base_usage()
    enhanced_usage1()
    enhanced_usage2()
