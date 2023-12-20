import sys

import torch

sys.path.append("../")
from tta_pytorch import Compose, HFlip, Merger, Rescale, VFlip


def base_usage():
    tta_trans = Compose(
        [
            Rescale(scales=[0.5, 1.25]),
            Rescale(scales=[1, 1.5]),
            HFlip(),
            VFlip(),
        ],
        verbose=True,
    )
    image = torch.randn(3, 1, 50, 50, dtype=torch.float32)
    tta_results = Merger()
    for trans in tta_trans:
        aug_image = trans.do(image=image)
        aug_image = aug_image["image"]
        deaug_image = trans.undo(mask=aug_image)
        deaug_image = deaug_image["mask"]
        tta_results.append(mask=deaug_image)
    print({k: v.shape for k, v in tta_results.result.items()})


def enhanced_usage():
    tta_trans = Compose(
        [
            Rescale(scales=[0.5, 1.25]),
            Rescale(scales=[1, 1.5]),
            HFlip(),
            VFlip(),
        ],
        verbose=True,
    )

    @tta_trans.decorate
    def do_something(image):
        return {"mask": image}

    image = torch.randn(3, 1, 50, 50, dtype=torch.float32)
    tta_results = do_something(image=image)
    print({k: v.shape for k, v in tta_results.items()})


if __name__ == "__main__":
    base_usage()
    enhanced_usage()
