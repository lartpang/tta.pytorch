import sys
from typing import List

import torch

sys.path.append("../")
from tta_pytorch import FLIP_MODES, TYPES, Chain, Compose, Flip, Merger, Rescale, Resize, Rotate

tta_trans = Compose(
    [
        Rescale(
            scales=[0.5],
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
        Flip(
            flip_modes=[
                FLIP_MODES.Horizontal,
                FLIP_MODES.HorizontalVertical,
                FLIP_MODES.Identity,
            ]
        ),
        Rotate(
            angles=[15, 45],
            image_mode="bilinear",
            mask_mode="bilinear",
        ),
    ],
    verbose=True,
)

image = torch.randn(3, 1, 50, 50, dtype=torch.float32)


def base_usage():
    tta_results = Merger()
    for trans in tta_trans:
        trans: Chain
        aug_image = trans.do_image(image)
        undo_image = trans.undo_image(aug_image)
        tta_results.append(undo_image)

    seg_results = tta_results.result
    print(seg_results.shape)
    return seg_results


def enhanced_usage1():
    tta_results = Merger()
    for trans in tta_trans:
        trans: Chain
        aug_images: List[torch.Tensor] = trans.do_all(inputs=[image], input_types=[TYPES.IMAGE])
        undo_images = trans.undo_all(outputs=aug_images, output_types=[TYPES.MASK])
        tta_results.append(undo_images[0])

    seg_results = tta_results.result
    print(seg_results.shape)
    return seg_results


def enhanced_usage2():
    # just a list with a merging function
    tta_seg_merger = Merger(mode="mean")
    tta_cls_merger = Merger(mode="mean")

    tta_seg_merger.reset()
    tta_cls_merger.reset()
    for tran in tta_trans:
        tran: Chain
        aug_tensor = tran.do_image(image)
        # simulate real data
        mask = aug_tensor
        label = torch.randn(3, 1000, dtype=torch.float32)

        # for segmentation, [B,K,H,W]
        undo_mask = tran.undo_image(mask)
        tta_seg_merger.append(undo_mask)
        # for classification, [B,K]
        undo_label = tran.undo_label(label)
        tta_cls_merger.append(undo_label)

    seg_results = tta_seg_merger.result
    seg_mask = seg_results.argmax(dim=1)  # [B,H,W]
    cls_results = tta_cls_merger.result
    cls_score, cls_index = cls_results.max(dim=1)  # [B], [B]
    print(seg_mask.shape, cls_score.shape, cls_index.shape)
    return seg_results


def enhanced_usage3():
    tta_seg_results = []
    tta_cls_results = []
    for tran in tta_trans:
        tran: Chain
        aug_tensor = tran.do_image(image)
        # simulate real data
        mask = aug_tensor
        label = torch.randn(3, 1000, dtype=torch.float32)

        # for segmentation, [B,K,H,W]
        undo_mask = tran.undo_image(mask)
        tta_seg_results.append(undo_mask)
        # for classification, [B,K]
        undo_label = tran.undo_label(label)
        tta_cls_results.append(undo_label)

    seg_results = sum(tta_seg_results) / len(tta_seg_results)
    seg_mask = seg_results.argmax(dim=1)  # [B,H,W]
    cls_results = sum(tta_cls_results) / len(tta_cls_results)
    cls_score, cls_index = cls_results.max(dim=1)  # [B], [B]
    print(seg_mask.shape, cls_score.shape, cls_index.shape)
    return seg_results


def enhanced_usage4():
    @tta_trans.decorate(
        input_infos={"image": TYPES.IMAGE},
        output_infos={"mask": TYPES.MASK, "label": TYPES.LABEL},
        merge_mode="mean",
    )
    def do_something(image=None):
        label = torch.randn(3, 1000, dtype=torch.float32)
        return {"mask": image, "label": label}

    tta_results = do_something(image=image)
    print({k: v.shape for k, v in tta_results.items()})
    return tta_results["mask"]


if __name__ == "__main__":
    results0 = base_usage()
    results1 = enhanced_usage1()
    results2 = enhanced_usage2()
    results3 = enhanced_usage3()
    results4 = enhanced_usage4()

    assert torch.allclose(results0, results1)
    assert torch.allclose(results0, results2)
    assert torch.allclose(results0, results3)
    assert torch.allclose(results0, results4)
    print("All tests passed!")

    print(tta_trans)
