# Test-Time Augmentation library for Pytorch

## Features

- [x] Support image segmentation task.
- [x] Support image classification task.

## Requirements

- torch
- torchvision

## Usage

More details can be found in the examples:
- [usage.py](./examples/usage.py)

First, we need to import all necessary classes and define a TTA transform and a dummy image.

```python
from tta_pytorch import TYPES, Chain, Compose, Flip, HFlip, Merger, Rescale, Resize, VFlip

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
        Flip(),
        HFlip(),
        # VFlip(),
    ],
    verbose=True,
)

image = torch.randn(3, 1, 50, 50, dtype=torch.float32)
```

Next, we can use the TTA transform to augment the image and get the deaugmented results.

Here are some examples of how to use the TTA transform.

1. Basic usage.

```python
tta_results = Merger()
for trans in tta_trans:
    trans: Chain
    aug_image = trans.do_image(image)
    undo_image = trans.undo_image(aug_image)
    tta_results.append(undo_image)

seg_results = tta_results.result
```

2. Usage with the Merger class and the integrated augmentation and deaugmentation pipline.

```python
tta_results = Merger()
for trans in tta_trans:
    trans: Chain
    aug_images: List[torch.Tensor] = trans.do_all(inputs=[image], input_types=[TYPES.IMAGE])
    undo_images = trans.undo_all(outputs=aug_images, output_types=[TYPES.MASK])
    tta_results.append(undo_images[0])

seg_results = tta_results.result
```

3. Usage with the Merger class for segmentation and classification, and the seperate augmentation and deaugmentation piplines.

```python
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
```

4. Usage with the built-in list and the seperate augmentation and deaugmentation piplines.

```python
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
```

5. Usage with the decorator.

```python
@tta_trans.decorate(
    input_infos={"image": TYPES.IMAGE},
    output_infos={"mask": TYPES.MASK, "label": TYPES.LABEL},
    merge_mode="mean",
)
def do_something(image=None):
    label = torch.randn(3, 1000, dtype=torch.float32)
    return {"mask": image, "label": label}

tta_results = do_something(image=image)
```

## Cite

If you find this library useful, please cite our bibtex:

```bibtex
@online{tta.pytorch,
    author="lartpang",
    title="{Test-Time Augmentation library for Pytorch}",
    url="https://github.com/lartpang/tta.pytorch",
    note="(Dec 20, 2023)",
}
```
