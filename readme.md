# Test-Time Augmentation library for Pytorch

## Features

- [x] Support image segmentation task.
- [x] Support image classification task.

## Requirements

- torch
- torchvision

## Usage

### Currently Effect

See [usage.py](./examples/usage.py)

### Expected Effect

```python
import tta_torch as tta

tta_trans = tta.Compose([tta.Scale(multi_scales), tta.HorizontalFlip()])
```

```python
# just a list with a merging function
tta_seg_merger = tta.Merger(mode='mean')
tta_cls_merger = tta.Merger(mode='mean')

for image_tensor in loader:
    tta_seg_merger.reset()
    tta_cls_merger.reset()
    for tran in tta_trans:
        aug_image_tensor = tran.augment_image(image_tensor)
        aug_outputs = net(aug_image_tensor)
        # for segmentation, [B,K,H,W]
        deaug_mask = tran.deaugment_mask(aug_outputs['mask'])
        tta_seg_merger.append(deaug_mask)
        # for classification, [B,K]
        deaug_label = tran.deaugment_label(aug_outputs['label'])
        tta_cls_merger.append(deaug_label)

    seg_results = tta_seg_merger.result
    seg_mask = seg_results.argmax(dim=1) # [B,H,W]
    cls_results = tta_cls_merger.result
    cls_score, cls_index = outputs.max(dim=1) # [B], [B]
```

```python
for image_tensor in loader:
    tta_seg_results = []
    tta_cls_results = []
    for tran in tta_trans:
        aug_image_tensor = tran.augment_image(image_tensor)
        aug_outputs = net(aug_image_tensor)
        # for segmentation, [B,K,H,W]
        deaug_mask = tran.deaugment_mask(aug_outputs['mask'])
        tta_seg_results.append(deaug_mask)
        # for classification, [B,K]
        deaug_label = tran.deaugment_label(aug_outputs['label'])
        tta_cls_results.append(deaug_label)

    seg_results = sum(tta_seg_merger) / len(tta_seg_merger)
    seg_mask = seg_results.argmax(dim=1) # [B,H,W]
    cls_results = sum(tta_cls_merger) / len(tta_cls_merger)
    cls_score, cls_index = outputs.max(dim=1) # [B], [B]
```

```python
net_forward = tta_trans.decorate(func=net)

for image_tensor in loader:
    # only support single input and single output
    deaug_output = net_forward(image)
```

```python
@tta_trans.decorate
def net_forward(image):
    # do something
    output = net(image)
    # do something
    return output

for image_tensor in loader:
    # only support single input and single output
    deaug_output = net_forward(image)
```
