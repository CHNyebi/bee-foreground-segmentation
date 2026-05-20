# Refined + BEE24-33 v5 Summary

## Data

- Base refined dataset: `data/bee_foreground_refined_20260521`
- Added annotation export: `C:\Users\dot\Downloads\20260521_2.zip`
- New prepared BEE24-33 annotations: 76 usable masks
- BEE24-33 split: 61 train, 15 val
- Merged dataset: `data/bee_foreground_refined_20260521_plus_bee33`
- Merged split: 202 train, 52 val

## Training

- Base checkpoint: `models/bee_foreground_unetpp_resnet18_imagenet_refined_v4/best_model.pt`
- New checkpoint: `models/bee_foreground_unetpp_resnet18_imagenet_refined_bee33_v5/best_model.pt`
- Architecture: Unet++ with ResNet18 encoder
- Input: 256 x 256 RGB resize
- Augmentation: strong
- Epochs: 60 finetune epochs
- Best training-scale epoch: 141
- Best training-scale val IoU: 0.819565
- Best training-scale val Dice: 0.900836

## Original-Scale Evaluation at Threshold 0.60

| Eval set | Samples | Model | Pixel IoU | Dice | Precision | Recall | Accuracy |
|---|---:|---|---:|---:|---:|---:|---:|
| refined val | 37 | v4 | 0.831921 | 0.908250 | 0.909957 | 0.906549 | 0.931735 |
| refined val | 37 | v5 | 0.834848 | 0.909992 | 0.909773 | 0.910210 | 0.932889 |
| BEE24-33 val | 15 | v4 | 0.685459 | 0.813380 | 0.713513 | 0.945752 | 0.857650 |
| BEE24-33 val | 15 | v5 | 0.790616 | 0.883066 | 0.882778 | 0.883354 | 0.923264 |
| combined val | 52 | v4 | 0.809297 | 0.894598 | 0.878317 | 0.911494 | 0.921299 |
| combined val | 52 | v5 | 0.829158 | 0.906600 | 0.906362 | 0.906839 | 0.931536 |

## Visual Reports

- `bee33_val_v4_vs_v5_t060.jpg`: all 15 BEE24-33 validation samples.
- `combined_val_top30_delta_v4_vs_v5_t060.jpg`: 30 combined-val samples with largest IoU improvement.
- `combined_val_delta_iou.csv`: per-image v4/v5 IoU delta table.

Overlay colors: green = true positive bee, red = background predicted as bee, magenta = missed bee.
