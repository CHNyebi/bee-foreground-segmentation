# ImageNet Encoder Initialization Experiment

Date: 2026-05-20

Dataset: `data/bee_foreground_v2/dataset`

Model: `UnetPlusPlus`, `resnet18` encoder, 256x256 resize input, binary foreground output.

The previous v2 release trained the ResNet18 encoder from scratch. This experiment keeps the same architecture and dataset, but initializes the encoder with ImageNet weights by training with `--encoder-weights imagenet`.

## Released Checkpoint

`models/bee_foreground_unetpp_resnet18_imagenet_v3/best_model.pt`

The v2 checkpoint remains available at:

`models/bee_foreground_unetpp_resnet18_v2/best_model.pt`

## Training Result

Best epoch: 44

Training-size validation metrics at best epoch:

| metric | value |
| --- | ---: |
| IoU | 0.8089 |
| Dice | 0.8943 |
| Precision | 0.8925 |
| Recall | 0.8962 |
| Accuracy | 0.9063 |
| Loss | 0.4421 |

## Original-Scale Validation

The model was also evaluated by restoring predictions to each validation image's original resolution.

| model | pixel IoU | pixel Dice | precision | recall | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2, from scratch | 0.8008 | 0.8894 | 0.8871 | 0.8916 | 0.8983 |
| v3, ImageNet init | 0.8268 | 0.9052 | 0.8969 | 0.9136 | 0.9165 |
| v3, ImageNet init + postprocess | 0.8269 | 0.9053 | 0.8967 | 0.9140 | 0.9166 |

Threshold sweep on v3 found the best pixel IoU at threshold 0.55:

| threshold | pixel IoU | pixel Dice | precision | recall |
| --- | ---: | ---: | ---: | ---: |
| 0.55 | 0.8270 | 0.9053 | 0.9029 | 0.9078 |

The gain comes from initialization, not a larger model. This should be the default baseline before trying larger encoders or higher input resolution.
