# Model Matrix 2026-05-16

Four variants were trained or evaluated on the fixed 53-image validation split.

| ID | Model | Image Size | Preprocess | Threshold | Pixel IoU | Dice | Precision | Recall | Macro IoU | Notes |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | Unet++ resnet18 | 256 | resize | 0.50 | 0.800793 | 0.889378 | 0.887128 | 0.891640 | 0.780676 | Current released checkpoint |
| B | Unet++ resnet18 | 384 | resize | 0.45 | 0.798212 | 0.887784 | 0.889880 | 0.885698 | 0.774570 | More conservative than A |
| C | Unet++ resnet34 | 384 | resize | 0.50 | 0.797990 | 0.887647 | 0.877015 | 0.898540 | 0.769579 | Higher recall, checkpoint is over GitHub's 100 MB single-file limit |
| D | Unet++ resnet18 | 384 | letterbox | 0.65 | 0.800284 | 0.889064 | 0.894507 | 0.883688 | 0.777365 | Nearly tied with A after threshold tuning |

With the same postprocessing used by the prediction pipeline and each model's best threshold:

| Rank | Model | Threshold | Pixel IoU | Dice | Precision | Recall |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | A current 256 resize | 0.50 | 0.800686 | 0.889312 | 0.886587 | 0.892054 |
| 2 | D resnet18 384 letterbox | 0.65 | 0.800412 | 0.889143 | 0.894027 | 0.884312 |
| 3 | B resnet18 384 resize | 0.45 | 0.798672 | 0.888068 | 0.889903 | 0.886241 |
| 4 | C resnet34 384 resize | 0.50 | 0.797837 | 0.887552 | 0.876113 | 0.899294 |

Conclusion: keep the current A checkpoint as the released default. D is close enough to revisit after adding more difficult labeled samples, but this run does not justify replacing the current weight.

Artifacts:

- `comparison_pixel_micro_iou.csv`
- `comparison_image_macro_iou.csv`
- `threshold_sweep_summary.csv`
- `comparison_postprocess_best_threshold.csv`
