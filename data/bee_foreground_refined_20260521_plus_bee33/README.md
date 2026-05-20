# bee_foreground_refined_20260521_plus_bee33

This is the v5 foreground-mask dataset.

- Total: 254 labeled image/mask pairs
- Train: 202
- Val: 52
- Source pool: ReID `train_20260501` only

Composition:

- 178 refined samples kept from the original v2 dataset. The user refined masks and removed 86 unclear/inconsistent v2 samples. The old v2 split was preserved for these retained samples: 141 train, 37 val.
- 76 newly annotated BEE24-33 samples. These were selected from `train_20260501`, labeled in CVAT, and split into 61 train, 15 val.

Use `annotation_registry.csv` as the de-duplication registry before sampling future annotation batches.
