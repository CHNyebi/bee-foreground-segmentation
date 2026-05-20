# Bee Foreground Segmentation

这个仓库保存蜜蜂 ReID 裁剪图的前景分割模型、人工 mask 数据集和训练/评估脚本。模型用途很窄：给 ReID 图像生成 `bee` 前景 mask，用于去背景预处理；它不是 ReID 模型本身。

当前 GitHub 主分支只保留两版发布数据和权重：

| Version | Dataset | Weight | Status |
|---|---|---|---|
| v2 baseline | `data/bee_foreground_v2` | `models/bee_foreground_unetpp_resnet18_v2/best_model.pt` | 旧服务器可继续使用 |
| v5 refined + BEE24-33 | `data/bee_foreground_refined_20260521_plus_bee33` | `models/bee_foreground_unetpp_resnet18_imagenet_refined_bee33_v5/best_model.pt` | 当前推荐 |

说明：之前曾提交过的 `bee_foreground_unetpp_resnet18_imagenet_v3` 不再作为当前发布版本保留。Git 历史里仍能看到旧提交，但当前文件树只保留 v2 和 v5。

## Quick Use

用当前推荐的 v5 权重：

```powershell
python scripts\predict_bee_foreground_segmentation.py `
  --input-dir D:\personal_resources\CV_research\AutoDL\train_20260501 `
  --checkpoint models\bee_foreground_unetpp_resnet18_imagenet_refined_bee33_v5\best_model.pt `
  --output-dir outputs\train_20260501_masked_v5 `
  --threshold 0.60 `
  --batch-size 64
```

如果要和旧服务器保持一致，继续用 v2：

```powershell
python scripts\predict_bee_foreground_segmentation.py `
  --input-dir <raw_reid_images> `
  --checkpoint models\bee_foreground_unetpp_resnet18_v2\best_model.pt `
  --output-dir <masked_output> `
  --threshold 0.50 `
  --batch-size 64
```

输出目录包含：

- `labels/`: 二值 mask，非零像素是 bee。
- `masked/`: 背景置黑后的图像。
- `overlays/`: 抽查用叠加图。
- `prediction_report.csv`: 每张图的面积、置信度等统计。
- `summary.json`: 本次推理配置摘要。

## Model Structure

两版都是二分类前景分割模型：

- Architecture: `UnetPlusPlus`
- Encoder: `resnet18`
- Input: RGB image, resized to `256 x 256`
- Output: one-channel foreground logit; sigmoid 后阈值化得到 bee mask
- Normalization: ImageNet mean/std
- Classes: `background`, `bee`

## Dataset Versions

### v2 baseline dataset

路径：

```text
data/bee_foreground_v2/
  dataset/train/images  # 211
  dataset/train/masks   # 211
  dataset/val/images    # 53
  dataset/val/masks     # 53
  annotation_registry.csv
  train.csv
  val.csv
  usable_annotations.csv
  dataset_summary.json
```

构成：

- 总数 264 张人工 mask。
- 全部样本从 ReID 原始训练集 `train_20260501` 抽取。
- split 为 211 train / 53 val。
- 这是最早可用基线，里面保留了一些后来认为不够清楚、标注不够一致的样本。

### v5 refined + BEE24-33 dataset

路径：

```text
data/bee_foreground_refined_20260521_plus_bee33/
  dataset/train/images  # 202
  dataset/train/masks   # 202
  dataset/val/images    # 52
  dataset/val/masks     # 52
  annotation_registry.csv
  train.csv
  val.csv
  usable_annotations.csv
  dataset_summary.json
```

构成：

- 总数 254 张人工 mask，202 train / 52 val。
- 第一部分是从 v2 里重新精修后的样本：保留 178 张，删除 86 张不清楚或不适合继续用的样本；保留样本沿用原 v2 train/val split，得到 141 train / 37 val。
- 第二部分是新增 BEE24-33 序列样本：从 `train_20260501` 里筛选、人工标注后得到 76 张，按 61 train / 15 val 加入。
- 相比 v2，v5 数据集更小但标注更干净，并显著增加了 BEE24-33 复杂背景覆盖。

新增数据时，用最新版 registry 去重：

```powershell
--registry data\bee_foreground_refined_20260521_plus_bee33\annotation_registry.csv
```

去重依据包括 sample id 和图像 SHA256，避免同一张图重复进入后续标注批次。

## Training Configs

### v2 training config

v2 是旧基线，训练配置来自 `models/bee_foreground_unetpp_resnet18_v2/summary.json` 和 `configs/bee_foreground_v2.json`：

- Dataset: `data/bee_foreground_v2/dataset`
- Architecture: `UnetPlusPlus`
- Encoder: `resnet18`
- Encoder weights: `none` / from scratch
- Image size: `256`
- Preprocess: direct resize
- Augmentation: current script's `light` level is the closest match
- Batch size: `4`
- Optimizer: `AdamW`
- Learning rate: `3e-4`
- Loss: `BCEWithLogitsLoss + DiceLoss`
- Requested epochs: `80`
- Completed epochs: `58`
- Best epoch: `41`
- Device: CUDA, with cuDNN disabled on the local Windows training run

Closest current-script command:

```powershell
python scripts\train_bee_foreground_segmentation.py `
  --dataset-dir data\bee_foreground_v2\dataset `
  --output-dir outputs\retrain_v2_baseline `
  --encoder resnet18 `
  --encoder-weights none `
  --image-size 256 `
  --batch-size 4 `
  --epochs 58 `
  --lr 3e-4 `
  --aug-level light `
  --disable-cudnn `
  --num-workers 0
```

注意：当前训练脚本后续加过 `--aug-level`、`--resume-checkpoint`、`letterbox` 等能力。用当前脚本能复现 v2 的主要配置，但不保证 bit-for-bit 得到同一个权重。

v2 指标：

- Training-size best val IoU/Dice: `0.7765 / 0.8742`
- Original-scale v2 val Pixel IoU/Dice at threshold 0.50: `0.8008 / 0.8894`

### v5 training config

v5 权重是后续迭代后的发布 artifact，不是从 v2 单次端到端训练得到的。实际历史链路是：

1. 先在更干净的 refined 标注上训练/微调中间模型。
2. 再加入 76 张 BEE24-33 新标注。
3. 最后一阶段从本地中间 checkpoint 继续微调 60 epoch，使用 strong augmentation。

最终 v5 发布权重：

```text
models/bee_foreground_unetpp_resnet18_imagenet_refined_bee33_v5/best_model.pt
```

最后一阶段训练配置：

- Dataset: `data/bee_foreground_refined_20260521_plus_bee33/dataset`
- Base checkpoint used locally: `models/bee_foreground_unetpp_resnet18_imagenet_refined_v4/best_model.pt`
- Architecture: `UnetPlusPlus`
- Encoder: `resnet18`
- Encoder initialization lineage: ImageNet-pretrained encoder in the earlier chain
- Image size: `256`
- Preprocess: direct resize
- Augmentation: `strong`
- Batch size: `4`
- Learning rate: `1e-4`
- Finetune epochs: `60`
- Best epoch in continued numbering: `141`
- Device: CUDA, cuDNN disabled

Historical final-stage command:

```powershell
python scripts\train_bee_foreground_segmentation.py `
  --dataset-dir data\bee_foreground_refined_20260521_plus_bee33\dataset `
  --output-dir outputs\refined_plus_bee33_20260521\finetune_v4_strongaug_lr1e4_e60 `
  --encoder resnet18 `
  --encoder-weights none `
  --resume-checkpoint models\bee_foreground_unetpp_resnet18_imagenet_refined_v4\best_model.pt `
  --image-size 256 `
  --batch-size 4 `
  --epochs 60 `
  --lr 1e-4 `
  --aug-level strong `
  --disable-cudnn `
  --num-workers 0
```

Because the intermediate v4 checkpoint is not published in the current two-version release, v5 is best treated as a fixed downloadable weight. You can train a comparable v5-style model from ImageNet initialization with the same dataset, but it will not exactly reproduce the published v5 checkpoint:

```powershell
python scripts\train_bee_foreground_segmentation.py `
  --dataset-dir data\bee_foreground_refined_20260521_plus_bee33\dataset `
  --output-dir outputs\train_v5_style_from_imagenet `
  --encoder resnet18 `
  --encoder-weights imagenet `
  --image-size 256 `
  --batch-size 4 `
  --epochs 80 `
  --lr 1e-4 `
  --aug-level strong `
  --disable-cudnn `
  --num-workers 0
```

## Evaluation

Evaluate any checkpoint:

```powershell
python scripts\evaluate_bee_foreground_segmentation.py `
  --checkpoint models\bee_foreground_unetpp_resnet18_imagenet_refined_bee33_v5\best_model.pt `
  --dataset-dir data\bee_foreground_refined_20260521_plus_bee33\dataset `
  --split val `
  --threshold 0.60 `
  --output-dir outputs\eval_v5_combined_val
```

Overlay colors:

- Green: predicted bee and labeled bee.
- Red: background predicted as bee.
- Magenta: bee missed by prediction.

Original-scale evaluation at threshold `0.60`:

| Eval set | Samples | Model | Pixel IoU | Dice | Precision | Recall | Accuracy |
|---|---:|---|---:|---:|---:|---:|---:|
| refined old val | 37 | v4 local predecessor | 0.8319 | 0.9083 | 0.9100 | 0.9065 | 0.9317 |
| refined old val | 37 | v5 | 0.8348 | 0.9100 | 0.9098 | 0.9102 | 0.9329 |
| BEE24-33 val | 15 | v4 local predecessor | 0.6855 | 0.8134 | 0.7135 | 0.9458 | 0.8577 |
| BEE24-33 val | 15 | v5 | 0.7906 | 0.8831 | 0.8828 | 0.8834 | 0.9233 |
| combined v5 val | 52 | v5 | 0.8292 | 0.9066 | 0.9064 | 0.9068 | 0.9315 |

The main v5 gain is on BEE24-33: it greatly reduces false positives on the difficult BEE24-33 backgrounds while keeping the older refined validation set roughly unchanged.

## Adding More Labels

Recommended policy:

- Sample new foreground-mask labels only from ReID `train_20260501`.
- Do not train the foreground model on ReID `val_20260501` or `eval`.
- Keep each published dataset as an immutable version. Add new data by creating a new dataset directory.
- Use the latest `annotation_registry.csv` to avoid duplicate images.

Typical flow:

```powershell
python tools\select_unlabeled_candidates.py `
  --source-root D:\personal_resources\CV_research\AutoDL\train_20260501 `
  --registry data\bee_foreground_refined_20260521_plus_bee33\annotation_registry.csv `
  --sample-size 100 `
  --seed 20260521 `
  --output outputs\candidate_batches\fg_next_candidates_100.csv

python tools\build_cvat_image_package.py `
  --candidates-csv outputs\candidate_batches\fg_next_candidates_100.csv `
  --output-dir outputs\cvat_batches\fg_next_batch001
```

In CVAT:

- Create only one label: `bee`.
- Upload `images.zip`.
- Annotate the target bee mask only.
- Export as `Segmentation mask 1.1`.

Convert the CVAT export:

```powershell
python scripts\prepare_bee_foreground_seg_dataset.py `
  --annotations-zip "C:\Users\dot\Downloads\job_xxx_annotations_segmentation mask 1.1.zip" `
  --subset-dir outputs\cvat_batches\fg_next_batch001 `
  --output-dir outputs\manual_datasets\fg_next_batch001 `
  --foreground-label bee `
  --val-fraction 0.20
```

Merge into a new version:

```powershell
python tools\merge_foreground_datasets.py `
  --base-root data\bee_foreground_refined_20260521_plus_bee33 `
  --add-root outputs\manual_datasets\fg_next_batch001 `
  --output-root data\bee_foreground_next

python tools\build_annotation_registry.py `
  --dataset-dir data\bee_foreground_next\dataset `
  --metadata-csv data\bee_foreground_next\usable_annotations.csv `
  --output data\bee_foreground_next\annotation_registry.csv
```
