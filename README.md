# Bee Foreground Segmentation

这个仓库保存当前蜜蜂前景分割模型、权重、人工标注数据集，以及后续继续加标注数据所需的去重和整理工具。

它的定位很明确：**给 ReID 数据集里的蜜蜂裁剪图生成 bee foreground mask**。当前模型只用于预处理 ReID 图片，不是 ReID 模型本身。

## 当前推荐版

- 模型：`UnetPlusPlus`
- Encoder：`resnet18`
- Encoder 初始化：`imagenet`
- 输入尺寸：`256x256`
- 预处理：直接 `resize`
- 类别：二分类，`bee` vs background
- 当前推荐权重：`models/bee_foreground_unetpp_resnet18_imagenet_v3/best_model.pt`
- 训练数据：264 张人工 mask 标注图
- 训练/验证：211 / 53
- 最佳 epoch：44

上一版从头训练的 v2 权重仍保留在：

```text
models/bee_foreground_unetpp_resnet18_v2/best_model.pt
```

ImageNet 预训练初始化的 v3 在同一验证集原图尺度上明显更好：

```text
v2 pixel IoU / Dice: 0.8008 / 0.8894
v3 pixel IoU / Dice: 0.8268 / 0.9052
```

当前这 264 张不是完整的 `train_20260501`，而是从 ReID 原始训练集 `train_20260501` 里抽出来并人工修正过 mask 的子集。

```text
data/bee_foreground_v2/
  dataset/
    train/images  # 211
    train/masks
    val/images    # 53
    val/masks
  annotation_registry.csv
  train.csv
  val.csv
  usable_annotations.csv
```

## 安装

```powershell
python -m pip install -r requirements.txt
```

如果在 CUDA 服务器上复现，优先安装和服务器 CUDA 匹配的 PyTorch，再安装其它依赖。本机原始训练环境是 `torch 1.11.0+cu113` / `torchvision 0.12.0+cu113`。

## 直接使用当前权重

对 ReID 的一个图片目录生成 mask、扣背景图和 overlay：

```powershell
python scripts/predict_bee_foreground_segmentation.py `
  --input-dir D:\personal_resources\CV_research\AutoDL\train_20260501 `
  --checkpoint models\bee_foreground_unetpp_resnet18_imagenet_v3\best_model.pt `
  --output-dir outputs\train_20260501_masked_v3 `
  --batch-size 64
```

输出：

- `labels/`：二值 mask，非零像素是 bee
- `masked/`：背景置黑后的图
- `overlays/`：抽查用叠加图
- `prediction_report.csv`：每张图的面积、置信度等统计
- `summary.json`：本次推理摘要

如果要处理 ReID 三个 split，就分别跑：

```powershell
python scripts/predict_bee_foreground_segmentation.py --input-dir <raw_train> --checkpoint models\bee_foreground_unetpp_resnet18_imagenet_v3\best_model.pt --output-dir <masked_train>
python scripts/predict_bee_foreground_segmentation.py --input-dir <raw_val>   --checkpoint models\bee_foreground_unetpp_resnet18_imagenet_v3\best_model.pt --output-dir <masked_val>
python scripts/predict_bee_foreground_segmentation.py --input-dir <raw_eval>  --checkpoint models\bee_foreground_unetpp_resnet18_imagenet_v3\best_model.pt --output-dir <masked_eval>
```

注意：训练 mask 模型的人工标签只来自 ReID train；把冻结后的 mask 模型应用到 ReID val/eval 是可以的，这属于推理，不算泄漏。

## 用现有 264 张数据重新训练

```powershell
python scripts/train_bee_foreground_segmentation.py `
  --dataset-dir data\bee_foreground_v2\dataset `
  --output-dir outputs\bee_foreground_unetpp_resnet18_imagenet_retrain `
  --encoder resnet18 `
  --encoder-weights imagenet `
  --image-size 256 `
  --batch-size 4 `
  --epochs 80 `
  --disable-cudnn
```

Windows 本机训练时建议保留 `--disable-cudnn`，之前 cuDNN 在本机出现过不稳定。Linux 服务器上可以先不加，若报 cuDNN 错再加。

如果想复现旧版从头训练，把 `--encoder-weights imagenet` 改成：

```powershell
--encoder-weights none
```

对当前只有几百张 mask 的数据量，`resnet18 + imagenet` 比从头训练更合适。推理脚本不需要额外参数，训练好的 `best_model.pt` 已经保存完整权重。

训练输出目录会包含：

- `best_model.pt`
- `last_model.pt`
- `history.csv`
- `summary.json`
- `val_previews/`，如果 `--preview-limit` 大于 0

## 评估模型

在当前 53 张人工验证集上评估：

```powershell
python scripts/evaluate_bee_foreground_segmentation.py `
  --checkpoint models\bee_foreground_unetpp_resnet18_imagenet_v3\best_model.pt `
  --dataset-dir data\bee_foreground_v2\dataset `
  --split val `
  --output-dir outputs\eval_val_imagenet_v3 `
  --batch-size 64
```

输出：

- `summary.json`
- `per_image_metrics.csv`
- `overlays/`

overlay 颜色：

- 绿色：预测和人工标注都认为是 bee
- 红色：模型误选了背景
- 紫色：模型漏掉了 bee

多个模型可以用这个工具排序：

```powershell
python tools/compare_eval_summaries.py `
  outputs\eval_run_a\summary.json `
  outputs\eval_run_b\summary.json `
  --metric iou `
  --scope pixel_micro_metrics `
  --output outputs\model_comparison.csv
```

## 继续新增人工标注数据

推荐原则：

- 只从 ReID 的 `train_20260501` 里继续抽图标注 mask。
- 不要从 ReID 的 `val_20260501` 或 `eval` 里抽图来训练 mask 模型。
- 不要覆盖 `data/bee_foreground_v2`，新增数据时创建 `data/bee_foreground_v3`、`v4` 这样的新版本。
- 每次新增后更新 `annotation_registry.csv`，避免重复抽图。

### 1. 从原始 train 池抽新候选

```powershell
python tools/select_unlabeled_candidates.py `
  --source-root D:\personal_resources\CV_research\AutoDL\train_20260501 `
  --registry data\bee_foreground_v2\annotation_registry.csv `
  --sample-size 100 `
  --seed 20260519 `
  --output outputs\candidate_batches\fg_v3_candidates_100.csv
```

这个工具会按两层规则去重：

- `sample_id`
- 图片内容 SHA256

也就是说，即使文件名变了，只要图片内容完全相同，也会被排除。

### 2. 生成 CVAT 上传包

```powershell
python tools/build_cvat_image_package.py `
  --candidates-csv outputs\candidate_batches\fg_v3_candidates_100.csv `
  --output-dir outputs\cvat_batches\fg_v3_batch001
```

输出：

```text
outputs/cvat_batches/fg_v3_batch001/
  images/
  images.zip
  manifest.csv
```

在 CVAT 新建 task：

- label 只建一个：`bee`
- 上传 `images.zip`
- 标注格式：只画目标蜜蜂主体 mask
- 如果图里有其它背景蜜蜂、残缺蜜蜂，通常不要标

标完后从 CVAT 导出：

```text
Segmentation mask 1.1
```

### 3. 把 CVAT 导出转换成训练数据

假设 CVAT 导出文件是：

```text
C:\Users\dot\Downloads\job_xxx_annotations_segmentation mask 1.1.zip
```

运行：

```powershell
python scripts/prepare_bee_foreground_seg_dataset.py `
  --annotations-zip "C:\Users\dot\Downloads\job_xxx_annotations_segmentation mask 1.1.zip" `
  --subset-dir outputs\cvat_batches\fg_v3_batch001 `
  --output-dir outputs\manual_datasets\fg_v3_batch001 `
  --foreground-label bee `
  --val-fraction 0.20
```

输出会包含：

```text
outputs/manual_datasets/fg_v3_batch001/
  dataset/train/images
  dataset/train/masks
  dataset/val/images
  dataset/val/masks
  usable_annotations.csv
  train.csv
  val.csv
  skipped_annotations.csv
  summary.json
```

先看 `summary.json` 和 `skipped_annotations.csv`，确认没有大量漏转换。

### 4. 合并成新的数据集版本

不要直接改 `data/bee_foreground_v2`。创建新版本，例如 `v3`：

```powershell
python tools/merge_foreground_datasets.py `
  --base-root data\bee_foreground_v2 `
  --add-root outputs\manual_datasets\fg_v3_batch001 `
  --output-root data\bee_foreground_v3
```

然后为新版本生成 registry：

```powershell
python tools/build_annotation_registry.py `
  --dataset-dir data\bee_foreground_v3\dataset `
  --metadata-csv data\bee_foreground_v3\usable_annotations.csv `
  --output data\bee_foreground_v3\annotation_registry.csv
```

以后再继续加数据时，把 `--registry` 指向最新版本：

```powershell
--registry data\bee_foreground_v3\annotation_registry.csv
```

### 5. 用新版本重训

```powershell
python scripts/train_bee_foreground_segmentation.py `
  --dataset-dir data\bee_foreground_v3\dataset `
  --output-dir outputs\train_fg_v3_unetpp_resnet18_imagenet_256 `
  --encoder resnet18 `
  --encoder-weights imagenet `
  --image-size 256 `
  --batch-size 4 `
  --epochs 80 `
  --disable-cudnn
```

也可以从当前权重微调：

```powershell
python scripts/train_bee_foreground_segmentation.py `
  --dataset-dir data\bee_foreground_v3\dataset `
  --output-dir outputs\finetune_fg_v3_from_imagenet_v3 `
  --encoder resnet18 `
  --encoder-weights imagenet `
  --image-size 256 `
  --batch-size 4 `
  --epochs 40 `
  --disable-cudnn `
  --resume-checkpoint models\bee_foreground_unetpp_resnet18_imagenet_v3\best_model.pt
```

### 6. 选择是否替换发布权重

先用固定验证集评估新模型：

```powershell
python scripts/evaluate_bee_foreground_segmentation.py `
  --checkpoint outputs\train_fg_v3_unetpp_resnet18_imagenet_256\best_model.pt `
  --dataset-dir data\bee_foreground_v3\dataset `
  --split val `
  --output-dir outputs\eval_fg_v3
```

再和旧模型比较：

```powershell
python tools/compare_eval_summaries.py `
  outputs\eval_val_imagenet_v3\summary.json `
  outputs\eval_fg_v3\summary.json `
  --metric iou `
  --scope pixel_micro_metrics
```

如果 IoU 差距很小，优先看：

- `image_macro_metrics.iou`
- recall，是否漏蜜蜂身体、翅膀、腿
- overlays 里的复杂背景图
- 随机抽 100 张原始 ReID train/val/eval 的 overlay 肉眼检查

## 数据泄漏规则

可以：

- 用 `train_20260501` 抽图标 mask
- 用训练好的 mask 模型处理 ReID train/val/eval
- 用 ReID val 比较不同 mask 版本对 ReID 的影响

不可以：

- 用 ReID val/eval 的图片人工标 mask 后训练 mask 模型
- 看 ReID eval 结果后反复回头挑 mask 模型
- 覆盖旧数据集版本导致结果不可追溯

更完整的数据层级说明见：

[docs/dataset_workflow.md](docs/dataset_workflow.md)
