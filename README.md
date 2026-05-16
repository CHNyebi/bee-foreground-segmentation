# Bee Foreground Segmentation

这是当前蜜蜂前景分割模型的可复现发布包。它包含训练脚本、推理脚本、当前最好的权重、已标注的 264 张训练数据，以及一个去重登记表，方便以后继续从 `train_20260501` 里挑新图标注时避免重复。

## 当前版本

- 模型：`UnetPlusPlus`
- Encoder：`resnet18`
- 输入尺寸：`256x256`
- 任务：二分类前景分割，`bee` vs background
- 训练集：211 张
- 验证集：53 张
- 当前权重：`models/bee_foreground_unetpp_resnet18_v2/best_model.pt`
- 最佳验证指标：IoU `0.7765`，Dice `0.8742`，Precision `0.8738`，Recall `0.8746`

## 目录

```text
.
├── configs/
│   └── bee_foreground_v2.json
├── data/
│   └── bee_foreground_v2/
│       ├── annotation_registry.csv
│       ├── dataset/
│       │   ├── train/images
│       │   ├── train/masks
│       │   ├── val/images
│       │   └── val/masks
│       ├── train.csv
│       ├── usable_annotations.csv
│       └── val.csv
├── models/
│   └── bee_foreground_unetpp_resnet18_v2/
│       ├── best_model.pt
│       ├── history.csv
│       ├── summary.json
│       └── val_preview_contact_sheet.jpg
├── reports/
│   └── random100_review_v2/
├── scripts/
└── tools/
```

## 环境

建议新建 Python 环境后安装：

```powershell
python -m pip install -r requirements.txt
```

如果在 CUDA 服务器上复现，优先按服务器 CUDA 版本安装匹配的 PyTorch，再安装其它依赖。例如本机训练环境是 `torch 1.11.0+cu113` / `torchvision 0.12.0+cu113`。

## 直接使用当前权重

对一个 ImageFolder 风格的蜜蜂裁剪目录做推理：

```powershell
python scripts/predict_bee_foreground_segmentation.py `
  --input-dir D:\personal_resources\CV_research\AutoDL\train_20260501 `
  --checkpoint models\bee_foreground_unetpp_resnet18_v2\best_model.pt `
  --output-dir outputs\train_20260501_masked_v2 `
  --batch-size 64
```

输出会包含：

- `labels/`：二值 mask
- `masked/`：扣背景后的图
- `overlays/`：抽查叠加图
- `prediction_report.csv`：每张图的面积、置信度等统计
- `summary.json`：本次推理摘要

## 复现训练

```powershell
python scripts/train_bee_foreground_segmentation.py `
  --dataset-dir data\bee_foreground_v2\dataset `
  --output-dir outputs\bee_foreground_unetpp_resnet18_v2_retrain `
  --encoder resnet18 `
  --image-size 256 `
  --batch-size 4 `
  --epochs 80 `
  --disable-cudnn
```

`--disable-cudnn` 是因为这次 Windows 本机训练时遇到过 cuDNN 相关报错；在 Linux/CUDA 服务器上可以先不加，若报 cuDNN 错再加。

## 以后继续加数据，怎么避免重复

`data/bee_foreground_v2/annotation_registry.csv` 是当前 264 张已标注样本的登记表，包含：

- `sample_id`
- 原始相对路径
- 原始图片路径
- 图片 SHA256
- mask SHA256

以后从原始裁剪池 `train_20260501` 里继续抽图时，先运行：

```powershell
python tools/select_unlabeled_candidates.py `
  --source-root D:\personal_resources\CV_research\AutoDL\train_20260501 `
  --sample-size 100 `
  --seed 20260516 `
  --output data\candidates_next100.csv
```

这个工具会同时按 `sample_id` 和图片 SHA256 去重。即使文件名变了，只要图片内容完全相同，也会被排除。

如果以后合并了新标注数据，重新生成登记表：

```powershell
python tools/build_annotation_registry.py `
  --dataset-dir data\bee_foreground_v2\dataset `
  --metadata-csv data\bee_foreground_v2\usable_annotations.csv `
  --output data\bee_foreground_v2\annotation_registry.csv
```

## 标注约定

CVAT 里只需要一个标签：

```text
bee
```

只标目标蜜蜂主体；如果图里有其它不完整或背景里的蜜蜂，通常不用标。mask 文件中非零像素都被视为 `bee`，0 是背景。

## 数据说明

本仓库提交的是已经人工修正并用于训练的 264 张标注子集，不提交完整原始裁剪池 `train_20260501`。这样仓库能直接复现实验，又不会把全部原始图片都塞进 GitHub。
