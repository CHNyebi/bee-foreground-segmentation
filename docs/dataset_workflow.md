# Dataset Workflow

这份文档说明 `data/bee_foreground_v2` 怎么用于训练，以及后续怎么在它的基础上继续添加人工 mask 标注。

## 1. 现有数据是什么

当前仓库里的训练数据是：

```text
data/bee_foreground_v2
```

它包含 264 张人工修正过的 bee foreground mask：

```text
train: 211
val:    53
total: 264
```

来源：

```text
v1_190: 190 张
hard75: 74 张可用
```

这些图都来自 ReID 原始训练池 `train_20260501` 的抽样子集。它们不是完整 ReID 训练集。

## 2. 为什么要有 annotation_registry.csv

`data/bee_foreground_v2/annotation_registry.csv` 是当前已标注样本的登记表。后续继续从 `train_20260501` 抽图时，必须用它去重。

关键字段：

```text
sample_id
filename
split
relative_path
source_image_path
image_sha256
mask_sha256
source_dataset
foreground_pixels
foreground_ratio
```

去重时同时使用：

- `sample_id`
- `image_sha256`

这样可以防止同一张图片因为路径或文件名变化而被重复抽中。

## 3. 重新训练当前数据集

```powershell
python scripts/train_bee_foreground_segmentation.py `
  --dataset-dir data\bee_foreground_v2\dataset `
  --output-dir outputs\retrain_fg_v2 `
  --encoder resnet18 `
  --image-size 256 `
  --batch-size 4 `
  --epochs 80 `
  --disable-cudnn
```

训练结束后用验证集评估：

```powershell
python scripts/evaluate_bee_foreground_segmentation.py `
  --checkpoint outputs\retrain_fg_v2\best_model.pt `
  --dataset-dir data\bee_foreground_v2\dataset `
  --split val `
  --output-dir outputs\eval_retrain_fg_v2
```

## 4. 新增数据的标准流程

### Step 1: 抽候选图

只从 ReID train 池抽：

```powershell
python tools/select_unlabeled_candidates.py `
  --source-root D:\personal_resources\CV_research\AutoDL\train_20260501 `
  --registry data\bee_foreground_v2\annotation_registry.csv `
  --sample-size 100 `
  --seed 20260519 `
  --output outputs\candidate_batches\fg_v3_candidates_100.csv
```

不要从 ReID val/eval 里抽图训练 mask 模型。

### Step 2: 创建 CVAT 图包

```powershell
python tools/build_cvat_image_package.py `
  --candidates-csv outputs\candidate_batches\fg_v3_candidates_100.csv `
  --output-dir outputs\cvat_batches\fg_v3_batch001
```

上传到 CVAT 的文件是：

```text
outputs/cvat_batches/fg_v3_batch001/images.zip
```

CVAT task 只建一个标签：

```text
bee
```

### Step 3: CVAT 标注和导出

标注规则：

- 只标目标蜜蜂主体
- 背景不用标
- 其它残缺蜜蜂、远处背景蜜蜂通常不用标
- mask 尽量包括身体、头、翅膀、腿，但不要把大块背景涂进去

导出格式：

```text
Segmentation mask 1.1
```

### Step 4: 转成训练数据

```powershell
python scripts/prepare_bee_foreground_seg_dataset.py `
  --annotations-zip "C:\Users\dot\Downloads\job_xxx_annotations_segmentation mask 1.1.zip" `
  --subset-dir outputs\cvat_batches\fg_v3_batch001 `
  --output-dir outputs\manual_datasets\fg_v3_batch001 `
  --foreground-label bee `
  --val-fraction 0.20
```

检查：

```text
outputs/manual_datasets/fg_v3_batch001/summary.json
outputs/manual_datasets/fg_v3_batch001/skipped_annotations.csv
```

如果 `skipped_annotations.csv` 很多，先不要合并，应该先查 CVAT 导出或 label 是否有问题。

### Step 5: 合并成新版本

```powershell
python tools/merge_foreground_datasets.py `
  --base-root data\bee_foreground_v2 `
  --add-root outputs\manual_datasets\fg_v3_batch001 `
  --output-root data\bee_foreground_v3
```

生成新 registry：

```powershell
python tools/build_annotation_registry.py `
  --dataset-dir data\bee_foreground_v3\dataset `
  --metadata-csv data\bee_foreground_v3\usable_annotations.csv `
  --output data\bee_foreground_v3\annotation_registry.csv
```

后续继续抽图时使用新 registry：

```powershell
--registry data\bee_foreground_v3\annotation_registry.csv
```

### Step 6: 用新版本训练

```powershell
python scripts/train_bee_foreground_segmentation.py `
  --dataset-dir data\bee_foreground_v3\dataset `
  --output-dir outputs\train_fg_v3 `
  --encoder resnet18 `
  --image-size 256 `
  --batch-size 4 `
  --epochs 80 `
  --disable-cudnn
```

## 5. 版本命名建议

不要覆盖旧目录。推荐：

```text
data/bee_foreground_v2
data/bee_foreground_v3
data/bee_foreground_v4

outputs/train_fg_v3_unetpp_resnet18_256
outputs/eval_fg_v3_unetpp_resnet18_256
```

如果某个版本要发布成默认权重，再复制到：

```text
models/bee_foreground_unetpp_resnet18_v3/
```

并在 README 里记录：

- 数据集版本
- 样本数量
- 模型结构
- checkpoint 路径
- 验证指标
- 推荐阈值

## 6. 和 ReID split 的关系

ReID 主任务 split 是最高层：

```text
train_20260501
val_20260501
eval
```

mask 模型的人工标注只从 `train_20260501` 抽。训练好的 mask 模型可以推理到三类 ReID 图片：

```text
train_20260501 -> masked_train
val_20260501   -> masked_val
eval           -> masked_eval
```

最终 ReID 训练仍然是：

```text
masked_train 训练
masked_val   选模型
masked_eval  最终测试
```

不要把 ReID val/eval 的人工 mask 加进 mask 模型训练集。
