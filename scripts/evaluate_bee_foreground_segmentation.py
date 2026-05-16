"""Evaluate a bee foreground segmentation checkpoint on labeled masks."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch

import segmentation_models_pytorch as smp


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def imread(path: Path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder = checkpoint.get("encoder", "resnet18")
    image_size = int(checkpoint.get("image_size", 256))
    model = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    mean = np.array(checkpoint.get("imagenet_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.array(checkpoint.get("imagenet_std", [0.229, 0.224, 0.225]), dtype=np.float32)
    return model, image_size, mean, std, checkpoint


def preprocess(image_bgr: np.ndarray, image_size: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)


def list_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = image_path.relative_to(images_dir)
        mask_path = masks_dir / rel.with_suffix(".png")
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.size == 0:
        return mask_u8.astype(bool)
    padded = np.pad(mask_u8, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = (flood == 0)[1:-1, 1:-1]
    return np.logical_or(mask_u8 > 0, holes)


def keep_near_main_components(mask: np.ndarray, radius_frac: float = 0.14, min_area: int = 2) -> np.ndarray:
    m = mask.astype(np.uint8)
    if int(m.sum()) == 0:
        return m.astype(bool)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m, 8)
    if n_labels <= 1:
        return m.astype(bool)
    h, w = m.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    best_label = 1
    best_score = -1.0
    for label in range(1, n_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        x, y = centroids[label]
        dist = np.hypot(x - cx, y - cy) / max(1.0, min(h, w))
        score = area / (1.0 + 1.8 * dist)
        if score > best_score:
            best_label = label
            best_score = score
    main = labels == best_label
    dist_map = cv2.distanceTransform((~main).astype(np.uint8), cv2.DIST_L2, 3)
    max_dist = max(3, int(round(min(h, w) * radius_frac)))
    out = main.copy()
    for label in range(1, n_labels):
        if label == best_label:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == label
        if float(dist_map[comp].min()) <= max_dist:
            out |= comp
    return out


def clean_prediction(mask: np.ndarray, min_area: int, close_ratio: float) -> np.ndarray:
    if int(mask.sum()) < min_area:
        return mask.astype(bool)
    h, w = mask.shape[:2]
    close_size = max(3, int(round(min(h, w) * close_ratio)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    out = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0
    out = fill_mask_holes(out)
    out = keep_near_main_components(out, radius_frac=0.15, min_area=max(2, min_area // 10))
    return out


def counts_for(pred: np.ndarray, target: np.ndarray) -> dict[str, int]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    return {
        "tp": int((pred & target).sum()),
        "fp": int((pred & ~target).sum()),
        "fn": int((~pred & target).sum()),
        "tn": int((~pred & ~target).sum()),
    }


def metrics_from_counts(counts: dict[str, int | float]) -> dict[str, float]:
    tp = float(counts["tp"])
    fp = float(counts["fp"])
    fn = float(counts["fn"])
    tn = float(counts["tn"])
    eps = 1e-7
    return {
        "iou": tp / max(tp + fp + fn, eps),
        "dice": 2.0 * tp / max(2.0 * tp + fp + fn, eps),
        "precision": tp / max(tp + fp, eps),
        "recall": tp / max(tp + fn, eps),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, eps),
    }


def format_float(value: float) -> str:
    return f"{value:.6f}"


def make_error_overlay(image_bgr: np.ndarray, target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    true_positive = target & pred
    false_positive = ~target & pred
    false_negative = target & ~pred
    color = np.zeros_like(image_bgr)

    color[:, :] = (0, 180, 0)
    blended = cv2.addWeighted(image_bgr, 0.62, color, 0.38, 0.0)
    overlay[true_positive] = blended[true_positive]

    color[:, :] = (0, 0, 255)
    blended = cv2.addWeighted(image_bgr, 0.55, color, 0.45, 0.0)
    overlay[false_positive] = blended[false_positive]

    color[:, :] = (255, 0, 255)
    blended = cv2.addWeighted(image_bgr, 0.55, color, 0.45, 0.0)
    overlay[false_negative] = blended[false_negative]
    return overlay


def resolve_eval_pairs(args) -> tuple[str, list[tuple[Path, Path]]]:
    if args.images_dir and args.masks_dir:
        images_dir = Path(args.images_dir).resolve()
        masks_dir = Path(args.masks_dir).resolve()
        return "custom", list_pairs(images_dir, masks_dir)

    dataset_dir = Path(args.dataset_dir).resolve()
    split = args.split
    images_dir = dataset_dir / split / "images"
    masks_dir = dataset_dir / split / "masks"
    return split, list_pairs(images_dir, masks_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", default="data/bee_foreground_v2/dataset")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--images-dir")
    parser.add_argument("--masks-dir")
    parser.add_argument("--output-dir", default="outputs/eval_bee_foreground")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--close-ratio", type=float, default=0.012)
    parser.add_argument("--overlay-limit", type=int, default=80)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    overlay_dir = output_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_name, pairs = resolve_eval_pairs(args)
    if not pairs:
        raise RuntimeError("no image/mask pairs found for evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size, mean, std, checkpoint = load_model(Path(args.checkpoint), device)

    rows: list[dict[str, str | int]] = []
    total_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    macro_metrics: list[dict[str, float]] = []
    batch_tensors: list[np.ndarray] = []
    batch_meta: list[tuple[Path, Path, np.ndarray, np.ndarray]] = []

    def flush_batch() -> None:
        if not batch_tensors:
            return
        tensor = torch.from_numpy(np.stack(batch_tensors, axis=0)).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(tensor))[:, 0].detach().cpu().numpy()

        for prob_small, (image_path, mask_path, image_bgr, target) in zip(probs, batch_meta):
            confidence = cv2.resize(prob_small, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
            pred = confidence >= args.threshold
            if args.postprocess:
                pred = clean_prediction(pred, min_area=args.min_area, close_ratio=args.close_ratio)

            counts = counts_for(pred, target)
            for key in total_counts:
                total_counts[key] += counts[key]
            metrics = metrics_from_counts(counts)
            macro_metrics.append(metrics)

            rel = image_path.name
            overlay_path = ""
            if args.overlay_limit <= 0 or len(rows) < args.overlay_limit:
                overlay_path = str((overlay_dir / image_path.with_suffix(".jpg").name).resolve())
                imwrite(Path(overlay_path), make_error_overlay(image_bgr, target, pred))

            row: dict[str, str | int] = {
                "image": str(image_path),
                "mask": str(mask_path),
                "filename": rel,
                "overlay": overlay_path,
                "gt_pixels": int(target.sum()),
                "pred_pixels": int(pred.sum()),
                "tp": counts["tp"],
                "fp": counts["fp"],
                "fn": counts["fn"],
                "tn": counts["tn"],
            }
            row.update({key: format_float(value) for key, value in metrics.items()})
            rows.append(row)

        batch_tensors.clear()
        batch_meta.clear()

    for image_path, mask_path in pairs:
        image_bgr = imread(image_path, cv2.IMREAD_COLOR)
        mask = imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or mask is None:
            continue
        target = mask > 0
        batch_tensors.append(preprocess(image_bgr, image_size, mean, std))
        batch_meta.append((image_path, mask_path, image_bgr, target))
        if len(batch_tensors) >= args.batch_size:
            flush_batch()
    flush_batch()

    aggregate = metrics_from_counts(total_counts)
    macro = {
        key: float(np.mean([metrics[key] for metrics in macro_metrics])) if macro_metrics else 0.0
        for key in ["iou", "dice", "precision", "recall", "accuracy"]
    }

    csv_path = output_dir / "per_image_metrics.csv"
    fieldnames = [
        "filename",
        "image",
        "mask",
        "overlay",
        "gt_pixels",
        "pred_pixels",
        "tp",
        "fp",
        "fn",
        "tn",
        "iou",
        "dice",
        "precision",
        "recall",
        "accuracy",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "encoder": checkpoint.get("encoder", "resnet18"),
        "image_size": image_size,
        "classes": checkpoint.get("classes", ["background", "bee"]),
        "device": str(device),
        "eval_set": eval_name,
        "samples": len(rows),
        "threshold": args.threshold,
        "postprocess": args.postprocess,
        "pixel_micro_metrics": aggregate,
        "image_macro_metrics": macro,
        "counts": total_counts,
        "per_image_metrics_csv": str(csv_path),
        "overlay_dir": str(overlay_dir),
        "overlay_legend": {
            "green": "true positive bee",
            "red": "false positive background predicted as bee",
            "magenta": "false negative bee missed by prediction",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
