"""Mine hard bee foreground annotation candidates from an unlabeled crop pool.

The script runs a trained foreground model over an ImageFolder-style root,
skips samples already present in an annotation registry, scores likely hard
cases, and builds a CVAT image package plus bee-only CVAT XML prelabels for
the selected samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("_")


def sample_id_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return f"{rel.parent.name}__{rel.stem}"


def load_registry(path: Path) -> tuple[set[str], set[str], list[float]]:
    sample_ids: set[str] = set()
    hashes: set[str] = set()
    ratios: list[float] = []
    if not path.exists():
        return sample_ids, hashes, ratios
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("sample_id"):
                sample_ids.add(row["sample_id"])
            if row.get("image_sha256"):
                hashes.add(row["image_sha256"])
            if row.get("foreground_ratio"):
                try:
                    ratios.append(float(row["foreground_ratio"]))
                except ValueError:
                    pass
    return sample_ids, hashes, ratios


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
    mean = np.array(checkpoint.get("imagenet_mean", IMAGENET_MEAN), dtype=np.float32)
    std = np.array(checkpoint.get("imagenet_std", IMAGENET_STD), dtype=np.float32)
    preprocess_mode = checkpoint.get("preprocess", "resize")
    return model, image_size, mean, std, preprocess_mode, checkpoint


def letterbox_image(image: np.ndarray, image_size: int) -> tuple[np.ndarray, dict[str, int]]:
    h, w = image.shape[:2]
    scale = min(image_size / max(w, 1), image_size / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    left = (image_size - new_w) // 2
    top = (image_size - new_h) // 2
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    boxed = np.zeros((image_size, image_size, image.shape[2]), dtype=image.dtype)
    boxed[top : top + new_h, left : left + new_w] = resized
    return boxed, {"top": top, "left": left, "new_h": new_h, "new_w": new_w}


def preprocess(image_bgr: np.ndarray, image_size: int, mean: np.ndarray, std: np.ndarray, preprocess_mode: str):
    rgb = cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
    meta = None
    if preprocess_mode == "letterbox":
        prepared, meta = letterbox_image(rgb, image_size)
    else:
        prepared = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    arr = prepared.astype(np.float32) / 255.0
    arr = (arr - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    return np.transpose(arr, (2, 0, 1)).astype(np.float32), meta


def restore_confidence(confidence: np.ndarray, original_shape: tuple[int, int], preprocess_mode: str, meta: dict[str, int] | None):
    h, w = original_shape
    if preprocess_mode != "letterbox" or meta is None:
        return cv2.resize(confidence, (w, h), interpolation=cv2.INTER_LINEAR)
    top = meta["top"]
    left = meta["left"]
    new_h = meta["new_h"]
    new_w = meta["new_w"]
    cropped = confidence[top : top + new_h, left : left + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


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


def quality_stats(image_bgr: np.ndarray) -> dict[str, float | int | str]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    scale = 128.0 / max(1, max(h, w))
    if scale < 1.0:
        small = cv2.resize(gray, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    else:
        small = gray
    lap_var = float(cv2.Laplacian(small, cv2.CV_64F).var())
    contrast = float(small.std())
    brightness = float(small.mean())
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    min_side = min(h, w)

    sharpness_score = min(1.0, math.log1p(lap_var) / math.log1p(550.0))
    edge_score = min(1.0, math.log1p(tenengrad) / math.log1p(9000.0))
    contrast_score = min(1.0, contrast / 42.0)
    size_score = min(1.0, max(0.0, (min_side - 24.0) / 56.0))
    exposure_score = 1.0 - min(1.0, abs(brightness - 128.0) / 128.0)
    quality_score = (
        0.34 * sharpness_score
        + 0.22 * edge_score
        + 0.20 * contrast_score
        + 0.14 * size_score
        + 0.10 * exposure_score
    )
    if quality_score >= 0.58 and sharpness_score >= 0.45:
        quality_bucket = "clear"
    elif quality_score >= 0.38:
        quality_bucket = "slightly_blurry_but_annotatable"
    else:
        quality_bucket = "too_blurry_or_low_quality"

    return {
        "image_width": w,
        "image_height": h,
        "min_side": min_side,
        "laplacian_var": lap_var,
        "tenengrad": tenengrad,
        "contrast": contrast,
        "brightness": brightness,
        "quality_score": quality_score,
        "sharpness_score": sharpness_score,
        "quality_bucket": quality_bucket,
    }


def mask_stats(mask: np.ndarray, confidence: np.ndarray, area_q10: float, area_q90: float) -> dict[str, float | int]:
    h, w = mask.shape[:2]
    total = max(1, h * w)
    fg = int(mask.sum())
    area_ratio = fg / total
    uncertainty = ((confidence >= 0.35) & (confidence <= 0.65))
    uncertainty_ratio = float(uncertainty.mean())
    entropy = -(confidence * np.log2(np.clip(confidence, 1e-6, 1.0)) + (1.0 - confidence) * np.log2(np.clip(1.0 - confidence, 1e-6, 1.0)))
    entropy_mean = float(entropy.mean())

    if fg:
        ys, xs = np.where(mask)
        bbox_area = int((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        bbox_ratio = bbox_area / total
        border_pixels = int(mask[0, :].sum() + mask[-1, :].sum() + mask[:, 0].sum() + mask[:, -1].sum())
        border_touch_ratio = border_pixels / max(1, 2 * h + 2 * w)
    else:
        bbox_ratio = 0.0
        border_touch_ratio = 0.0

    n_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    component_count = max(0, n_labels - 1)
    component_areas = [int(cc_stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)]
    largest_component_ratio = (max(component_areas) / max(1, fg)) if component_areas else 0.0
    small_component_count = sum(1 for area in component_areas if area < max(3, int(total * 0.01)))

    kernel_size = max(3, int(round(min(h, w) * 0.08)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    outer_ring = dilated & ~mask
    inner_ring = mask & ~eroded
    outer_uncertain_ratio = float(((confidence >= 0.25) & (confidence < 0.50) & outer_ring).sum() / max(1, outer_ring.sum()))
    inner_low_conf_ratio = float(((confidence >= 0.35) & (confidence < 0.70) & inner_ring).sum() / max(1, inner_ring.sum()))

    area_low_score = max(0.0, (area_q10 - area_ratio) / max(area_q10, 1e-6))
    area_high_score = max(0.0, (area_ratio - area_q90) / max(1.0 - area_q90, 1e-6))
    area_anomaly_score = min(1.0, max(area_low_score, area_high_score))
    component_score = min(1.0, math.log1p(max(0, component_count - 1)) / math.log(8.0) + max(0.0, 1.0 - largest_component_ratio) * 0.6)
    uncertainty_score = min(1.0, uncertainty_ratio / 0.18)
    boundary_score = min(1.0, (outer_uncertain_ratio * 0.6 + inner_low_conf_ratio * 0.4) / 0.35)
    border_score = min(1.0, border_touch_ratio / 0.20)
    bbox_score = min(1.0, max(0.0, bbox_ratio - 0.72) / 0.28)
    hard_score = (
        0.26 * uncertainty_score
        + 0.22 * area_anomaly_score
        + 0.18 * component_score
        + 0.18 * boundary_score
        + 0.10 * max(border_score, bbox_score)
        + 0.06 * min(1.0, entropy_mean / 0.75)
    )

    fp_like_score = min(1.0, 0.35 * area_high_score + 0.25 * bbox_score + 0.20 * border_score + 0.20 * component_score)
    fn_like_score = min(1.0, 0.45 * outer_uncertain_ratio + 0.35 * inner_low_conf_ratio + 0.20 * uncertainty_score)
    area_abnormal_score = min(1.0, max(area_anomaly_score, max(0.0, bbox_ratio - 0.82) / 0.18))

    return {
        "foreground_pixels": fg,
        "foreground_ratio": area_ratio,
        "bbox_ratio": bbox_ratio,
        "border_touch_ratio": border_touch_ratio,
        "component_count": component_count,
        "small_component_count": small_component_count,
        "largest_component_ratio": largest_component_ratio,
        "uncertainty_ratio": uncertainty_ratio,
        "entropy_mean": entropy_mean,
        "outer_uncertain_ratio": outer_uncertain_ratio,
        "inner_low_conf_ratio": inner_low_conf_ratio,
        "area_anomaly_score": area_anomaly_score,
        "component_score": component_score,
        "uncertainty_score": uncertainty_score,
        "boundary_score": boundary_score,
        "fp_like_score": fp_like_score,
        "fn_like_score": fn_like_score,
        "area_abnormal_score": area_abnormal_score,
        "hard_score": hard_score,
    }


def reasons_for(row: dict[str, str]) -> str:
    reasons = []
    if float(row["fp_like_score"]) >= 0.35:
        reasons.append("fp_like_background_or_other_bee")
    if float(row["fn_like_score"]) >= 0.45:
        reasons.append("fn_like_edges_legs_wings")
    if float(row["area_abnormal_score"]) >= 0.35:
        reasons.append("mask_area_abnormal")
    if float(row["component_score"]) >= 0.30:
        reasons.append("fragmented_mask")
    if float(row["uncertainty_score"]) >= 0.45:
        reasons.append("low_confidence")
    if float(row["foreground_ratio"]) < 0.12:
        reasons.append("very_small_mask")
    if float(row["foreground_ratio"]) > 0.82:
        reasons.append("very_large_mask")
    return ";".join(reasons[:4]) or "diverse_hard_sample"


def video_key_for(sample_id: str) -> str:
    match = re.search(r"(BEE\d+-\d+_\d+).*?(tr\d+)", sample_id)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    parts = sample_id.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else sample_id


def select_diverse(rows: list[dict[str, str]], limit: int, max_per_parent: int, max_per_video_key: int, min_quality_score: float) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    parent_counts: dict[str, int] = defaultdict(int)
    video_counts: dict[str, int] = defaultdict(int)
    rows = [row for row in rows if float(row.get("quality_score", "0")) >= min_quality_score and row.get("quality_bucket") != "too_blurry_or_low_quality"]
    if len(rows) < limit:
        rows = sorted(rows, key=lambda r: (float(r.get("quality_score", "0")), float(r["hard_score"])), reverse=True)

    buckets = [
        ("fp_like_score", int(round(limit * 0.25))),
        ("fn_like_score", int(round(limit * 0.25))),
        ("area_abnormal_score", int(round(limit * 0.20))),
        ("uncertainty_score", int(round(limit * 0.20))),
        ("hard_score", limit),
    ]
    for key, quota in buckets:
        pool = sorted(rows, key=lambda r: (float(r[key]), float(r["hard_score"])), reverse=True)
        added_for_bucket = 0
        for row in pool:
            if len(selected) >= limit:
                break
            sid = row["sample_id"]
            if sid in seen:
                continue
            parent = row["source_parent"]
            video_key = row["video_key"]
            if parent_counts[parent] >= max_per_parent or video_counts[video_key] >= max_per_video_key:
                continue
            selected.append(row)
            seen.add(sid)
            parent_counts[parent] += 1
            video_counts[video_key] += 1
            added_for_bucket += 1
            if added_for_bucket >= quota:
                break

    if len(selected) < limit:
        for row in sorted(rows, key=lambda r: float(r["hard_score"]), reverse=True):
            if len(selected) >= limit:
                break
            if row["sample_id"] in seen:
                continue
            selected.append(row)
            seen.add(row["sample_id"])
    return selected[:limit]


def make_overlay(image_bgr: np.ndarray, mask: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    color = np.zeros_like(image_bgr)
    color[:, :] = (255, 96, 0)
    blended = cv2.addWeighted(image_bgr, 0.62, color, 0.38, 0.0)
    overlay[mask] = blended[mask]
    uncertain = (confidence >= 0.35) & (confidence <= 0.65)
    edge_color = np.zeros_like(image_bgr)
    edge_color[:, :] = (255, 0, 255)
    overlay[uncertain] = cv2.addWeighted(overlay, 0.65, edge_color, 0.35, 0.0)[uncertain]
    return overlay


def rle_counts(mask: np.ndarray) -> list[int]:
    flat = mask.astype(np.uint8).reshape(-1)
    counts: list[int] = []
    last = 0
    run = 0
    for value in flat:
        v = int(value)
        if v == last:
            run += 1
        else:
            counts.append(run)
            run = 1
            last = v
    counts.append(run)
    while counts and counts[-1] == 0:
        counts.pop()
    return counts


def add_mask_xml(image_el: ET.Element, mask: np.ndarray, label: str) -> None:
    if int(mask.sum()) == 0:
        return
    ys, xs = np.where(mask)
    top = int(ys.min())
    left = int(xs.min())
    bottom = int(ys.max()) + 1
    right = int(xs.max()) + 1
    crop = mask[top:bottom, left:right]
    ET.SubElement(
        image_el,
        "mask",
        {
            "label": label,
            "source": "auto",
            "occluded": "0",
            "rle": ", ".join(str(v) for v in rle_counts(crop)),
            "left": str(left),
            "top": str(top),
            "width": str(right - left),
            "height": str(bottom - top),
            "z_order": "0",
        },
    )


def indent_xml(element: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = indent + "  "
        for child in element:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indent


def build_cvat_xml(rows: list[dict[str, str]], label: str, output_path: Path) -> None:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "0"
    ET.SubElement(task, "name").text = "bee_foreground_hard100"
    ET.SubElement(task, "size").text = str(len(rows))
    ET.SubElement(task, "mode").text = "annotation"
    ET.SubElement(task, "overlap").text = "0"
    ET.SubElement(task, "bugtracker").text = ""
    ET.SubElement(task, "flipped").text = "False"
    labels_el = ET.SubElement(task, "labels")
    label_el = ET.SubElement(labels_el, "label")
    ET.SubElement(label_el, "name").text = label
    ET.SubElement(label_el, "color").text = "#00c853"
    ET.SubElement(label_el, "type").text = "any"
    ET.SubElement(label_el, "attributes")

    for idx, row in enumerate(rows):
        image_path = Path(row["package_image_path"])
        mask_path = Path(row["label_path"])
        image = imread(image_path, cv2.IMREAD_COLOR)
        mask = imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        image_el = ET.SubElement(
            root,
            "image",
            {
                "id": str(idx),
                "name": image_path.name,
                "width": str(image.shape[1]),
                "height": str(image.shape[0]),
            },
        )
        add_mask_xml(image_el, mask > 0, label)

    tree = ET.ElementTree(root)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    else:
        indent_xml(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def make_contact_sheet(rows: list[dict[str, str]], output_path: Path, cols: int = 5, thumb_w: int = 220, thumb_h: int = 170) -> None:
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("arial.ttf", 12)
        font_b = ImageFont.truetype("arialbd.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
        font_b = font

    cell_h = thumb_h + 72
    rows_n = math.ceil(len(rows) / cols)
    canvas = Image.new("RGB", (cols * thumb_w, rows_n * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, row in enumerate(rows):
        overlay = imread(Path(row["overlay_path"]), cv2.IMREAD_COLOR)
        if overlay is None:
            continue
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        scale = min(thumb_w / img.width, thumb_h / img.height)
        new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        x = (idx % cols) * thumb_w
        y = (idx // cols) * cell_h
        canvas.paste(img, (x + (thumb_w - img.width) // 2, y))
        draw.text((x + 4, y + thumb_h + 3), f"{idx + 1:03d} score {float(row['hard_score']):.3f}", font=font_b, fill=(20, 20, 20))
        draw.text((x + 4, y + thumb_h + 20), row["sample_id"][:34], font=font, fill=(45, 45, 45))
        draw.text((x + 4, y + thumb_h + 36), row["reasons"][:42], font=font, fill=(90, 40, 40))
        draw.text((x + 4, y + thumb_h + 52), f"area {float(row['foreground_ratio']):.2f} unc {float(row['uncertainty_ratio']):.2f}", font=font, fill=(75, 75, 75))
    canvas.save(output_path, quality=92)


def format_row(row: dict[str, str | int | float]) -> dict[str, str]:
    out = {}
    for key, value in row.items():
        if isinstance(value, float):
            out[key] = f"{value:.6f}"
        else:
            out[key] = str(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--registry", default="data/bee_foreground_v2/annotation_registry.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--close-ratio", type=float, default=0.012)
    parser.add_argument("--max-per-parent", type=int, default=2)
    parser.add_argument("--max-per-video-key", type=int, default=1)
    parser.add_argument("--min-quality-score", type=float, default=0.38)
    parser.add_argument("--label", default="bee")
    parser.add_argument("--log-interval", type=int, default=1000)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    registry_path = Path(args.registry).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    used_ids, used_hashes, labeled_ratios = load_registry(registry_path)
    if labeled_ratios:
        area_q10 = float(np.quantile(labeled_ratios, 0.10))
        area_q90 = float(np.quantile(labeled_ratios, 0.90))
    else:
        area_q10, area_q90 = 0.18, 0.78

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size, mean, std, preprocess_mode, checkpoint = load_model(checkpoint_path, device)

    all_paths = list(iter_images(source_root))
    rows: list[dict[str, str]] = []
    skipped_id = 0
    skipped_hash = 0
    failed = 0
    batch_tensors: list[np.ndarray] = []
    batch_meta: list[tuple[Path, str, str, str, np.ndarray, dict[str, int] | None]] = []
    start = time.time()

    def flush_batch() -> None:
        nonlocal failed
        if not batch_tensors:
            return
        tensor = torch.from_numpy(np.stack(batch_tensors, axis=0)).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(tensor))[:, 0].detach().cpu().numpy()

        for prob_small, (path, sid, digest, rel, image_bgr, meta) in zip(probs, batch_meta):
            confidence = restore_confidence(prob_small, image_bgr.shape[:2], preprocess_mode, meta)
            mask = clean_prediction(confidence >= args.threshold, min_area=args.min_area, close_ratio=args.close_ratio)
            try:
                q_stats = quality_stats(image_bgr)
                stats = mask_stats(mask, confidence, area_q10, area_q90)
            except Exception:
                failed += 1
                continue
            row = {
                "sample_id": sid,
                "relative_path": rel,
                "source_image_path": str(path),
                "source_parent": str(Path(rel).parent),
                "video_key": video_key_for(sid),
                "image_sha256": digest,
                **q_stats,
                **stats,
            }
            row = format_row(row)
            row["reasons"] = reasons_for(row)
            rows.append(row)

        batch_tensors.clear()
        batch_meta.clear()

    processed = 0
    for path in all_paths:
        sid = sample_id_for(path, source_root)
        if sid in used_ids:
            skipped_id += 1
            continue
        digest = sha256_file(path)
        if digest in used_hashes:
            skipped_hash += 1
            continue
        image_bgr = imread(path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            failed += 1
            continue
        tensor, meta = preprocess(image_bgr, image_size, mean, std, preprocess_mode)
        batch_tensors.append(tensor)
        batch_meta.append((path, sid, digest, str(path.relative_to(source_root)), image_bgr, meta))
        processed += 1
        if len(batch_tensors) >= args.batch_size:
            flush_batch()
        if args.log_interval and processed % args.log_interval == 0:
            elapsed = time.time() - start
            print(f"processed={processed} scored={len(rows)} skipped_id={skipped_id} skipped_hash={skipped_hash} elapsed={elapsed:.1f}s", flush=True)
    flush_batch()

    rows.sort(key=lambda r: float(r["hard_score"]), reverse=True)
    scored_path = output_dir / "candidates_scored.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with scored_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    selected = select_diverse(rows, args.limit, args.max_per_parent, args.max_per_video_key, args.min_quality_score)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "prelabel_masks"
    overlays_dir = output_dir / "overlays"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    overlays_dir.mkdir(exist_ok=True)

    used_names: set[str] = set()
    final_rows: list[dict[str, str]] = []
    for idx, row in enumerate(selected, start=1):
        source_path = Path(row["source_image_path"])
        flat_stem = safe_name(row["sample_id"]) or f"sample_{idx:06d}"
        flat_name = f"{idx:06d}__{flat_stem}{source_path.suffix.lower()}"
        if flat_name in used_names:
            flat_name = f"{idx:06d}_{flat_name}"
        used_names.add(flat_name)

        package_image_path = images_dir / flat_name
        shutil.copy2(source_path, package_image_path)

        image_bgr = imread(source_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        tensor, meta = preprocess(image_bgr, image_size, mean, std, preprocess_mode)
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.from_numpy(tensor[None]).to(device)))[0, 0].detach().cpu().numpy()
        confidence = restore_confidence(prob, image_bgr.shape[:2], preprocess_mode, meta)
        mask = clean_prediction(confidence >= args.threshold, min_area=args.min_area, close_ratio=args.close_ratio)
        label_path = labels_dir / Path(flat_name).with_suffix(".png").name
        overlay_path = overlays_dir / Path(flat_name).with_suffix(".jpg").name
        imwrite(label_path, mask.astype(np.uint8))
        imwrite(overlay_path, make_overlay(image_bgr, mask, confidence))

        out_row = dict(row)
        out_row.update(
            {
                "index": str(idx),
                "flat_filename": flat_name,
                "package_image_path": str(package_image_path),
                "label_path": str(label_path),
                "overlay_path": str(overlay_path),
            }
        )
        final_rows.append(out_row)

    selected_path = output_dir / "hard_candidates_100.csv"
    selected_fields = [
        "index",
        "sample_id",
        "flat_filename",
        "relative_path",
        "source_image_path",
        "package_image_path",
        "label_path",
        "overlay_path",
        "image_sha256",
        "hard_score",
        "reasons",
        "fp_like_score",
        "fn_like_score",
        "area_abnormal_score",
        "uncertainty_score",
        "component_score",
        "foreground_ratio",
        "bbox_ratio",
        "border_touch_ratio",
        "component_count",
        "small_component_count",
        "uncertainty_ratio",
        "quality_score",
        "quality_bucket",
        "laplacian_var",
        "contrast",
        "brightness",
        "min_side",
        "outer_uncertain_ratio",
        "inner_low_conf_ratio",
        "source_parent",
        "video_key",
    ]
    with selected_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=selected_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        manifest_fields = ["index", "sample_id", "flat_filename", "relative_path", "source_image_path", "image_sha256", "reasons"]
        writer = csv.DictWriter(f, fieldnames=manifest_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)

    images_zip = output_dir / "images.zip"
    with zipfile.ZipFile(images_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for image_path in sorted(images_dir.glob("*")):
            zf.write(image_path, arcname=image_path.name)

    annotations_xml = output_dir / "annotations.xml"
    build_cvat_xml(final_rows, args.label, annotations_xml)
    prelabels_zip = output_dir / "prelabels_cvat_for_images_1_1_bee_only.zip"
    with zipfile.ZipFile(prelabels_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(annotations_xml, arcname="annotations.xml")

    contact_sheet = output_dir / "contact_sheet.jpg"
    make_contact_sheet(final_rows, contact_sheet)

    summary = {
        "source_root": str(source_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_encoder": checkpoint.get("encoder", "resnet18"),
        "checkpoint_encoder_weights": checkpoint.get("encoder_weights"),
        "registry": str(registry_path),
        "total_images": len(all_paths),
        "scored_candidates": len(rows),
        "selected": len(final_rows),
        "skipped_sample_id": skipped_id,
        "skipped_sha256": skipped_hash,
        "failed": failed,
        "area_q10_from_labeled": area_q10,
        "area_q90_from_labeled": area_q90,
        "min_quality_score": args.min_quality_score,
        "output_dir": str(output_dir),
        "images_zip": str(images_zip),
        "prelabels_zip": str(prelabels_zip),
        "selected_csv": str(selected_path),
        "scored_csv": str(scored_path),
        "contact_sheet": str(contact_sheet),
        "elapsed_seconds": round(time.time() - start, 2),
        "notes": "Scores are model-derived hard-mining proxies because unlabeled ReID train images do not have GT masks yet.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
