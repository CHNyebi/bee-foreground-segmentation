"""Merge an existing bee foreground dataset with a newly prepared dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["filename"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pair_paths(root: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = root / "dataset" / split / "images"
    mask_dir = root / "dataset" / split / "masks"
    pairs: list[tuple[Path, Path]] = []
    if not image_dir.exists():
        return pairs
    for image_path in sorted(image_dir.glob("*")):
        mask_path = mask_dir / image_path.with_suffix(".png").name
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def copy_dataset(source_root: Path, output_root: Path, existing_hashes: set[str]) -> tuple[dict[str, int], set[str]]:
    stats = {"copied": 0, "skipped_duplicate": 0}
    copied_filenames: set[str] = set()
    for split in ("train", "val"):
        for image_path, mask_path in pair_paths(source_root, split):
            digest = sha256_file(image_path)
            if digest in existing_hashes:
                stats["skipped_duplicate"] += 1
                continue
            existing_hashes.add(digest)
            copied_filenames.add(image_path.name)

            out_image = output_root / "dataset" / split / "images" / image_path.name
            out_mask = output_root / "dataset" / split / "masks" / mask_path.name
            out_image.parent.mkdir(parents=True, exist_ok=True)
            out_mask.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, out_image)
            shutil.copy2(mask_path, out_mask)
            stats["copied"] += 1
    return stats, copied_filenames


def rows_for_copied(root: Path, copied_names: set[str]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    usable = []
    train = []
    val = []
    for name, target in (
        ("usable_annotations.csv", usable),
        ("train.csv", train),
        ("val.csv", val),
    ):
        for row in read_csv(root / name):
            filename = row.get("combined_filename") or row.get("flat_filename") or row.get("filename")
            if not copied_names or filename in copied_names:
                target.append(row)
    return usable, train, val


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", required=True, help="Existing dataset root, e.g. data/bee_foreground_v2")
    parser.add_argument("--add-root", required=True, help="New prepared dataset root from prepare_bee_foreground_seg_dataset.py")
    parser.add_argument("--output-root", required=True, help="Merged output dataset root, create a new version here")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_root = Path(args.base_root).resolve()
    add_root = Path(args.add_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    existing_hashes: set[str] = set()
    base_stats, base_names = copy_dataset(base_root, output_root, existing_hashes)
    add_stats, add_names = copy_dataset(add_root, output_root, existing_hashes)

    base_usable, base_train, base_val = rows_for_copied(base_root, base_names)
    add_usable, add_train, add_val = rows_for_copied(add_root, add_names)
    write_csv(output_root / "usable_annotations.csv", base_usable + add_usable)
    write_csv(output_root / "train.csv", base_train + add_train)
    write_csv(output_root / "val.csv", base_val + add_val)

    train_count = len(pair_paths(output_root, "train"))
    val_count = len(pair_paths(output_root, "val"))
    summary = {
        "base_root": str(base_root),
        "add_root": str(add_root),
        "output_root": str(output_root),
        "base_stats": base_stats,
        "add_stats": add_stats,
        "train_samples": train_count,
        "val_samples": val_count,
        "total_samples": train_count + val_count,
    }
    (output_root / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
