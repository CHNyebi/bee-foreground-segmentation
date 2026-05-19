"""Build a CVAT image upload package from a candidate CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import zipfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(text: str) -> str:
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() or ch in "._-" else "_")
    return "".join(keep).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    candidates_csv = Path(args.candidates_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    images_dir = output_dir / "images"
    manifest_path = output_dir / "manifest.csv"
    zip_path = output_dir / "images.zip"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    with candidates_csv.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    out_rows: list[dict[str, str]] = []
    used_names: set[str] = set()
    for idx, row in enumerate(rows, start=1):
        source_text = row.get("source_image_path") or row.get("source") or row.get("image_path")
        if not source_text:
            continue
        source_path = Path(source_text)
        if not source_path.exists():
            continue

        sample_id = row.get("sample_id") or source_path.stem
        flat_stem = safe_name(sample_id) or f"sample_{idx:06d}"
        flat_name = f"{flat_stem}{source_path.suffix.lower()}"
        if flat_name in used_names:
            flat_name = f"{idx:06d}_{flat_name}"
        used_names.add(flat_name)

        dst = images_dir / flat_name
        shutil.copy2(source_path, dst)
        out_rows.append(
            {
                "index": str(idx),
                "sample_id": sample_id,
                "flat_filename": flat_name,
                "filename": source_path.name,
                "relative_path": row.get("relative_path", ""),
                "source_image_path": str(source_path),
                "image_sha256": row.get("image_sha256") or sha256_file(source_path),
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "index",
            "sample_id",
            "flat_filename",
            "filename",
            "relative_path",
            "source_image_path",
            "image_sha256",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for image_path in sorted(images_dir.glob("*")):
            zf.write(image_path, arcname=image_path.name)

    print(f"wrote {len(out_rows)} images")
    print(f"images_dir: {images_dir}")
    print(f"manifest: {manifest_path}")
    print(f"images_zip: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
