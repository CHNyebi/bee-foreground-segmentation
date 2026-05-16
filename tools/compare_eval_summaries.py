"""Rank multiple evaluation summary.json files by a chosen metric."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ["iou", "dice", "precision", "recall", "accuracy"]


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def load_row(path: Path, scope: str, metric: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get(scope) or {}
    if metric not in metrics:
        raise KeyError(f"{path} does not contain {scope}.{metric}")
    row: dict[str, object] = {
        "summary": str(path),
        "checkpoint": payload.get("checkpoint", ""),
        "eval_set": payload.get("eval_set", ""),
        "samples": payload.get("samples", ""),
        "threshold": payload.get("threshold", ""),
        "postprocess": payload.get("postprocess", ""),
        "rank_metric": metric,
        "rank_value": float(metrics[metric]),
    }
    for name in METRICS:
        row[name] = float(metrics.get(name, 0.0))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", help="Evaluation summary.json files")
    parser.add_argument("--scope", default="pixel_micro_metrics", choices=["pixel_micro_metrics", "image_macro_metrics"])
    parser.add_argument("--metric", default="iou", choices=METRICS)
    parser.add_argument("--output", default="outputs/model_comparison.csv")
    args = parser.parse_args()

    rows = [load_row(Path(path).resolve(), args.scope, args.metric) for path in args.summaries]
    rows.sort(key=lambda row: float(row["rank_value"]), reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "rank_metric",
        "rank_value",
        "summary",
        "checkpoint",
        "eval_set",
        "samples",
        "threshold",
        "postprocess",
        *METRICS,
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote comparison to {output_path}")
    for row in rows:
        print(
            f"#{row['rank']} {args.metric}={fmt(row['rank_value'])} "
            f"dice={fmt(row['dice'])} recall={fmt(row['recall'])} "
            f"precision={fmt(row['precision'])} postprocess={row['postprocess']} "
            f"{row['summary']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
