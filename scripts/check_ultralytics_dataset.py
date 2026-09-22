#!/usr/bin/env python3
"""Load every dataset split with Ultralytics and print a compact report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import ultralytics
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset/dataset.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/ultralytics_dataset_check.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = check_det_dataset(str(args.data.resolve()), autodownload=False)
    result = {
        "python": sys.version.split()[0],
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "data": str(args.data),
        "names": data["names"],
        "splits": {},
    }

    for split_name in ("train", "val", "test"):
        dataset = YOLODataset(
            img_path=data[split_name],
            data=data,
            task="detect",
            imgsz=args.imgsz,
            augment=False,
            cache=False,
            prefix=f"{split_name}: ",
        )
        counts: Counter[int] = Counter()
        for label in dataset.labels:
            counts.update(int(value) for value in label["cls"].reshape(-1))
        sample = dataset[0]
        result["splits"][split_name] = {
            "images": len(dataset),
            "objects": {
                data["names"][class_id]: counts[class_id]
                for class_id in sorted(data["names"])
            },
            "sample_image_shape": list(sample["img"].shape),
            "sample_boxes": int(sample["bboxes"].shape[0]),
        }

    report_text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_text, encoding="utf-8")
    print(report_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
