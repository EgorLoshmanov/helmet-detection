#!/usr/bin/env python3
"""Create a reproducible provisional RKNN INT8 calibration set."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import cv2

from validate_onnx import annotation_group, sha256


SEED = 20260919
GROUPS = ("helmet_only", "no_helmet_only", "both_classes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/processed/helmet_yolo_v1/images/train"))
    parser.add_argument("--labels-dir", type=Path, default=Path("dataset/processed/helmet_yolo_v1/labels/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/processed/rknn_calibration_v1"))
    parser.add_argument("--manifest", type=Path, default=Path("export/calibration_dataset.txt"))
    parser.add_argument("--report", type=Path, default=Path("reports/rknn_calibration.json"))
    parser.add_argument("--per-class", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--restore", action="store_true", help="Restore ignored calibration images from the tracked report")
    return parser.parse_args()


def letterbox(image, size: int):
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized_width, resized_height = round(width * ratio), round(height * ratio)
    pad_width = (size - resized_width) / 2
    pad_height = (size - resized_height) / 2
    if (width, height) != (resized_width, resized_height):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(
        image,
        round(pad_height - 0.1),
        round(pad_height + 0.1),
        round(pad_width - 0.1),
        round(pad_width + 0.1),
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )


def main() -> int:
    args = parse_args()
    if args.per_class < 1:
        raise SystemExit("--per-class must be at least 1")
    if args.restore:
        if not args.report.is_file():
            raise SystemExit(f"Calibration report is missing: {args.report}")
        report = json.loads(args.report.read_text())
        records = report["records"]
        for record in records:
            source = Path(record["source"])
            image = Path(record["calibration_image"])
            if not source.is_file() or sha256(source) != record["source_sha256"]:
                raise SystemExit(f"Source image is missing or changed: {source}")
            if not image.exists():
                decoded = cv2.imread(str(source))
                if decoded is None:
                    raise SystemExit(f"Cannot decode {source}")
                image.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image), letterbox(decoded, report["imgsz"])):
                    raise SystemExit(f"Cannot restore {image}")
            if sha256(image) != record["calibration_sha256"]:
                raise SystemExit(f"Calibration image has changed: {image}")
        write_manifest(args.manifest, records)
        print(f"Restored {len(records)} calibration images and {args.manifest}")
        return 0
    if args.manifest.exists() or args.report.exists():
        raise SystemExit("Calibration manifest or report already exists; choose a new output path")
    candidates = {group: [] for group in GROUPS}
    for image in sorted(args.images_dir.glob("*.png")):
        label = args.labels_dir / f"{image.stem}.txt"
        if label.is_file():
            group, _ = annotation_group(label)
            if group in candidates:
                candidates[group].append(image)
    random_generator = random.Random(SEED)
    selected = []
    for group, paths in candidates.items():
        if len(paths) < args.per_class:
            raise SystemExit(f"Not enough {group} images: {len(paths)} < {args.per_class}")
        selected.extend((path, group) for path in random_generator.sample(paths, args.per_class))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source, group in selected:
        image = cv2.imread(str(source))
        if image is None:
            raise SystemExit(f"Cannot decode {source}")
        destination = args.output_dir / source.name
        if destination.exists():
            raise SystemExit(f"Calibration image already exists: {destination}")
        padded = letterbox(image, args.imgsz)
        if padded.shape != (args.imgsz, args.imgsz, 3) or not cv2.imwrite(str(destination), padded):
            raise SystemExit(f"Failed to write {destination}")
        records.append({
            "source": str(source),
            "source_sha256": sha256(source),
            "group": group,
            "calibration_image": str(destination),
            "calibration_sha256": sha256(destination),
        })

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(args.manifest, records)
    args.report.write_text(json.dumps({
        "purpose": "Provisional INT8 calibration from v1 training images; replace with target-camera scenes",
        "seed": SEED,
        "imgsz": args.imgsz,
        "images": len(records),
        "per_group": {group: args.per_class for group in GROUPS},
        "records": records,
    }, indent=2) + "\n")
    print(f"Prepared {len(records)} calibration images and {args.manifest}")
    return 0


def write_manifest(path: Path, records: list[dict]) -> None:
    # Toolkit resolves entries relative to the manifest's directory.
    base = path.parent.resolve()
    lines = [os.path.relpath(Path(record["calibration_image"]).resolve(), base) for record in records]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
