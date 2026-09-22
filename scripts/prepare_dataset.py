#!/usr/bin/env python3
"""Audit Pascal VOC helmet data and build a deterministic YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import random
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


CLASS_MAP = {"helmet": 0, "head": 1}
CLASS_NAMES = {0: "helmet", 1: "no_helmet"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class Box:
    class_id: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class Record:
    stem: str
    image: Path
    annotation: Path
    width: int
    height: int
    sha256: str
    boxes: list[Box]
    source_counts: Counter[str]
    clipped_boxes: int

    @property
    def stratum(self) -> str:
        present = {box.class_id for box in self.boxes}
        if present == {0, 1}:
            return "helmet+no_helmet"
        if present == {0}:
            return "helmet"
        if present == {1}:
            return "no_helmet"
        return "negative"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("dataset/source/hard-hat-detection-v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/processed/helmet_yolo_v1"),
    )
    parser.add_argument(
        "--dataset-yaml", type=Path, default=Path("dataset/dataset.yaml")
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--preview-count", type=int, default=30)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated output directory.",
    )
    return parser.parse_args()


def verify_png(path: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    width = height = None
    saw_iend = False
    with path.open("rb") as handle:
        signature = handle.read(8)
        digest.update(signature)
        if signature != PNG_SIGNATURE:
            raise ValueError("invalid PNG signature")
        while True:
            length_raw = handle.read(4)
            if not length_raw:
                break
            digest.update(length_raw)
            if len(length_raw) != 4:
                raise ValueError("truncated PNG chunk length")
            length = struct.unpack(">I", length_raw)[0]
            chunk_type = handle.read(4)
            chunk_data = handle.read(length)
            crc_raw = handle.read(4)
            digest.update(chunk_type)
            digest.update(chunk_data)
            digest.update(crc_raw)
            if len(chunk_type) != 4 or len(chunk_data) != length or len(crc_raw) != 4:
                raise ValueError("truncated PNG chunk")
            expected_crc = struct.unpack(">I", crc_raw)[0]
            actual_crc = zlib.crc32(chunk_type)
            actual_crc = zlib.crc32(chunk_data, actual_crc) & 0xFFFFFFFF
            if expected_crc != actual_crc:
                raise ValueError(f"CRC mismatch in {chunk_type!r}")
            if chunk_type == b"IHDR":
                if length != 13:
                    raise ValueError("invalid IHDR length")
                width, height = struct.unpack(">II", chunk_data[:8])
            if chunk_type == b"IEND":
                saw_iend = True
                if handle.read(1):
                    raise ValueError("data found after IEND")
                break
    if width is None or height is None or not saw_iend:
        raise ValueError("PNG is missing IHDR or IEND")
    return width, height, digest.hexdigest()


def number(element: ET.Element | None, field: str) -> float:
    if element is None or element.text is None:
        raise ValueError(f"missing {field}")
    value = float(element.text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field}")
    return value


def parse_record(xml_path: Path, images_dir: Path) -> tuple[Record | None, list[dict]]:
    issues: list[dict] = []
    try:
        root = ET.parse(xml_path).getroot()
        filename = (root.findtext("filename") or f"{xml_path.stem}.png").strip()
        image_path = images_dir / filename
        if not image_path.is_file():
            raise ValueError(f"missing image {filename}")
        xml_width = int(number(root.find("./size/width"), "width"))
        xml_height = int(number(root.find("./size/height"), "height"))
        image_width, image_height, image_sha = verify_png(image_path)
        if (xml_width, xml_height) != (image_width, image_height):
            raise ValueError(
                f"XML size {xml_width}x{xml_height} != PNG size "
                f"{image_width}x{image_height}"
            )
    except (ET.ParseError, OSError, ValueError) as exc:
        issues.append({"file": xml_path.name, "kind": "invalid_record", "detail": str(exc)})
        return None, issues

    boxes: list[Box] = []
    source_counts: Counter[str] = Counter()
    clipped = 0
    for index, obj in enumerate(root.findall("object")):
        name = (obj.findtext("name") or "").strip().lower()
        source_counts[name or "<missing>"] += 1
        if name not in CLASS_MAP:
            continue
        bbox = obj.find("bndbox")
        try:
            xmin = number(None if bbox is None else bbox.find("xmin"), "xmin")
            ymin = number(None if bbox is None else bbox.find("ymin"), "ymin")
            xmax = number(None if bbox is None else bbox.find("xmax"), "xmax")
            ymax = number(None if bbox is None else bbox.find("ymax"), "ymax")
        except ValueError as exc:
            issues.append(
                {
                    "file": xml_path.name,
                    "kind": "invalid_box",
                    "object": index,
                    "detail": str(exc),
                }
            )
            continue
        original = (xmin, ymin, xmax, ymax)
        xmin = min(max(xmin, 0.0), image_width)
        ymin = min(max(ymin, 0.0), image_height)
        xmax = min(max(xmax, 0.0), image_width)
        ymax = min(max(ymax, 0.0), image_height)
        if original != (xmin, ymin, xmax, ymax):
            clipped += 1
        if xmax - xmin < 1.0 or ymax - ymin < 1.0:
            issues.append(
                {
                    "file": xml_path.name,
                    "kind": "invalid_box",
                    "object": index,
                    "detail": "box is empty or smaller than one pixel after clipping",
                }
            )
            continue
        boxes.append(Box(CLASS_MAP[name], xmin, ymin, xmax, ymax))

    if not boxes and source_counts.get("person", 0):
        issues.append(
            {
                "file": xml_path.name,
                "kind": "excluded_person_only",
                "detail": "person exists but helmet/head labels are absent",
            }
        )
        return None, issues

    return (
        Record(
            stem=Path(filename).stem,
            image=image_path,
            annotation=xml_path,
            width=image_width,
            height=image_height,
            sha256=image_sha,
            boxes=boxes,
            source_counts=source_counts,
            clipped_boxes=clipped,
        ),
        issues,
    )


def split_records(
    records: list[Record], seed: int, train_ratio: float, val_ratio: float
) -> dict[str, list[Record]]:
    strata: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        strata[record.stratum].append(record)
    result = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    for name in sorted(strata):
        values = sorted(strata[name], key=lambda item: item.stem)
        rng.shuffle(values)
        train_end = round(len(values) * train_ratio)
        val_end = train_end + round(len(values) * val_ratio)
        result["train"].extend(values[:train_end])
        result["val"].extend(values[train_end:val_end])
        result["test"].extend(values[val_end:])
    for values in result.values():
        values.sort(key=lambda item: item.stem)
    return result


def yolo_line(box: Box, width: int, height: int) -> str:
    x_center = ((box.xmin + box.xmax) / 2.0) / width
    y_center = ((box.ymin + box.ymax) / 2.0) / height
    box_width = (box.xmax - box.xmin) / width
    box_height = (box.ymax - box.ymin) / height
    return (
        f"{box.class_id} {x_center:.8f} {y_center:.8f} "
        f"{box_width:.8f} {box_height:.8f}"
    )


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def write_preview(
    path: Path, selected: list[tuple[str, Record]], output_root: Path
) -> None:
    cards = []
    colors = {0: "#25d366", 1: "#ff3b30"}
    for split, record in selected:
        image_path = output_root / "images" / split / record.image.name
        relative = os.path.relpath(image_path, path.parent)
        overlays = []
        for box in record.boxes:
            left = 100.0 * box.xmin / record.width
            top = 100.0 * box.ymin / record.height
            width = 100.0 * (box.xmax - box.xmin) / record.width
            height = 100.0 * (box.ymax - box.ymin) / record.height
            label = html.escape(CLASS_NAMES[box.class_id])
            overlays.append(
                f'<div class="box" style="left:{left:.4f}%;top:{top:.4f}%;'
                f'width:{width:.4f}%;height:{height:.4f}%;'
                f'border-color:{colors[box.class_id]}"><span style="background:'
                f'{colors[box.class_id]}">{label}</span></div>'
            )
        cards.append(
            '<article><div class="canvas">'
            f'<img src="{html.escape(relative)}" alt="{html.escape(record.stem)}">'
            + "".join(overlays)
            + f'</div><p>{html.escape(record.stem)} · {split}</p></article>'
        )
    document = """<!doctype html>
<meta charset="utf-8">
<title>FORTNITEBALLS dataset preview</title>
<style>
body{font:14px system-ui;background:#16181d;color:#eee;margin:24px}
h1{font-size:22px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:18px}
article{background:#23262d;padding:10px;border-radius:10px}.canvas{position:relative;line-height:0}
img{width:100%;height:auto}.box{position:absolute;border:2px solid;box-sizing:border-box}
.box span{position:absolute;left:-2px;top:-18px;color:white;font-size:11px;line-height:16px;padding:0 3px}
p{margin:8px 0 0;color:#bbb}
</style>
<h1>FORTNITEBALLS — случайная выборка разметки</h1>
<p>Зелёный: helmet. Красный: no_helmet.</p>
<div class="grid">""" + "".join(cards) + "</div>"
    path.write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.train <= 0 or args.val <= 0 or args.train + args.val >= 1:
        raise SystemExit("Split ratios must satisfy train > 0, val > 0, train + val < 1")
    images_dir = args.source_root / "images"
    annotations_dir = args.source_root / "annotations"
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise SystemExit(f"Source directories not found under {args.source_root}")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"Output {args.output_root} is not empty; pass --overwrite to replace it"
            )
        shutil.rmtree(args.output_root)

    records: list[Record] = []
    issues: list[dict] = []
    seen_hashes: dict[str, Record] = {}
    duplicates: list[dict] = []
    xml_files = sorted(annotations_dir.glob("*.xml"))
    for index, xml_path in enumerate(xml_files, start=1):
        record, record_issues = parse_record(xml_path, images_dir)
        issues.extend(record_issues)
        if record is None:
            continue
        duplicate_of = seen_hashes.get(record.sha256)
        if duplicate_of is not None:
            duplicates.append(
                {
                    "file": record.image.name,
                    "duplicate_of": duplicate_of.image.name,
                    "same_labels": sorted(record.boxes, key=str)
                    == sorted(duplicate_of.boxes, key=str),
                }
            )
            continue
        seen_hashes[record.sha256] = record
        records.append(record)
        if index % 500 == 0:
            print(f"audited {index}/{len(xml_files)}", file=sys.stderr)

    split = split_records(records, args.seed, args.train, args.val)
    link_modes: Counter[str] = Counter()
    manifest_rows: list[dict] = []
    for split_name, values in split.items():
        image_output = args.output_root / "images" / split_name
        label_output = args.output_root / "labels" / split_name
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        for record in values:
            link_modes[link_or_copy(record.image, image_output / record.image.name)] += 1
            label_text = "\n".join(
                yolo_line(box, record.width, record.height) for box in record.boxes
            )
            if label_text:
                label_text += "\n"
            (label_output / f"{record.stem}.txt").write_text(
                label_text, encoding="utf-8"
            )
            counts = Counter(box.class_id for box in record.boxes)
            manifest_rows.append(
                {
                    "file": record.image.name,
                    "split": split_name,
                    "sha256": record.sha256,
                    "width": record.width,
                    "height": record.height,
                    "helmet": counts[0],
                    "no_helmet": counts[1],
                    "stratum": record.stratum,
                }
            )

    args.dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_yaml.write_text(
        "# Generated by scripts/prepare_dataset.py\n"
        "train: processed/helmet_yolo_v1/images/train\n"
        "val: processed/helmet_yolo_v1/images/val\n"
        "test: processed/helmet_yolo_v1/images/test\n\n"
        "names:\n"
        "  0: helmet\n"
        "  1: no_helmet\n",
        encoding="utf-8",
    )

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.reports_dir / "dataset_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    class_by_split = {}
    strata_by_split = {}
    for split_name, values in split.items():
        class_counts = Counter(box.class_id for record in values for box in record.boxes)
        class_by_split[split_name] = {
            CLASS_NAMES[class_id]: class_counts[class_id] for class_id in CLASS_NAMES
        }
        strata_by_split[split_name] = dict(
            sorted(Counter(record.stratum for record in values).items())
        )
    issue_counts = Counter(item["kind"] for item in issues)
    report = {
        "source": {
            "root": str(args.source_root),
            "xml_files": len(xml_files),
            "license": "CC0-1.0 / Public Domain (per Kaggle source page)",
            "url": "https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection",
        },
        "mapping": {"helmet": "helmet", "head": "no_helmet", "person": None},
        "seed": args.seed,
        "split_ratios": {
            "train": args.train,
            "val": args.val,
            "test": round(1.0 - args.train - args.val, 10),
        },
        "accepted_images": len(records),
        "images_by_split": {name: len(values) for name, values in split.items()},
        "objects_by_split": class_by_split,
        "strata_by_split": strata_by_split,
        "duplicates_removed": len(duplicates),
        "duplicates": duplicates,
        "issues_by_kind": dict(sorted(issue_counts.items())),
        "issues": issues,
        "clipped_boxes": sum(record.clipped_boxes for record in records),
        "link_modes": dict(link_modes),
    }
    (args.reports_dir / "dataset_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    preview_pool = [
        (split_name, record)
        for split_name, values in split.items()
        for record in values
    ]
    preview_rng = random.Random(args.seed + 1)
    preview_rng.shuffle(preview_pool)
    write_preview(
        args.reports_dir / "dataset_preview.html",
        preview_pool[: args.preview_count],
        args.output_root,
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
