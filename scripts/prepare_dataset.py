#!/usr/bin/env python3
"""Assemble a YOLO dataset from several annotated sources declared in a config."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import random
import re
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8"
JPEG_SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)
JPEG_STANDALONE = frozenset({0x01, 0xD8} | set(range(0xD0, 0xD8)))
PNG_SUFFIXES = frozenset({".png"})
JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})
ROLES = frozenset({"core", "filler"})
FORMATS = frozenset({"voc", "yolo"})
UNMAPPED_POLICIES = frozenset({"error", "ignore"})


@dataclass(frozen=True)
class Box:
    class_id: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class Source:
    name: str
    role: str
    priority: int
    format: str
    images: Path
    annotations: Path | None
    labels: Path | None
    image_suffixes: tuple[str, ...]
    class_map: dict[str, str | None]
    class_names: tuple[str, ...]
    keep_negatives: bool
    on_unmapped: str
    group_by: re.Pattern[str] | None
    prefix: str
    max_images: int | None
    license: str
    url: str
    citation: str


@dataclass
class Record:
    source: str
    stem: str
    output_name: str
    image: Path
    origin: Path
    width: int
    height: int
    sha256: str
    phash: int | None
    boxes: list[Box]
    raw_counts: Counter
    clipped_boxes: int
    group: str
    notes: list[str] = field(default_factory=list)


@dataclass
class Group:
    source: str
    group_id: str
    role: str
    records: list[Record] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return (self.source, self.group_id)

    def counts(self) -> Counter:
        totals: Counter = Counter()
        for record in self.records:
            totals.update(box.class_id for box in record.boxes)
        return totals


def stratum_of(class_ids: set[int], names: dict[int, str]) -> str:
    if not class_ids:
        return "negative"
    return "+".join(names[class_id] for class_id in sorted(class_ids))


def record_stratum(record: Record, names: dict[int, str]) -> str:
    return stratum_of({box.class_id for box in record.boxes}, names)


def group_stratum(group: Group, names: dict[int, str]) -> str:
    present: set[int] = set()
    for record in group.records:
        present.update(box.class_id for box in record.boxes)
    return stratum_of(present, names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("dataset/sources.yaml"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dataset-yaml", type=Path)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--reports-prefix")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--preview-count", type=int, default=30)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Use only the named source; repeatable, overrides the enabled flag.",
    )
    parser.add_argument(
        "--no-perceptual",
        action="store_true",
        help="Skip perceptual deduplication and image decoding.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated output directory.",
    )
    return parser.parse_args()


def verify_png(path: Path) -> tuple[int, int, str, list[str]]:
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
    return width, height, digest.hexdigest(), []


def verify_jpeg(path: Path) -> tuple[int, int, str, list[str]]:
    """Validate JPEG structure and read its dimensions.

    JPEG carries no checksum and real archives routinely hold files with padding,
    trailing text or no EOI marker at all that still decode to a complete image.
    Those are reported as notes; integrity is established by decoding instead.
    """
    data = path.read_bytes()
    if not data.startswith(JPEG_SIGNATURE):
        raise ValueError("invalid JPEG signature")
    # The condition is deliberately the exact one Ultralytics uses - the final
    # two bytes must be EOI - because anything it judges broken it rewrites in
    # place, and output images are hardlinks back into the source archive.
    notes: list[str] = []
    if data[-2:] != b"\xff\xd9":
        if b"\xff\xd9" in data:
            notes.append("bytes present after the EOI marker")
        else:
            notes.append("no EOI marker")
    width = height = None
    total = len(data)
    offset = 2
    while offset < total - 1:
        if data[offset] != 0xFF:
            raise ValueError("expected a JPEG marker")
        marker = data[offset + 1]
        if marker == 0xFF:
            offset += 1
            continue
        if marker in JPEG_STANDALONE:
            offset += 2
            continue
        if marker == 0xD9:
            break
        if offset + 4 > total:
            raise ValueError("truncated JPEG segment header")
        length = struct.unpack(">H", data[offset + 2 : offset + 4])[0]
        if length < 2 or offset + 2 + length > total:
            raise ValueError("invalid JPEG segment length")
        if marker in JPEG_SOF_MARKERS:
            if length < 7:
                raise ValueError("invalid SOF segment")
            height, width = struct.unpack(">HH", data[offset + 5 : offset + 9])
        if marker == 0xDA:
            break
        offset += 2 + length
    if not width or not height:
        raise ValueError("JPEG is missing SOF dimensions")
    return width, height, hashlib.sha256(data).hexdigest(), notes


def verify_image(path: Path) -> tuple[int, int, str, list[str]]:
    suffix = path.suffix.lower()
    if suffix in PNG_SUFFIXES:
        return verify_png(path)
    if suffix in JPEG_SUFFIXES:
        return verify_jpeg(path)
    raise ValueError(f"unsupported image suffix {suffix}")


def decode_image(path: Path) -> None:
    """Force a full decode so truncated or malformed pixel data is rejected."""
    with Image.open(path) as image:
        image.load()


def perceptual_hash(path: Path, size: int) -> int:
    with Image.open(path) as image:
        grey = image.convert("L").resize(
            (size + 1, size), Image.Resampling.LANCZOS
        )
        pixels = list(grey.getdata())
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for column in range(size):
            bits <<= 1
            if pixels[base + column] > pixels[base + column + 1]:
                bits |= 1
    return bits


class Deduper:
    def __init__(self, hash_size: int, max_distance: int, perceptual: bool) -> None:
        self.bits = hash_size * hash_size
        self.max_distance = max_distance
        self.perceptual = perceptual
        band_count = min(max_distance + 1, self.bits)
        base, extra = divmod(self.bits, band_count)
        self.band_widths = tuple(
            base + (1 if index < extra else 0) for index in range(band_count)
        )
        self.index: dict[tuple[int, int], list[Record]] = defaultdict(list)
        self.by_sha: dict[str, Record] = {}

    def _bands(self, value: int) -> list[tuple[int, int]]:
        bands = []
        shift = 0
        for index, width in enumerate(self.band_widths):
            bands.append((index, (value >> shift) & ((1 << width) - 1)))
            shift += width
        return bands

    def find(self, record: Record) -> tuple[Record | None, str, int]:
        exact = self.by_sha.get(record.sha256)
        if exact is not None:
            return exact, "exact", 0
        if not self.perceptual or record.phash is None:
            return None, "", -1
        best: Record | None = None
        best_distance = self.max_distance + 1
        for band in self._bands(record.phash):
            for other in self.index.get(band, ()):
                if other.phash is None:
                    continue
                distance = (other.phash ^ record.phash).bit_count()
                if distance <= self.max_distance and distance < best_distance:
                    best, best_distance = other, distance
        if best is not None:
            return best, "perceptual", best_distance
        return None, "", -1

    def add(self, record: Record) -> None:
        self.by_sha.setdefault(record.sha256, record)
        if self.perceptual and record.phash is not None:
            for band in self._bands(record.phash):
                self.index[band].append(record)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Config error: {message}")


def build_source(raw: dict, names: dict[int, str], index: int) -> Source:
    name = str(raw.get("name") or f"source[{index}]")
    require("name" in raw, f"{name}: missing name")
    role = str(raw.get("role", "core"))
    require(role in ROLES, f"{name}: role must be one of {sorted(ROLES)}")
    fmt = str(raw.get("format", "voc"))
    require(fmt in FORMATS, f"{name}: format must be one of {sorted(FORMATS)}")
    policy = str(raw.get("on_unmapped", "error"))
    require(
        policy in UNMAPPED_POLICIES,
        f"{name}: on_unmapped must be one of {sorted(UNMAPPED_POLICIES)}",
    )
    require("images" in raw, f"{name}: missing images directory")
    require("class_map" in raw, f"{name}: missing class_map")

    valid_targets = set(names.values())
    class_map: dict[str, str | None] = {}
    for key, value in dict(raw["class_map"]).items():
        target = None if value is None else str(value)
        require(
            target is None or target in valid_targets,
            f"{name}: class_map maps {key!r} to unknown target {target!r}",
        )
        class_map[str(key).strip().lower()] = target

    annotations = labels = None
    if fmt == "voc":
        require("annotations" in raw, f"{name}: voc source needs an annotations directory")
        annotations = Path(raw["annotations"])
    else:
        require("labels" in raw, f"{name}: yolo source needs a labels directory")
        labels = Path(raw["labels"])
        require("class_names" in raw, f"{name}: yolo source needs class_names")

    suffixes = tuple(
        str(item).lower() for item in raw.get("image_suffixes", [".png", ".jpg", ".jpeg"])
    )
    require(bool(suffixes), f"{name}: image_suffixes must not be empty")
    for suffix in suffixes:
        require(
            suffix in PNG_SUFFIXES | JPEG_SUFFIXES,
            f"{name}: unsupported image suffix {suffix}",
        )

    pattern = raw.get("group_by")
    compiled = None
    if pattern:
        compiled = re.compile(str(pattern))
        require(
            compiled.groups >= 1,
            f"{name}: group_by needs one capturing group",
        )

    max_images = raw.get("max_images")
    if max_images is not None:
        max_images = int(max_images)
        require(max_images > 0, f"{name}: max_images must be positive")

    return Source(
        name=name,
        role=role,
        priority=int(raw.get("priority", 0)),
        format=fmt,
        images=Path(raw["images"]),
        annotations=annotations,
        labels=labels,
        image_suffixes=suffixes,
        class_map=class_map,
        class_names=tuple(str(item).strip().lower() for item in raw.get("class_names", ())),
        keep_negatives=bool(raw.get("keep_negatives", False)),
        on_unmapped=policy,
        group_by=compiled,
        prefix=str(raw.get("prefix", "")),
        max_images=max_images,
        license=str(raw.get("license", "")),
        url=str(raw.get("url", "")),
        citation=str(raw.get("citation", "")),
    )


def load_config(path: Path, args: argparse.Namespace) -> tuple[dict, list[Source], dict[int, str]]:
    if not path.is_file():
        raise SystemExit(f"Config not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    raw_classes = config.get("target_classes") or {}
    require(bool(raw_classes), "target_classes must not be empty")
    names = {int(key): str(value) for key, value in raw_classes.items()}
    require(
        sorted(names) == list(range(len(names))),
        "target_classes ids must be contiguous and start at 0",
    )

    split = config.get("split") or {}
    train = float(split.get("train", 0.8))
    val = float(split.get("val", 0.1))
    require(
        train > 0 and val > 0 and train + val < 1,
        "split must satisfy train > 0, val > 0, train + val < 1",
    )
    config["_train"] = train
    config["_val"] = val
    config["_seed"] = int(args.seed if args.seed is not None else config.get("seed", 0))

    raw_sources = config.get("sources") or []
    require(bool(raw_sources), "sources must not be empty")
    sources = [build_source(raw, names, index) for index, raw in enumerate(raw_sources)]

    wanted = set(args.only)
    if wanted:
        known = {source.name for source in sources}
        missing = sorted(wanted - known)
        require(not missing, f"--only names an unknown source: {', '.join(missing)}")
        selected = [source for source in sources if source.name in wanted]
    else:
        selected = [
            source
            for source, raw in zip(sources, raw_sources)
            if bool(raw.get("enabled", False))
        ]
    require(bool(selected), "no source is enabled; use enabled: true or --only")

    seen: set[str] = set()
    for source in selected:
        require(source.name not in seen, f"duplicate source name {source.name}")
        seen.add(source.name)

    selected.sort(key=lambda source: (-source.priority, source.name))
    return config, selected, names


def group_id_for(source: Source, stem: str) -> str:
    if source.group_by is None:
        return stem
    match = source.group_by.match(stem)
    if match is None:
        return stem
    return match.group(1)


def clip_boxes(
    raw_boxes: list[tuple[str, float, float, float, float, int]],
    source: Source,
    names: dict[int, str],
    width: int,
    height: int,
    origin: Path,
    issues: list[dict],
) -> tuple[list[Box], int]:
    target_ids = {value: key for key, value in names.items()}
    boxes: list[Box] = []
    clipped = 0
    for label, xmin, ymin, xmax, ymax, order in raw_boxes:
        target = source.class_map.get(label)
        if target is None:
            continue
        original = (xmin, ymin, xmax, ymax)
        xmin = min(max(xmin, 0.0), width)
        ymin = min(max(ymin, 0.0), height)
        xmax = min(max(xmax, 0.0), width)
        ymax = min(max(ymax, 0.0), height)
        if original != (xmin, ymin, xmax, ymax):
            clipped += 1
        if xmax - xmin < 1.0 or ymax - ymin < 1.0:
            issues.append(
                {
                    "source": source.name,
                    "file": origin.name,
                    "kind": "invalid_box",
                    "object": order,
                    "detail": "box is empty or smaller than one pixel after clipping",
                }
            )
            continue
        boxes.append(Box(target_ids[target], xmin, ymin, xmax, ymax))
    return boxes, clipped


def number(element: ET.Element | None, field_name: str) -> float:
    if element is None or element.text is None:
        raise ValueError(f"missing {field_name}")
    value = float(element.text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field_name}")
    return value


_IMAGE_INDEX: dict[Path, dict[str, Path]] = {}


def image_index(directory: Path) -> dict[str, Path]:
    """Map lowercased file names to the real paths in an image directory.

    Archives mix suffix casing: SHWD ships 7571 .jpg next to 10 .JPG. Building a
    candidate path from the configured suffix finds those only on a case
    insensitive filesystem, so on Linux ten annotated images would vanish
    without a word.
    """
    cached = _IMAGE_INDEX.get(directory)
    if cached is None:
        cached = {}
        for path in directory.iterdir():
            if path.is_file():
                cached.setdefault(path.name.lower(), path)
        _IMAGE_INDEX[directory] = cached
    return cached


def resolve_image(source: Source, stem: str, declared: str | None) -> Path:
    if declared:
        candidate = source.images / declared
        if candidate.is_file():
            return candidate
    for suffix in source.image_suffixes:
        candidate = source.images / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    index = image_index(source.images)
    if declared:
        found = index.get(declared.lower())
        if found is not None:
            return found
    for suffix in source.image_suffixes:
        found = index.get(f"{stem}{suffix}".lower())
        if found is not None:
            return found
    raise ValueError(f"missing image for {stem}")


def parse_voc(
    xml_path: Path, source: Source, names: dict[int, str]
) -> tuple[Record | None, list[dict], Counter]:
    issues: list[dict] = []
    raw_counts: Counter = Counter()
    notes: list[str] = []
    try:
        root = ET.parse(xml_path).getroot()
        declared = (root.findtext("filename") or "").strip()
        image_path = resolve_image(source, xml_path.stem, declared or None)
        xml_width = int(number(root.find("./size/width"), "width"))
        xml_height = int(number(root.find("./size/height"), "height"))
        width, height, sha, notes = verify_image(image_path)
        if (xml_width, xml_height) != (width, height):
            raise ValueError(
                f"XML size {xml_width}x{xml_height} != image size {width}x{height}"
            )
    except (ET.ParseError, OSError, ValueError) as exc:
        issues.append(
            {
                "source": source.name,
                "file": xml_path.name,
                "kind": "invalid_record",
                "detail": str(exc),
            }
        )
        return None, issues, raw_counts

    for note in notes:
        issues.append(
            {
                "source": source.name,
                "file": image_path.name,
                "kind": "image_note",
                "detail": note,
            }
        )

    raw_boxes: list[tuple[str, float, float, float, float, int]] = []
    for order, obj in enumerate(root.findall("object")):
        label = (obj.findtext("name") or "").strip().lower()
        raw_counts[label or "<missing>"] += 1
        bbox = obj.find("bndbox")
        try:
            xmin = number(None if bbox is None else bbox.find("xmin"), "xmin")
            ymin = number(None if bbox is None else bbox.find("ymin"), "ymin")
            xmax = number(None if bbox is None else bbox.find("xmax"), "xmax")
            ymax = number(None if bbox is None else bbox.find("ymax"), "ymax")
        except ValueError as exc:
            issues.append(
                {
                    "source": source.name,
                    "file": xml_path.name,
                    "kind": "invalid_box",
                    "object": order,
                    "detail": str(exc),
                }
            )
            continue
        raw_boxes.append((label, xmin, ymin, xmax, ymax, order))

    boxes, clipped = clip_boxes(
        raw_boxes, source, names, width, height, xml_path, issues
    )
    stem = Path(declared).stem if declared else xml_path.stem
    record = Record(
        source=source.name,
        stem=stem,
        output_name=f"{source.prefix}{image_path.name}",
        image=image_path,
        origin=xml_path,
        width=width,
        height=height,
        sha256=sha,
        phash=None,
        boxes=boxes,
        raw_counts=raw_counts,
        clipped_boxes=clipped,
        group=group_id_for(source, stem),
        notes=notes,
    )
    return record, issues, raw_counts


def parse_yolo(
    label_path: Path, source: Source, names: dict[int, str]
) -> tuple[Record | None, list[dict], Counter]:
    issues: list[dict] = []
    raw_counts: Counter = Counter()
    try:
        image_path = resolve_image(source, label_path.stem, None)
        width, height, sha, notes = verify_image(image_path)
    except (OSError, ValueError) as exc:
        issues.append(
            {
                "source": source.name,
                "file": label_path.name,
                "kind": "invalid_record",
                "detail": str(exc),
            }
        )
        return None, issues, raw_counts

    for note in notes:
        issues.append(
            {
                "source": source.name,
                "file": image_path.name,
                "kind": "image_note",
                "detail": note,
            }
        )

    raw_boxes: list[tuple[str, float, float, float, float, int]] = []
    for order, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            issues.append(
                {
                    "source": source.name,
                    "file": label_path.name,
                    "kind": "invalid_box",
                    "object": order,
                    "detail": f"expected 5 fields, got {len(parts)}",
                }
            )
            continue
        try:
            class_index = int(parts[0])
            x_center, y_center, box_width, box_height = (
                float(value) for value in parts[1:]
            )
        except ValueError as exc:
            issues.append(
                {
                    "source": source.name,
                    "file": label_path.name,
                    "kind": "invalid_box",
                    "object": order,
                    "detail": str(exc),
                }
            )
            continue
        if not 0 <= class_index < len(source.class_names):
            raw_counts[f"<id {class_index}>"] += 1
            continue
        label = source.class_names[class_index]
        raw_counts[label] += 1
        raw_boxes.append(
            (
                label,
                (x_center - box_width / 2.0) * width,
                (y_center - box_height / 2.0) * height,
                (x_center + box_width / 2.0) * width,
                (y_center + box_height / 2.0) * height,
                order,
            )
        )

    boxes, clipped = clip_boxes(
        raw_boxes, source, names, width, height, label_path, issues
    )
    record = Record(
        source=source.name,
        stem=label_path.stem,
        output_name=f"{source.prefix}{image_path.name}",
        image=image_path,
        origin=label_path,
        width=width,
        height=height,
        sha256=sha,
        phash=None,
        boxes=boxes,
        raw_counts=raw_counts,
        clipped_boxes=clipped,
        group=group_id_for(source, label_path.stem),
        notes=notes,
    )
    return record, issues, raw_counts


def collect_source(
    source: Source,
    names: dict[int, str],
    deduper: Deduper,
    hash_size: int,
    perceptual: bool,
    issues: list[dict],
    duplicates: list[dict],
    unmapped: dict[str, Counter],
    dropped: Counter,
) -> list[Record]:
    if not source.images.is_dir():
        raise SystemExit(f"{source.name}: images directory not found: {source.images}")
    if source.format == "voc":
        assert source.annotations is not None
        if not source.annotations.is_dir():
            raise SystemExit(
                f"{source.name}: annotations directory not found: {source.annotations}"
            )
        origins = sorted(source.annotations.glob("*.xml"))
        parser = parse_voc
    else:
        assert source.labels is not None
        if not source.labels.is_dir():
            raise SystemExit(
                f"{source.name}: labels directory not found: {source.labels}"
            )
        origins = sorted(source.labels.glob("*.txt"))
        parser = parse_yolo
    if not origins:
        raise SystemExit(f"{source.name}: no annotation files found")

    records: list[Record] = []
    for index, origin in enumerate(origins, start=1):
        record, record_issues, raw_counts = parser(origin, source, names)
        issues.extend(record_issues)
        for label, count in raw_counts.items():
            if label not in source.class_map:
                unmapped[source.name][label] += count
        if record is None:
            continue
        if not record.boxes:
            if record.raw_counts:
                ignored_only = all(
                    source.class_map.get(label) is None for label in record.raw_counts
                )
                reason = "ignored_only" if ignored_only else "all_boxes_invalid"
                dropped[f"{source.name}:{reason}"] += 1
                continue
            if not source.keep_negatives:
                dropped[f"{source.name}:negative"] += 1
                continue
        try:
            if perceptual:
                record.phash = perceptual_hash(record.image, hash_size)
            elif record.image.suffix.lower() in JPEG_SUFFIXES:
                decode_image(record.image)
        except (OSError, ValueError) as exc:
            issues.append(
                {
                    "source": source.name,
                    "file": record.image.name,
                    "kind": "undecodable_image",
                    "detail": str(exc),
                }
            )
            continue
        match, kind, distance = deduper.find(record)
        if match is not None:
            duplicates.append(
                {
                    "source": source.name,
                    "file": record.image.name,
                    "kind": kind,
                    "distance": distance,
                    "kept_source": match.source,
                    "kept_file": match.image.name,
                }
            )
            continue
        deduper.add(record)
        records.append(record)
        if index % 500 == 0:
            print(f"{source.name}: audited {index}/{len(origins)}", file=sys.stderr)
    return records


def build_groups(records: list[Record], role: str) -> list[Group]:
    grouped: dict[tuple[str, str], Group] = {}
    for record in records:
        key = (record.source, record.group)
        group = grouped.get(key)
        if group is None:
            group = Group(source=record.source, group_id=record.group, role=role)
            grouped[key] = group
        group.records.append(record)
    for group in grouped.values():
        group.records.sort(key=lambda record: (record.source, record.stem))
    return list(grouped.values())


def select_fillers(
    core_counts: Counter,
    groups: list[Group],
    weights: dict[str, float],
    names: dict[int, str],
    caps: dict[str, int | None],
) -> tuple[list[Group], dict]:
    target_ids = {value: key for key, value in names.items()}
    anchor = 0.0
    for name, weight in weights.items():
        if weight <= 0 or name not in target_ids:
            continue
        anchor = max(anchor, core_counts.get(target_ids[name], 0) / weight)
    desired = {
        target_ids[name]: anchor * weight
        for name, weight in weights.items()
        if weight > 0 and name in target_ids
    }
    remaining = {
        class_id: max(0.0, value - core_counts.get(class_id, 0))
        for class_id, value in desired.items()
    }
    deficit_ids = {class_id for class_id, value in remaining.items() if value > 0}

    def rank(group: Group) -> tuple:
        counts = group.counts()
        gain = sum(counts.get(class_id, 0) for class_id in deficit_ids)
        cost = sum(
            count for class_id, count in counts.items() if class_id not in deficit_ids
        )
        mixed = len({class_id for class_id in counts if counts[class_id] > 0}) > 1
        return (0 if mixed else 1, -gain, cost, group.key)

    chosen: list[Group] = []
    used: Counter = Counter()
    per_source: Counter = Counter()
    for group in sorted(groups, key=rank):
        if not deficit_ids or all(value <= 0 for value in remaining.values()):
            break
        counts = group.counts()
        if not any(counts.get(class_id, 0) for class_id in deficit_ids):
            continue
        cap = caps.get(group.source)
        if cap is not None and per_source[group.source] + len(group.records) > cap:
            continue
        chosen.append(group)
        per_source[group.source] += len(group.records)
        used.update(counts)
        for class_id in list(remaining):
            remaining[class_id] -= counts.get(class_id, 0)

    report = {
        "anchor": anchor,
        "desired_objects": {names[k]: round(v, 3) for k, v in desired.items()},
        "core_objects": {names[k]: core_counts.get(k, 0) for k in names},
        "filler_objects_added": {names[k]: used.get(k, 0) for k in names},
        "unmet_deficit": {
            names[k]: round(max(0.0, v), 3) for k, v in remaining.items()
        },
        "filler_images_added": dict(per_source),
        "filler_groups_available": len(groups),
        "filler_groups_used": len(chosen),
    }
    return chosen, report


def cap_negatives(
    groups: list[Group], names: dict[int, str], fraction: float
) -> tuple[list[Group], dict]:
    if fraction >= 1.0:
        return groups, {"limit": None, "dropped": 0}
    negatives = [group for group in groups if group_stratum(group, names) == "negative"]
    positives = [group for group in groups if group_stratum(group, names) != "negative"]
    positive_images = sum(len(group.records) for group in positives)
    if fraction <= 0.0:
        limit = 0
    else:
        limit = int(math.floor(fraction / (1.0 - fraction) * positive_images))
    kept: list[Group] = []
    used = 0
    dropped = 0
    for group in sorted(negatives, key=lambda item: item.key):
        if used + len(group.records) <= limit:
            kept.append(group)
            used += len(group.records)
        else:
            dropped += len(group.records)
    return positives + kept, {
        "limit": limit,
        "kept": used,
        "dropped": dropped,
        "positive_images": positive_images,
    }


def split_groups(
    groups: list[Group],
    names: dict[int, str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[Record]]:
    strata: dict[str, list[Group]] = defaultdict(list)
    for group in groups:
        strata[group_stratum(group, names)].append(group)
    result: dict[str, list[Record]] = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    for name in sorted(strata):
        values = sorted(strata[name], key=lambda item: item.key)
        rng.shuffle(values)
        train_end = round(len(values) * train_ratio)
        val_end = train_end + round(len(values) * val_ratio)
        for split_name, chunk in (
            ("train", values[:train_end]),
            ("val", values[train_end:val_end]),
            ("test", values[val_end:]),
        ):
            for group in chunk:
                result[split_name].extend(group.records)
    for values in result.values():
        values.sort(key=lambda item: (item.source, item.stem))
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


def repair_jpeg_tail(source: Path, destination: Path) -> str:
    """Copy a JPEG, normalising its tail so the last two bytes are the EOI marker.

    Ultralytics rewrites in place any JPEG that does not end in FFD9. Output
    images are hardlinks, so that rewrite reaches through the link and re-encodes
    the pristine source archive. Truncating trailing bytes or appending the
    marker leaves the entropy-coded data untouched, so no pixel is re-encoded and
    the source stays exactly as it was downloaded.
    """
    data = source.read_bytes()
    end = data.rfind(b"\xff\xd9")
    repaired = data[: end + 2] if end != -1 else data + b"\xff\xd9"
    destination.unlink(missing_ok=True)
    destination.write_bytes(repaired)
    return "repaired copy"


def link_or_copy(source: Path, destination: Path) -> str:
    """Materialise one image in the output tree.

    Duplicate output names are rejected before anything is written, so a
    destination that already exists means leftover state in a directory this
    script owns. Replacing it keeps the write idempotent rather than failing
    halfway through with a half-built dataset on disk.
    """
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def write_preview(
    path: Path,
    selected: list[tuple[str, Record]],
    output_root: Path,
    names: dict[int, str],
) -> None:
    cards = []
    palette = ["#25d366", "#ff3b30", "#0a84ff", "#ffd60a", "#bf5af2"]
    for split, record in selected:
        image_path = output_root / "images" / split / record.output_name
        relative = os.path.relpath(image_path, path.parent)
        overlays = []
        for box in record.boxes:
            colour = palette[box.class_id % len(palette)]
            left = 100.0 * box.xmin / record.width
            top = 100.0 * box.ymin / record.height
            width = 100.0 * (box.xmax - box.xmin) / record.width
            height = 100.0 * (box.ymax - box.ymin) / record.height
            label = html.escape(names[box.class_id])
            overlays.append(
                f'<div class="box" style="left:{left:.4f}%;top:{top:.4f}%;'
                f'width:{width:.4f}%;height:{height:.4f}%;'
                f'border-color:{colour}"><span style="background:'
                f'{colour}">{label}</span></div>'
            )
        cards.append(
            '<article><div class="canvas">'
            f'<img src="{html.escape(relative)}" alt="{html.escape(record.stem)}">'
            + "".join(overlays)
            + f'</div><p>{html.escape(record.stem)} · {split}'
            f'<br><b>{html.escape(record.source)}</b></p></article>'
        )
    legend = " · ".join(
        f'<span style="color:{palette[class_id % len(palette)]}">'
        f"{html.escape(names[class_id])}</span>"
        for class_id in sorted(names)
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
<p>""" + legend + """</p>
<div class="grid">""" + "".join(cards) + "</div>"
    path.write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()
    config, sources, names = load_config(args.config, args)
    seed = config["_seed"]
    train_ratio = config["_train"]
    val_ratio = config["_val"]

    output_root = args.output_root or Path(
        config.get("output_root", "dataset/processed/helmet_yolo_v2")
    )
    dataset_yaml = args.dataset_yaml or Path(
        config.get("dataset_yaml", "dataset/dataset_v2.yaml")
    )
    prefix = args.reports_prefix or str(config.get("reports_prefix", "dataset_v2"))

    dedup_config = config.get("dedup") or {}
    hash_size = int(dedup_config.get("hash_size", 8))
    perceptual = bool(dedup_config.get("perceptual", True)) and not args.no_perceptual
    deduper = Deduper(
        hash_size=hash_size,
        max_distance=int(dedup_config.get("max_distance", 4)),
        perceptual=perceptual,
    )

    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"Output {output_root} is not empty; pass --overwrite to replace it"
            )
        shutil.rmtree(output_root)

    issues: list[dict] = []
    duplicates: list[dict] = []
    unmapped: dict[str, Counter] = defaultdict(Counter)
    dropped: Counter = Counter()
    by_source: dict[str, list[Record]] = {}
    for source in sources:
        by_source[source.name] = collect_source(
            source,
            names,
            deduper,
            hash_size,
            perceptual,
            issues,
            duplicates,
            unmapped,
            dropped,
        )

    strict = {
        source.name for source in sources if source.on_unmapped == "error"
    }
    offending = {
        name: dict(counter)
        for name, counter in unmapped.items()
        if name in strict and counter
    }
    if offending:
        lines = ["Undeclared class names found; add them to class_map or map to null:"]
        for name in sorted(offending):
            for label, count in sorted(offending[name].items()):
                lines.append(f"  {name}: {label!r} ({count} objects)")
        raise SystemExit("\n".join(lines))

    roles = {source.name: source.role for source in sources}
    caps = {source.name: source.max_images for source in sources}
    core_records = [
        record
        for source in sources
        if roles[source.name] == "core"
        for record in by_source[source.name]
    ]
    filler_records = [
        record
        for source in sources
        if roles[source.name] == "filler"
        for record in by_source[source.name]
    ]
    core_groups = build_groups(core_records, "core")
    filler_groups = build_groups(filler_records, "filler")

    balance_config = config.get("balance") or {}
    weights = {
        str(key): float(value)
        for key, value in (balance_config.get("weights") or {}).items()
    }
    core_counts: Counter = Counter()
    for record in core_records:
        core_counts.update(box.class_id for box in record.boxes)
    if filler_groups and weights and not core_records:
        raise SystemExit(
            "Every enabled source has role 'filler', so there is no core set to "
            "balance against and the quota would select nothing. Enable a core "
            "source, change this one's role to 'core', or clear balance.weights "
            "to take the filler sources in full."
        )
    if filler_groups and weights:
        chosen_fillers, balance_report = select_fillers(
            core_counts, filler_groups, weights, names, caps
        )
    else:
        chosen_fillers = filler_groups
        balance_report = {"note": "balance disabled or no filler source"}

    groups = core_groups + chosen_fillers
    negatives_fraction = float(balance_config.get("negatives_max_fraction", 1.0))
    groups, negatives_report = cap_negatives(groups, names, negatives_fraction)

    # Compared case insensitively: Windows and macOS would let two names that
    # differ only in case overwrite each other without a word.
    collisions = Counter(
        record.output_name.lower() for group in groups for record in group.records
    )
    clashing = sorted(name for name, count in collisions.items() if count > 1)
    if clashing:
        raise SystemExit(
            "Output image names collide across sources; set a distinct prefix.\n"
            + "\n".join(f"  {name}" for name in clashing[:20])
        )

    split = split_groups(groups, names, seed, train_ratio, val_ratio)
    if not any(split.values()):
        raise SystemExit("No records survived assembly")

    link_modes: Counter = Counter()
    manifest_rows: list[dict] = []
    for split_name, values in split.items():
        image_output = output_root / "images" / split_name
        label_output = output_root / "labels" / split_name
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        for record in values:
            destination = image_output / record.output_name
            if record.notes:
                link_modes[repair_jpeg_tail(record.image, destination)] += 1
            else:
                link_modes[link_or_copy(record.image, destination)] += 1
            label_text = "\n".join(
                yolo_line(box, record.width, record.height) for box in record.boxes
            )
            if label_text:
                label_text += "\n"
            (label_output / f"{Path(record.output_name).stem}.txt").write_text(
                label_text, encoding="utf-8"
            )
            counts = Counter(box.class_id for box in record.boxes)
            row = {
                "file": record.output_name,
                "source": record.source,
                "group": record.group,
                "split": split_name,
                "sha256": record.sha256,
                "phash": "" if record.phash is None else f"{record.phash:016x}",
                "width": record.width,
                "height": record.height,
                "stratum": record_stratum(record, names),
            }
            for class_id in sorted(names):
                row[names[class_id]] = counts[class_id]
            manifest_rows.append(row)

    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    relative_root = os.path.relpath(output_root, dataset_yaml.parent).replace("\\", "/")
    lines = ["# Generated by scripts/prepare_dataset.py"]
    for split_name in ("train", "val", "test"):
        lines.append(f"{split_name}: {relative_root}/images/{split_name}")
    lines.append("")
    lines.append("names:")
    for class_id in sorted(names):
        lines.append(f"  {class_id}: {names[class_id]}")
    dataset_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.reports_dir / f"{prefix}_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    class_by_split = {}
    strata_by_split = {}
    source_by_split = {}
    for split_name, values in split.items():
        counts = Counter(box.class_id for record in values for box in record.boxes)
        class_by_split[split_name] = {
            names[class_id]: counts[class_id] for class_id in sorted(names)
        }
        strata_by_split[split_name] = dict(
            sorted(Counter(record_stratum(record, names) for record in values).items())
        )
        source_by_split[split_name] = dict(
            sorted(Counter(record.source for record in values).items())
        )

    report = {
        "config": str(args.config),
        "seed": seed,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": round(1.0 - train_ratio - val_ratio, 10),
        },
        "target_classes": {str(k): v for k, v in sorted(names.items())},
        "sources": [
            {
                "name": source.name,
                "role": source.role,
                "priority": source.priority,
                "format": source.format,
                "license": source.license,
                "url": source.url,
                "citation": source.citation,
                "class_map": source.class_map,
                "accepted_images": len(by_source[source.name]),
            }
            for source in sources
        ],
        "dedup": {
            "perceptual": perceptual,
            "hash_size": hash_size,
            "max_distance": deduper.max_distance,
            "bands": list(deduper.band_widths),
            "removed": len(duplicates),
            "removed_by_kind": dict(Counter(item["kind"] for item in duplicates)),
            "examples": duplicates[:50],
        },
        "dropped_images": dict(sorted(dropped.items())),
        "balance": balance_report,
        "negatives": negatives_report,
        "images_total": sum(len(values) for values in split.values()),
        "images_by_split": {name: len(values) for name, values in split.items()},
        "objects_by_split": class_by_split,
        "strata_by_split": strata_by_split,
        "sources_by_split": source_by_split,
        "issues_by_kind": dict(sorted(Counter(item["kind"] for item in issues).items())),
        "issues": issues[:200],
        "clipped_boxes": sum(
            record.clipped_boxes for values in split.values() for record in values
        ),
        "link_modes": dict(link_modes),
    }
    (args.reports_dir / f"{prefix}_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    preview_pool = [
        (split_name, record)
        for split_name, values in split.items()
        for record in values
    ]
    preview_rng = random.Random(seed + 1)
    preview_rng.shuffle(preview_pool)
    write_preview(
        args.reports_dir / f"{prefix}_preview.html",
        preview_pool[: args.preview_count],
        output_root,
        names,
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
