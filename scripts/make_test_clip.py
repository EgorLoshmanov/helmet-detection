#!/usr/bin/env python3
"""Build a scripted video clip from a dataset split to exercise firefly_app.

The Firefly stand has no working camera sensor, so the only way to run the whole
chain on the board is to feed it a recording. A clip of randomly ordered frames
proves little: the supervisor confirms a violation only while one is held for
alarm_on_seconds and releases it only after alarm_off_seconds. This builds three
segments instead - safe, violation, safe - long enough for both transitions, and
picks frames by where their boxes fall relative to the configured ROI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np


DEFAULT_ROI = (0.15, 0.15, 0.85, 0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images", type=Path, default=Path("dataset/processed/helmet_yolo_v2/images/test")
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("dataset/processed/helmet_yolo_v2/labels/test")
    )
    parser.add_argument("--output", type=Path, default=Path("runtime/test_clip.mp4"))
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.5,
        help="How long each source image stays on screen.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=4.0,
        help="Length of each safe segment; the violation segment gets 1.5x this.",
    )
    parser.add_argument(
        "--roi",
        type=float,
        nargs=4,
        default=list(DEFAULT_ROI),
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Normalised ROI the clip is built against; must match the app config.",
    )
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_width, new_height = round(width * ratio), round(height * ratio)
    left = round((size - new_width) / 2 - 0.1)
    top = round((size - new_height) / 2 - 0.1)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, top, size - new_height - top, left, size - new_width - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, ratio, (left, top)


def centres_in_roi(
    label: Path, source_shape: tuple[int, int], size: int, ratio: float,
    padding: tuple[int, int], roi: tuple[float, ...],
) -> dict[int, int]:
    """Count boxes per class whose centre lands inside the ROI after letterboxing."""
    height, width = source_shape
    counts: dict[int, int] = {0: 0, 1: 0}
    for line in label.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x_centre, y_centre = int(line.split()[0]), *map(float, line.split()[1:3])
        x = (x_centre * width * ratio + padding[0]) / size
        y = (y_centre * height * ratio + padding[1]) / size
        if roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]:
            counts[class_id] = counts.get(class_id, 0) + 1
    return counts


def classify(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    """Split the source images into frames that are safe and frames that violate."""
    roi = tuple(args.roi)
    safe: list[Path] = []
    violating: list[Path] = []
    for image_path in sorted(args.images.iterdir()):
        label = args.labels / f"{image_path.stem}.txt"
        if not label.is_file():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        _, ratio, padding = letterbox(image, args.size)
        counts = centres_in_roi(
            label, image.shape[:2], args.size, ratio, padding, roi
        )
        if counts.get(1):
            violating.append(image_path)
        elif counts.get(0):
            safe.append(image_path)
    return safe, violating


def main() -> int:
    args = parse_args()
    if not 1 <= args.fps <= 60:
        raise SystemExit("--fps must be between 1 and 60")
    if args.hold_seconds <= 0 or args.segment_seconds <= 0:
        raise SystemExit("--hold-seconds and --segment-seconds must be positive")
    roi = tuple(args.roi)
    if roi[0] >= roi[2] or roi[1] >= roi[3] or any(not 0 <= v <= 1 for v in roi):
        raise SystemExit("--roi must be normalised with positive width and height")
    for directory in (args.images, args.labels):
        if not directory.is_dir():
            raise SystemExit(f"Not a directory: {directory}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite to replace it")

    print(f"Scanning {args.images}", flush=True)
    safe, violating = classify(args)
    print(f"  frames safe inside the ROI      : {len(safe)}")
    print(f"  frames violating inside the ROI : {len(violating)}")
    if not safe or not violating:
        raise SystemExit("Need both safe and violating frames; check --roi and the split")

    rng = random.Random(args.seed)
    rng.shuffle(safe)
    rng.shuffle(violating)
    per_image = max(1, round(args.hold_seconds * args.fps))
    safe_images = max(1, round(args.segment_seconds / args.hold_seconds))
    violation_images = max(1, round(args.segment_seconds * 1.5 / args.hold_seconds))

    def take(pool: list[Path], count: int) -> list[Path]:
        return [pool[index % len(pool)] for index in range(count)]

    segments = [
        ("safe", take(safe, safe_images)),
        ("violation", take(violating, violation_images)),
        ("safe", take(safe[safe_images:] or safe, safe_images)),
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.size, args.size)
    )
    if not writer.isOpened():
        raise SystemExit(f"Cannot open {args.output} for writing")
    timeline = []
    frame_index = 0
    try:
        for kind, images in segments:
            start = frame_index
            for image_path in images:
                image = cv2.imread(str(image_path))
                padded, _, _ = letterbox(image, args.size)
                for _ in range(per_image):
                    writer.write(padded)
                    frame_index += 1
            timeline.append({
                "segment": kind,
                "first_frame": start,
                "last_frame": frame_index - 1,
                "seconds": round((frame_index - start) / args.fps, 3),
                "images": [p.name for p in images],
            })
    finally:
        writer.release()

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    provenance = {
        "clip": args.output.name,
        "sha256": digest,
        "bytes": args.output.stat().st_size,
        "fps": args.fps,
        "size": args.size,
        "frames": frame_index,
        "seconds": round(frame_index / args.fps, 3),
        "roi": list(roi),
        "seed": args.seed,
        "images_dir": args.images.as_posix(),
        "labels_dir": args.labels.as_posix(),
        "hold_seconds": args.hold_seconds,
        "pool": {"safe": len(safe), "violating": len(violating)},
        "timeline": timeline,
    }
    report = args.output.with_suffix(".json")
    report.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print(f"clip     : {args.output}")
    print(f"sha256   : {digest}")
    print(f"length   : {frame_index} frames, {frame_index / args.fps:.1f} s at {args.fps} fps")
    for entry in timeline:
        print(f"  {entry['segment']:9} frames {entry['first_frame']:>4}-{entry['last_frame']:<4} "
              f"{entry['seconds']:>5.1f} s")
    print(f"details  : {report}")
    print()
    print("Expected states while the clip plays, with the shipped configs/firefly.json:")
    print("  safe segment      -> SAFE")
    print("  violation segment -> PENDING, then ALARM after alarm_on_seconds")
    print("  safe segment      -> ALARM held for alarm_off_seconds, then SAFE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
