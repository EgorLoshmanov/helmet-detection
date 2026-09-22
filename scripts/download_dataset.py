#!/usr/bin/env python3
"""Download and validate Hard Hat Workers through the official Kaggle client."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub


DATASET_HANDLE = "andrewmvd/hard-hat-detection/versions/1"
DEFAULT_OUTPUT = Path("dataset/source/hard-hat-detection-v1")
EXPECTED_IMAGES = 5000
EXPECTED_ANNOTATIONS = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an incomplete or existing downloaded directory.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate local files without contacting Kaggle.",
    )
    return parser.parse_args()


def counts(path: Path) -> tuple[int, int]:
    images = len(list((path / "images").glob("*.png")))
    annotations = len(list((path / "annotations").glob("*.xml")))
    return images, annotations


def validate(path: Path) -> None:
    image_count, annotation_count = counts(path)
    if (image_count, annotation_count) != (
        EXPECTED_IMAGES,
        EXPECTED_ANNOTATIONS,
    ):
        raise SystemExit(
            f"Dataset validation failed at {path}: expected "
            f"{EXPECTED_IMAGES} PNG and {EXPECTED_ANNOTATIONS} XML files, got "
            f"{image_count} PNG and {annotation_count} XML files"
        )
    print(
        f"Dataset is ready: {path} "
        f"({image_count} PNG, {annotation_count} XML)"
    )


def main() -> int:
    args = parse_args()
    if args.verify_only:
        validate(args.output)
        return 0

    image_count, annotation_count = counts(args.output)
    if (image_count, annotation_count) == (
        EXPECTED_IMAGES,
        EXPECTED_ANNOTATIONS,
    ) and not args.force:
        validate(args.output)
        return 0

    if args.output.exists() and not args.force:
        raise SystemExit(
            f"{args.output} exists but is incomplete; rerun with --force to replace it"
        )
    if args.output.exists():
        shutil.rmtree(args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kaggle dataset {DATASET_HANDLE}")
    downloaded = Path(
        kagglehub.dataset_download(
            DATASET_HANDLE,
            output_dir=str(args.output),
            force_download=args.force,
        )
    )
    validate(downloaded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

