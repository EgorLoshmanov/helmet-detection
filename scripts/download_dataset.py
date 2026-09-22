#!/usr/bin/env python3
"""Download, verify, and extract the public Hard Hat Workers dataset."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "andrewmvd/hard-hat-detection"
)
EXPECTED_SHA256 = "aa5c80a85f9f4bd3b27e44256f8e36f9a32c53ee423132fa6cd5ea603781be62"
DEFAULT_ARCHIVE = Path("dataset/source/hard-hat-detection-v1.zip")
DEFAULT_OUTPUT = Path("dataset/source/hard-hat-detection-v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace a mismatched archive or incomplete extracted directory.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the local archive without downloading or extracting it.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_complete_dataset(path: Path) -> bool:
    images = path / "images"
    annotations = path / "annotations"
    return (
        images.is_dir()
        and annotations.is_dir()
        and len(list(images.glob("*.png"))) == 5000
        and len(list(annotations.glob("*.xml"))) == 5000
    )


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "FORTNITEBALLS-dataset-downloader/1.0"},
    )
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open(
            "wb"
        ) as output:
            total = int(response.headers.get("Content-Length", "0"))
            copied = 0
            next_report = 10
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                copied += len(block)
                if total:
                    percent = copied * 100 // total
                    if percent >= next_report:
                        print(f"  {percent}%")
                        next_report += 10
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Dataset download failed: {exc}") from exc
    os.replace(temporary, destination)


def validate_members(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != destination and destination not in target.parents:
            raise SystemExit(f"Unsafe path in ZIP archive: {member.filename}")
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise SystemExit(f"Symbolic link is not allowed in ZIP: {member.filename}")


def extract(archive_path: Path, output: Path, force: bool) -> None:
    temporary = output.parent / f".{output.name}.extracting"
    if output.exists():
        if is_complete_dataset(output):
            print(f"Dataset is already prepared at {output}")
            return
        if not force:
            raise SystemExit(
                f"{output} exists but is incomplete; rerun with --force to replace it"
            )
        shutil.rmtree(output)
    if temporary.exists():
        if not force:
            raise SystemExit(
                f"Temporary directory {temporary} exists; rerun with --force"
            )
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            validate_members(archive, temporary)
            archive.extractall(temporary)
        if not is_complete_dataset(temporary):
            raise SystemExit(
                "Extracted archive does not contain 5000 PNG images and 5000 XML files"
            )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Dataset extracted to {output}")


def main() -> int:
    args = parse_args()
    if args.verify_only and not args.archive.is_file():
        raise SystemExit(f"Archive not found: {args.archive}")

    if not args.archive.is_file():
        download(DATASET_URL, args.archive)

    actual_sha256 = sha256_file(args.archive)
    if actual_sha256 != EXPECTED_SHA256:
        if args.force and not args.verify_only:
            print("Archive checksum mismatch; downloading a clean copy", file=sys.stderr)
            args.archive.unlink()
            download(DATASET_URL, args.archive)
            actual_sha256 = sha256_file(args.archive)
        if actual_sha256 != EXPECTED_SHA256:
            raise SystemExit(
                "Archive SHA-256 mismatch:\n"
                f"  expected: {EXPECTED_SHA256}\n"
                f"  actual:   {actual_sha256}"
            )

    print(f"SHA-256 verified: {actual_sha256}")
    if not args.verify_only:
        extract(args.archive, args.output, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

