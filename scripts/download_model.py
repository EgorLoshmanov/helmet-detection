#!/usr/bin/env python3
"""Download and verify the published FORTNITEBALLS baseline model."""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path


MODEL_URL = (
    "https://github.com/EgorLoshmanov/helmet-detection/releases/download/"
    "v0.1.0/helmet_detector_best.pt"
)
EXPECTED_SHA256 = "96c709bf5fdcdb7a3b2a5c18aa746218d3175aa58fc4c0983285ff66e4e25b58"
DEFAULT_OUTPUT = Path("models/helmet_detector_best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing invalid model file."
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "FORTNITEBALLS-model-downloader/1.0"},
    )
    print(f"Downloading {MODEL_URL}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open(
            "wb"
        ) as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Model download failed: {exc}") from exc
    os.replace(temporary, destination)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        actual = sha256_file(args.output)
        if actual == EXPECTED_SHA256:
            print(f"Model is ready: {args.output}")
            print(f"SHA-256 verified: {actual}")
            return 0
        if not args.force:
            raise SystemExit(
                f"Existing model has an unexpected SHA-256: {actual}; "
                "rerun with --force to replace it"
            )
        args.output.unlink()

    download(args.output)
    actual = sha256_file(args.output)
    if actual != EXPECTED_SHA256:
        args.output.unlink(missing_ok=True)
        raise SystemExit(
            "Downloaded model SHA-256 mismatch:\n"
            f"  expected: {EXPECTED_SHA256}\n"
            f"  actual:   {actual}"
        )
    print(f"Model is ready: {args.output}")
    print(f"SHA-256 verified: {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

