#!/usr/bin/env python3
"""Download and verify a published FORTNITEBALLS model."""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path


RELEASES_URL = "https://github.com/EgorLoshmanov/helmet-detection/releases/download"

# Both releases stay reachable: comparing a new model against the previous one
# needs the previous weights, and they are not kept in Git.
RELEASES: dict[str, tuple[str, str]] = {
    # version: (asset name, sha256)
    "v0.1.0": (
        "helmet_detector_best.pt",
        "96c709bf5fdcdb7a3b2a5c18aa746218d3175aa58fc4c0983285ff66e4e25b58",
    ),
    "v0.2.0": (
        "helmet_detector_best.pt",
        "abc0764a859476d33757a8d2389e685d2db48bdd14bbdfbbff2f655d0c952a1b",
    ),
}
DEFAULT_VERSION = "v0.2.0"
DEFAULT_OUTPUT = Path("models/helmet_detector_best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        choices=sorted(RELEASES),
        help=f"Release to download (default: {DEFAULT_VERSION}).",
    )
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


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "FORTNITEBALLS-model-downloader/1.0"},
    )
    print(f"Downloading {url}")
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
    asset, expected = RELEASES[args.version]
    url = f"{RELEASES_URL}/{args.version}/{asset}"
    if args.output.exists():
        actual = sha256_file(args.output)
        if actual == expected:
            print(f"Model is ready: {args.output}")
            print(f"SHA-256 verified: {actual}")
            return 0
        if not args.force:
            raise SystemExit(
                f"Existing model has an unexpected SHA-256: {actual}; "
                "rerun with --force to replace it"
            )
        args.output.unlink()

    download(url, args.output)
    actual = sha256_file(args.output)
    if actual != expected:
        args.output.unlink(missing_ok=True)
        raise SystemExit(
            "Downloaded model SHA-256 mismatch:\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )
    print(f"Model is ready: {args.output} ({args.version})")
    print(f"SHA-256 verified: {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

