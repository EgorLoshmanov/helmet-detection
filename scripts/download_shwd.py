#!/usr/bin/env python3
"""Download and validate the Safety Helmet Wearing Dataset (SHWD)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _datasets import (  # noqa: E402
    find_voc_root,
    google_drive_download,
    human,
    report_layout,
    safe_extract,
    sha256_file,
    validate_counts,
    write_provenance,
)


DRIVE_FILE_ID = "1qWm7rrwvjAWs1slymbrLaCf7Q-wnGLEX"
ARCHIVE_NAME = "VOC2028.zip"
EXPECTED_BYTES = 1117970537
EXPECTED_IMAGES = 7581
EXPECTED_ANNOTATIONS = 7581
DEFAULT_OUTPUT = Path("dataset/source/shwd")
SOURCE_URL = "https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset"
LICENSE = (
    "repository MIT (c) 2019 njvisionpower; the negative objects derive from "
    "SCUT-HEAD, which is free for academic research use only"
)
CITATION = "Peng et al., arXiv:1803.09256, 2018 (SCUT-HEAD)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch SHWD from Google Drive, verify it and print the class names "
            "needed for dataset/sources.yaml."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an already downloaded VOC2028.zip instead of contacting Drive.",
    )
    parser.add_argument(
        "--expect-sha256",
        help="Fail unless the archive matches this checksum.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded zip after extraction.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate what is already on disk without downloading.",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Warn instead of failing when the file counts differ.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing extracted directory.",
    )
    return parser.parse_args()


def audit(output: Path, args: argparse.Namespace) -> dict:
    voc_root, images, annotations = find_voc_root(output)
    summary = report_layout("SHWD", voc_root, images, annotations)
    validate_counts(
        "SHWD images", summary["images"], EXPECTED_IMAGES, args.allow_count_mismatch
    )
    validate_counts(
        "SHWD annotations",
        summary["annotations"],
        EXPECTED_ANNOTATIONS,
        args.allow_count_mismatch,
    )
    return summary


def main() -> int:
    args = parse_args()
    output: Path = args.output

    if args.verify_only:
        if not output.is_dir():
            raise SystemExit(f"Nothing to verify: {output} does not exist")
        audit(output, args)
        print(f"SHWD is ready: {output}")
        return 0

    if output.exists() and any(output.iterdir()):
        if not args.force:
            if args.expect_sha256:
                raise SystemExit(
                    f"{output} already holds extracted data, so no archive is "
                    "fetched and --expect-sha256 cannot be checked. Rerun with "
                    "--force to download and verify it again."
                )
            print(f"{output} already exists; validating instead of downloading")
            audit(output, args)
            print(f"SHWD is ready: {output}")
            print("Rerun with --force to download it again.")
            return 0
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=True)
    archive = args.archive
    downloaded = False
    if archive is None:
        archive = output / ARCHIVE_NAME
        print(
            f"SHWD is about {human(EXPECTED_BYTES)}; the download takes a while.",
            file=sys.stderr,
        )
        google_drive_download(DRIVE_FILE_ID, archive)
        downloaded = True
    elif not archive.is_file():
        raise SystemExit(f"Archive not found: {archive}")

    size = archive.stat().st_size
    checksum = sha256_file(archive)
    print(f"archive : {archive}")
    print(f"size    : {size} bytes ({human(size)})")
    print(f"sha256  : {checksum}")
    if downloaded and size != EXPECTED_BYTES:
        print(
            f"WARNING expected {EXPECTED_BYTES} bytes from Drive, got {size}. "
            "The upstream file may have been replaced.",
            file=sys.stderr,
        )
    if args.expect_sha256 and checksum != args.expect_sha256:
        raise SystemExit(
            "Archive checksum mismatch:\n"
            f"  expected: {args.expect_sha256}\n"
            f"  actual:   {checksum}"
        )

    print(f"Extracting {archive.name}", file=sys.stderr)
    safe_extract(archive, output)
    if downloaded and not args.keep_archive:
        archive.unlink(missing_ok=True)

    summary = audit(output, args)
    provenance = write_provenance(
        output,
        {
            "dataset": "SHWD (Safety Helmet Wearing Dataset)",
            "url": SOURCE_URL,
            "drive_file_id": DRIVE_FILE_ID,
            "archive_name": ARCHIVE_NAME,
            "archive_bytes": size,
            "archive_sha256": checksum,
            "license": LICENSE,
            "citation": CITATION,
            "contents": summary,
        },
    )
    print(f"provenance written to {provenance}")
    print(f"SHWD is ready: {output}")
    print(
        "Record the checksum above in dataset/README.md, then enable the shwd "
        "source in dataset/sources.yaml."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
