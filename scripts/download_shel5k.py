#!/usr/bin/env python3
"""Unpack and validate SHEL5K, the re-annotation of the hard-hat 5000 images."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _datasets import (  # noqa: E402
    download,
    find_voc_root,
    human,
    report_layout,
    safe_extract,
    sha256_file,
    validate_counts,
    write_provenance,
)


LANDING_PAGE = "https://data.mendeley.com/datasets/9rcv8mm682"
DOI = "10.17632/9rcv8mm682.4"
EXPECTED_IMAGES = 5000
EXPECTED_ANNOTATIONS = 5000
DEFAULT_OUTPUT = Path("dataset/source/shel5k")
LICENSE = "CC BY 4.0"
CITATION = (
    "Otgonbold et al., SHEL5K: An Extended Dataset and Benchmarking for Safety "
    "Helmet Detection. Sensors 22(6):2315, 2022. doi:10.3390/s22062315"
)

INSTRUCTIONS = f"""\
SHEL5K is hosted on Mendeley Data, which serves its files through a browser
session rather than a stable download API, so this step cannot be automated
safely. Fetch the archive once by hand:

  1. Open {LANDING_PAGE}
  2. Press "Download All" (about 5000 images with their annotations).
  3. Save the archive, then rerun this script pointing at it:

       python scripts/download_shel5k.py --archive path/to/downloaded.zip

If you have a direct link instead, pass it with --url. Everything after the
download - checksum, safe extraction, structure detection, count checks and the
class-name inventory - is handled here.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and unpack a SHEL5K archive, then print the class names "
            "needed for dataset/sources.yaml."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path, help="Path to the downloaded archive.")
    parser.add_argument("--url", help="Direct download URL, if you have one.")
    parser.add_argument(
        "--expect-sha256",
        help="Fail unless the archive matches this checksum.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the archive after extraction when it was downloaded here.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate what is already on disk without unpacking anything.",
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
    summary = report_layout("SHEL5K", voc_root, images, annotations)
    validate_counts(
        "SHEL5K images", summary["images"], EXPECTED_IMAGES, args.allow_count_mismatch
    )
    validate_counts(
        "SHEL5K annotations",
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
        print(f"SHEL5K is ready: {output}")
        return 0

    if args.archive is None and args.url is None:
        print(INSTRUCTIONS)
        return 1

    if output.exists() and any(output.iterdir()):
        if not args.force:
            if args.expect_sha256:
                raise SystemExit(
                    f"{output} already holds extracted data, so no archive is "
                    "unpacked and --expect-sha256 cannot be checked. Rerun with "
                    "--force to unpack and verify it again."
                )
            print(f"{output} already exists; validating instead of unpacking")
            audit(output, args)
            print(f"SHEL5K is ready: {output}")
            print("Rerun with --force to replace it.")
            return 0
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=True)
    downloaded = False
    if args.archive is not None:
        archive = args.archive
        if not archive.is_file():
            raise SystemExit(f"Archive not found: {archive}")
    else:
        archive = output / "shel5k_download"
        download(args.url, archive)
        downloaded = True

    size = archive.stat().st_size
    checksum = sha256_file(archive)
    print(f"archive : {archive}")
    print(f"size    : {size} bytes ({human(size)})")
    print(f"sha256  : {checksum}")
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
            "dataset": "SHEL5K (Safety HELmet dataset with 5K images)",
            "url": LANDING_PAGE,
            "doi": DOI,
            "archive_name": archive.name,
            "archive_bytes": size,
            "archive_sha256": checksum,
            "license": LICENSE,
            "citation": CITATION,
            "contents": summary,
        },
    )
    print(f"provenance written to {provenance}")
    print(f"SHEL5K is ready: {output}")
    print(
        "Check the class names and paths printed above against the shel5k entry "
        "in dataset/sources.yaml; an undeclared class name aborts the build."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
