"""Shared download, extraction and audit helpers for the dataset fetchers."""

from __future__ import annotations

import hashlib
import http.cookiejar
import json
import os
import re
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

USER_AGENT = "FORTNITEBALLS-dataset-downloader/1.0"
BLOCK = 1024 * 1024
HIDDEN_INPUT = re.compile(r'name="([^"]+)"\s+value="([^"]*)"')


def human(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GiB"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _stream(response, destination: Path) -> Path:
    total = response.headers.get("Content-Length")
    expected = int(total) if total and total.isdigit() else None
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    written = 0
    milestone = 0
    try:
        with temporary.open("wb") as output:
            while True:
                block = response.read(BLOCK)
                if not block:
                    break
                output.write(block)
                written += len(block)
                if written - milestone >= 32 * BLOCK:
                    milestone = written
                    if expected:
                        share = 100.0 * written / expected
                        print(
                            f"  {human(written)} / {human(expected)} ({share:.0f}%)",
                            file=sys.stderr,
                        )
                    else:
                        print(f"  {human(written)}", file=sys.stderr)
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Download failed: {exc}") from exc
    if expected is not None and written != expected:
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            f"Download truncated: expected {expected} bytes, received {written}"
        )
    os.replace(temporary, destination)
    return destination


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"Downloading {url}", file=sys.stderr)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return _stream(response, destination)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Download failed: HTTP {exc.code} for {url}") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise SystemExit(f"Download failed: {exc}") from exc


def google_drive_download(file_id: str, destination: Path) -> Path:
    """Fetch a large Drive file, clearing the interstitial virus-scan warning."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", USER_AGENT)]
    base = "https://drive.usercontent.google.com/download"
    url = f"{base}?id={urllib.parse.quote(file_id)}&export=download"
    print(f"Requesting Google Drive file {file_id}", file=sys.stderr)
    try:
        response = opener.open(url, timeout=60)
        if "text/html" in response.headers.get("Content-Type", ""):
            body = response.read(4 * BLOCK).decode("utf-8", "replace")
            fields = dict(HIDDEN_INPUT.findall(body))
            if "confirm" not in fields:
                raise SystemExit(
                    "Google Drive did not return a download form. The file may be "
                    "private, removed, or temporarily over its download quota."
                )
            fields["id"] = file_id
            response = opener.open(
                f"{base}?{urllib.parse.urlencode(fields)}", timeout=60
            )
        disposition = response.headers.get("Content-Disposition", "")
        match = re.search(r'filename="([^"]+)"', disposition)
        if match:
            print(f"  remote name: {match.group(1)}", file=sys.stderr)
        return _stream(response, destination)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Download failed: HTTP {exc.code}") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise SystemExit(f"Download failed: {exc}") from exc


def safe_extract(archive: Path, destination: Path) -> str:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                target = (root / name).resolve()
                if target != root and root not in target.parents:
                    raise SystemExit(
                        f"Refusing to extract outside {destination}: {name}"
                    )
            bundle.extractall(root)
        return "zip"
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as bundle:
            bundle.extractall(root, filter="data")
        return "tar"
    raise SystemExit(
        f"Unsupported archive format: {archive.name}. Only zip and tar are handled; "
        "unpack it manually and pass the resulting directory."
    )


def find_voc_root(start: Path) -> tuple[Path, Path, Path]:
    """Locate a Pascal VOC layout: a directory holding annotations and images.

    Directory names are matched case insensitively but reported exactly as they
    are on disk, because Windows would accept a wrongly cased path here while
    Linux would not.
    """
    image_names = ("jpegimages", "images")
    candidates = [start, *(p for p in start.rglob("*") if p.is_dir())]
    for candidate in candidates:
        try:
            entries = {p.name.lower(): p for p in candidate.iterdir() if p.is_dir()}
        except OSError:
            continue
        annotations = entries.get("annotations")
        images = next((entries[name] for name in image_names if name in entries), None)
        if annotations is not None and images is not None:
            return candidate, images, annotations
    raise SystemExit(
        f"No Pascal VOC layout found under {start}. Expected a directory containing "
        "both an annotations folder and an images folder."
    )


def voc_inventory(annotations: Path) -> tuple[int, Counter]:
    counts: Counter = Counter()
    files = sorted(annotations.glob("*.xml"))
    for path in files:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            counts["<unparsable file>"] += 1
            continue
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            counts[name or "<missing name>"] += 1
    return len(files), counts


def count_images(images: Path) -> tuple[int, Counter]:
    suffixes: Counter = Counter()
    total = 0
    for path in images.iterdir():
        if path.is_file():
            suffixes[path.suffix.lower()] += 1
            total += 1
    return total, suffixes


def report_layout(name: str, voc_root: Path, images: Path, annotations: Path) -> dict:
    image_total, suffixes = count_images(images)
    annotation_total, classes = voc_inventory(annotations)
    print()
    print(f"{name} layout")
    print(f"  root        : {voc_root}")
    print(f"  images      : {image_total} files {dict(suffixes)}")
    print(f"  annotations : {annotation_total} xml files")
    print("  class names found in the annotations:")
    for label, count in sorted(classes.items(), key=lambda item: -item[1]):
        print(f"    {label!r}: {count}")
    print()
    print("  paths for dataset/sources.yaml:")
    print(f"    images: {images.as_posix()}")
    print(f"    annotations: {annotations.as_posix()}")
    print("  class_map skeleton (decide each target yourself):")
    for label in sorted(classes):
        if label.startswith("<"):
            continue
        print(f"    {label}: null")
    return {
        "images": image_total,
        "image_suffixes": dict(suffixes),
        "annotations": annotation_total,
        "classes": dict(sorted(classes.items())),
    }


def validate_counts(
    label: str, actual: int, expected: int, allow_mismatch: bool
) -> None:
    if actual == expected:
        return
    message = f"{label}: expected {expected}, found {actual}"
    if allow_mismatch:
        print(f"WARNING {message}", file=sys.stderr)
        return
    raise SystemExit(
        f"{message}. The archive may be a different release than the one this "
        "script expects; rerun with --allow-count-mismatch to continue anyway."
    )


def write_provenance(target: Path, payload: dict) -> Path:
    payload = dict(payload)
    payload["recorded_at"] = datetime.now(timezone.utc).isoformat()
    path = target / "_provenance.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path
