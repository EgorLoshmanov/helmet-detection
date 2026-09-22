#!/usr/bin/env python3
"""Run the PyTorch helmet detector on an image, video, or local camera."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path, default=Path("models/helmet_detector_best.pt")
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Image/video path, stream URL, or numeric camera index (default: 0).",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or CUDA index")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def select_device(requested: str) -> str | int:
    normalized = requested.strip().lower()
    if normalized != "auto":
        if normalized == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested but is unavailable")
        if normalized in {"cuda", "gpu"}:
            if not torch.cuda.is_available():
                raise SystemExit("CUDA was requested but is unavailable")
            return 0
        return int(normalized) if normalized.isdigit() else normalized
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(
            f"Model not found: {args.model}. Run python scripts/download_model.py"
        )
    if not 0.0 < args.conf <= 1.0:
        raise SystemExit("--conf must be in the interval (0, 1]")
    source: str | int = int(args.source) if args.source.isdigit() else args.source
    device = select_device(args.device)
    print(f"device={device}")
    model = YOLO(args.model)
    model.predict(
        source=source,
        device=device,
        conf=args.conf,
        show=args.show,
        save=args.save,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

