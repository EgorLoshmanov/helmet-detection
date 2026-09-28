#!/usr/bin/env python3
"""Export the released FORTNITEBALLS detector to a static ONNX model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


DEFAULT_MODEL = Path("models/helmet_detector_v0.2.0.pt")
DEFAULT_OUTPUT = Path("models/helmet_detector.onnx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing output file."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(
            f"Model not found: {args.model}\n"
            "Download it with: python scripts/download_model.py "
            "--version v0.2.0 --output models/helmet_detector_v0.2.0.pt"
        )
    if args.output.exists() and not args.force:
        raise SystemExit(f"Output already exists: {args.output}; use --force to replace it")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            batch=1,
            dynamic=False,
            simplify=True,
            opset=args.opset,
            nms=False,
            device="cpu",
        )
    )
    if exported.resolve() != args.output.resolve():
        os.replace(exported, args.output)

    print(f"ONNX model is ready: {args.output}")
    print(f"Input shape: 1x3x{args.imgsz}x{args.imgsz}")
    print(f"Opset: {args.opset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
