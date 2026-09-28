#!/usr/bin/env python3
"""Convert the verified YOLOv8 ONNX model to RK3588 RKNN."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import onnx
from rknn.api import RKNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=Path("models/helmet_detector.onnx"))
    parser.add_argument("--mode", choices=("fp16", "int8"), default="fp16")
    parser.add_argument("--dataset", type=Path, help="Calibration image list, required for INT8")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true", help="Replace an existing RKNN file")
    parser.add_argument("--verbose", action="store_true", help="Print detailed Toolkit graph diagnostics")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_success(operation: str, result: int) -> None:
    if result != 0:
        raise RuntimeError(f"RKNN {operation} failed with code {result}")


def main() -> int:
    args = parse_args()
    output = args.output or Path(f"models/helmet_detector_{args.mode}.rknn")
    if not args.onnx.is_file():
        raise SystemExit(f"ONNX file is missing: {args.onnx}")
    if args.mode == "int8" and (args.dataset is None or not args.dataset.is_file()):
        raise SystemExit("INT8 requires --dataset with a calibration image list")
    if args.mode == "int8":
        outputs = onnx.load(args.onnx).graph.output
        shapes = [[dimension.dim_value for dimension in output.type.tensor_type.shape.dim] for output in outputs]
        if shapes != [[1, 4, 8400], [1, 2, 8400]]:
            raise SystemExit("INT8 requires separate box and class-score outputs; run split_onnx_outputs.py")
    if args.mode == "fp16" and args.dataset is not None:
        raise SystemExit("--dataset is only used with --mode int8")
    if output.exists() and not args.force:
        raise SystemExit(f"Output exists: {output}; pass --force to replace it")

    print(f"ONNX: {args.onnx} (SHA-256 {sha256(args.onnx)})", flush=True)
    print(f"Mode: {args.mode}; target_platform: rk3588", flush=True)
    print("Input contract: RGB uint8 NHWC; mean=0, std=255", flush=True)
    converter = RKNN(verbose=args.verbose)
    try:
        require_success(
            "config",
            converter.config(
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                target_platform="rk3588",
            ),
        )
        require_success("load_onnx", converter.load_onnx(model=str(args.onnx)))
        require_success(
            "build",
            converter.build(
                do_quantization=args.mode == "int8",
                dataset=str(args.dataset) if args.dataset else None,
            ),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        require_success("export_rknn", converter.export_rknn(str(output)))
    finally:
        converter.release()
    print(f"RKNN: {output} (SHA-256 {sha256(output)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
