#!/usr/bin/env python3
"""Run an RKNN model on saved images using the Firefly NPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized_width, resized_height = round(width * ratio), round(height * ratio)
    pad_width = (size - resized_width) / 2
    pad_height = (size - resized_height) / 2
    if (width, height) != (resized_width, resized_height):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(
        image,
        round(pad_height - 0.1),
        round(pad_height + 0.1),
        round(pad_width - 0.1),
        round(pad_width + 0.1),
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )


def main() -> int:
    args = parse_args()
    if args.repeats < 1 or args.warmup < 0:
        raise SystemExit("--repeats must be positive and --warmup nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = RKNNLite()
    try:
        result = runtime.load_rknn(str(args.model))
        if result != 0:
            raise RuntimeError(f"load_rknn failed: {result}")
        result = runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if result != 0:
            raise RuntimeError(f"init_runtime failed: {result}")

        records = []
        for path in args.images:
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise RuntimeError(f"Cannot decode: {path}")
            padded = letterbox(bgr, args.imgsz)
            if padded.shape != (args.imgsz, args.imgsz, 3):
                raise RuntimeError(f"Unexpected letterbox shape: {padded.shape}")
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)[None]
            for _ in range(args.warmup):
                runtime.inference(inputs=[rgb], data_format=["nhwc"])
            timings = []
            outputs = None
            for _ in range(args.repeats):
                start = time.perf_counter_ns()
                outputs = runtime.inference(inputs=[rgb], data_format=["nhwc"])
                timings.append((time.perf_counter_ns() - start) / 1_000_000)
                if not outputs:
                    raise RuntimeError(f"NPU returned no outputs for {path}")
            arrays = {f"output_{index}": np.asarray(value) for index, value in enumerate(outputs)}
            output_path = args.output_dir / f"{path.stem}.npz"
            np.savez_compressed(output_path, **arrays)
            record = {
                "image": path.name,
                "image_sha256": sha256(path),
                "output": output_path.name,
                "output_tensors": [
                    {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in arrays.items()
                ],
                "latency_ms_median": statistics.median(timings),
                "latency_ms_min": min(timings),
                "latency_ms_max": max(timings),
            }
            records.append(record)
            print(f"{path.name}: {record['output_tensors']}; median {record['latency_ms_median']:.2f} ms", flush=True)
    finally:
        runtime.release()

    report = {
        "model_sha256": sha256(args.model),
        "input": "RGB uint8 NHWC; letterbox 640x640; model mean=0 std=255",
        "npu_core": 0,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "images": records,
    }
    (args.output_dir / "inference.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
