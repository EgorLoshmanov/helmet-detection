#!/usr/bin/env python3
"""Reproduce the experimental INT8/FP16 conversion for split-output YOLOv8."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rknn.api import RKNN


# Keep the coordinate path after distribution-focal-loss decoding in FP16.
# These names are specific to models/helmet_detector_split.onnx.
FP16_LAYERS = (
    "/model.22/dfl/Softmax_output_0_sw_sw",
    "/model.22/dfl/Transpose_1_output_0_sw",
    "/model.22/dfl/Reshape_1_output_0_rs",
    "/model.22/Slice_output_0-rs",
    "/model.22/Slice_1_output_0-rs",
    "/model.22/Sub_output_0-rs",
    "/model.22/Add_1_output_0-rs",
    "/model.22/Sub_1_output_0-rs",
    "/model.22/Add_2_output_0-rs",
    "/model.22/Div_1_output_0-rs",
    "/model.22/Concat_2_output_0-rs",
    "/model.22/Mul_2_output_0-rs",
)


def require_success(name: str, result: int) -> None:
    if result != 0:
        raise RuntimeError(f"{name} failed with code {result}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=Path("models/helmet_detector_split.onnx"))
    parser.add_argument("--dataset", type=Path, default=Path("export/calibration_dataset.txt"))
    parser.add_argument("--work-dir", type=Path, default=Path("dataset/processed/rknn_hybrid_build"))
    parser.add_argument("--output", type=Path, default=Path("models/helmet_detector_hybrid.rknn"))
    args = parser.parse_args()
    onnx_path, dataset_path, output_path = (path.resolve() for path in (args.onnx, args.dataset, args.output))
    work_dir = args.work_dir.resolve()
    if not onnx_path.is_file() or not dataset_path.is_file():
        raise SystemExit("Split ONNX and calibration manifest must exist")
    if output_path.exists():
        raise SystemExit(f"Output already exists: {output_path}")
    stem = onnx_path.stem
    if any((work_dir / f"{stem}.{extension}").exists() for extension in ("model", "data", "quantization.cfg")):
        raise SystemExit("Hybrid work directory already contains generated files; choose a fresh --work-dir")
    work_dir.mkdir(parents=True, exist_ok=True)

    original_dir = Path.cwd()
    try:
        os.chdir(work_dir)
        converter = RKNN()
        try:
            require_success("config", converter.config(
                mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588"
            ))
            require_success("load_onnx", converter.load_onnx(model=str(onnx_path)))
            require_success("hybrid step1", converter.hybrid_quantization_step1(
                dataset=str(dataset_path), proposal=False
            ))
        finally:
            converter.release()

        config_path = Path(f"{stem}.quantization.cfg")
        config_text = config_path.read_text()
        if not config_text.startswith("custom_quantize_layers: {}\n"):
            raise RuntimeError("Unexpected hybrid configuration layout; inspect it manually")
        override = "custom_quantize_layers:\n" + "".join(f"    {name}: float16\n" for name in FP16_LAYERS)
        for name in FP16_LAYERS:
            if f"    {name}:\n" not in config_text:
                raise RuntimeError(f"Hybrid graph no longer contains expected layer {name}")
        config_path.write_text(config_text.replace("custom_quantize_layers: {}\n", override, 1))

        converter = RKNN()
        try:
            require_success("hybrid step2", converter.hybrid_quantization_step2(
                model_input=f"{stem}.model",
                data_input=f"{stem}.data",
                model_quantization_cfg=str(config_path),
            ))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            require_success("export_rknn", converter.export_rknn(str(output_path)))
        finally:
            converter.release()
    finally:
        os.chdir(original_dir)
    print(f"Saved experimental hybrid model to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
