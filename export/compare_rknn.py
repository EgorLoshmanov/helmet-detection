#!/usr/bin/env python3
"""Compare saved Firefly RKNN outputs against ONNX on identical images."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

from validate_onnx import compare_detections, preprocess, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=Path("models/helmet_detector.onnx"))
    parser.add_argument("--rknn", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True, help="Directory copied from Firefly")
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/processed/helmet_yolo_v1/images/test"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-box-delta-px", type=float, default=3.0)
    parser.add_argument("--max-conf-delta", type=float, default=0.03)
    parser.add_argument("--min-match-iou", type=float, default=0.9)
    return parser.parse_args()


def decoded_boxes(raw: np.ndarray, original_shape: tuple[int, int], args: argparse.Namespace) -> np.ndarray:
    if raw.shape != (1, 6, 8400):
        raise ValueError(f"Unexpected YOLO output shape: {raw.shape}")
    boxes = non_max_suppression(
        torch.from_numpy(raw.astype(np.float32, copy=False)),
        conf_thres=args.conf,
        iou_thres=args.iou,
        nc=2,
        max_det=300,
    )[0]
    boxes[:, :4] = scale_boxes((args.imgsz, args.imgsz), boxes[:, :4], original_shape)
    return boxes.numpy()


def main() -> int:
    args = parse_args()
    remote = json.loads((args.results / "inference.json").read_text())
    if remote["model_sha256"] != sha256(args.rknn):
        raise SystemExit("Remote RKNN model SHA-256 differs from local model")
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    records = []
    for record in remote["images"]:
        image = args.images_dir / record["image"]
        if sha256(image) != record["image_sha256"]:
            raise SystemExit(f"Image SHA-256 differs from remote copy: {image}")
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise SystemExit(f"Cannot decode image: {image}")
        input_tensor = preprocess(bgr, args.imgsz)
        onnx_raw = session.run(None, {input_name: input_tensor})[0]
        saved = np.load(args.results / record["output"])
        if set(saved.files) == {"output_0"}:
            rknn_raw = saved["output_0"]
        elif set(saved.files) == {"output_0", "output_1"}:
            boxes, scores = saved["output_0"], saved["output_1"]
            if boxes.shape != (1, 4, 8400) or scores.shape != (1, 2, 8400):
                raise SystemExit(f"Unexpected split RKNN output shapes: {boxes.shape}, {scores.shape}")
            rknn_raw = np.concatenate((boxes, scores), axis=1)
        else:
            raise SystemExit(f"Expected one or two RKNN outputs for {image}, got {saved.files}")
        if onnx_raw.shape != rknn_raw.shape:
            raise SystemExit(f"Output shapes differ for {image}: {onnx_raw.shape} vs {rknn_raw.shape}")
        onnx_boxes = decoded_boxes(onnx_raw, bgr.shape[:2], args)
        rknn_boxes = decoded_boxes(rknn_raw, bgr.shape[:2], args)
        match = compare_detections(
            onnx_boxes,
            rknn_boxes,
            max_box_delta_px=args.max_box_delta_px,
            max_conf_delta=args.max_conf_delta,
            min_match_iou=args.min_match_iou,
        )
        difference = np.abs(onnx_raw - rknn_raw)
        item = {
            "image": record["image"],
            "onnx_detections": len(onnx_boxes),
            "rknn_detections": len(rknn_boxes),
            "raw_mean_abs_delta": float(difference.mean()),
            "raw_max_abs_delta": float(difference.max()),
            "detections": match,
            "latency_ms_median": record["latency_ms_median"],
        }
        records.append(item)
        print(
            f"{'PASS' if match['passed'] else 'FAIL'} {image.name}: "
            f"ONNX={len(onnx_boxes)}, RKNN={len(rknn_boxes)}, "
            f"matched={match['matched_count']}, max_box_delta={match['max_box_delta_px']}"
        )

    report = {
        "purpose": "ONNX vs RKNN parity on saved images; not an independent quality evaluation",
        "onnx_sha256": sha256(args.onnx),
        "rknn_sha256": sha256(args.rknn),
        "input_contract": remote["input"],
        "runtime_settings": {"npu_core": remote["npu_core"], "warmup": remote["warmup"], "repeats": remote["repeats"]},
        "comparison_settings": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_box_delta_px": args.max_box_delta_px,
            "max_conf_delta": args.max_conf_delta,
            "min_match_iou": args.min_match_iou,
        },
        "summary": {
            "images": len(records),
            "passed_images": sum(item["detections"]["passed"] for item in records),
            "onnx_detections": sum(item["onnx_detections"] for item in records),
            "rknn_detections": sum(item["rknn_detections"] for item in records),
            "median_npu_call_ms": statistics.median(item["latency_ms_median"] for item in records),
            "passed": all(item["detections"]["passed"] for item in records),
        },
        "images": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Report: {args.report}; {report['summary']}")
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
