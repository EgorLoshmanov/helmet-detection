#!/usr/bin/env python3
"""Compare raw and final PyTorch/ONNX detections on the same real images."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox


DEFAULT_IMAGES = Path("dataset/processed/helmet_yolo_v1/images/test")
DEFAULT_LABELS = Path("dataset/processed/helmet_yolo_v1/labels/test")
DEFAULT_REPORT = Path("reports/onnx_comparison.json")
SEED = 20260919
RAW_ATOL = 1e-3
RAW_RTOL = 1e-4
MAX_BOX_DELTA_PX = 2.0
MAX_CONF_DELTA = 0.01
MIN_MATCH_IOU = 0.98


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt", type=Path, default=Path("models/helmet_detector_v0.2.0.pt"))
    parser.add_argument("--onnx", type=Path, default=Path("models/helmet_detector.onnx"))
    parser.add_argument("--images", type=Path, nargs="*", help="Explicit real images to compare.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annotation_group(label: Path) -> tuple[str, int]:
    lines = [line.split() for line in label.read_text().splitlines() if line.strip()]
    classes = {tokens[0] for tokens in lines}
    group = {
        frozenset({"0"}): "helmet_only",
        frozenset({"1"}): "no_helmet_only",
        frozenset({"0", "1"}): "both_classes",
        frozenset(): "empty",
    }.get(frozenset(classes), "other")
    return group, len(lines)


def select_images(images_dir: Path, labels_dir: Path, per_class: int) -> list[tuple[Path, str, int]]:
    if per_class < 1:
        raise SystemExit("--per-class must be at least 1")
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit("Image or label directory is missing; pass --images with real image paths")
    groups: dict[str, list[tuple[Path, str, int]]] = {
        "helmet_only": [],
        "no_helmet_only": [],
        "both_classes": [],
    }
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label = labels_dir / f"{image.stem}.txt"
        if not label.is_file():
            continue
        group, count = annotation_group(label)
        if group in groups:
            groups[group].append((image, group, count))
    rng = random.Random(SEED)
    selected: list[tuple[Path, str, int]] = []
    for name, candidates in groups.items():
        if len(candidates) < per_class:
            raise SystemExit(f"Not enough {name} images: {len(candidates)} < {per_class}")
        selected.extend(rng.sample(candidates, per_class))
    # Exercise the same postprocessing on a crowded real scene as well.
    densest = max((item for values in groups.values() for item in values), key=lambda item: item[2])
    if densest[0] not in {item[0] for item in selected}:
        selected.append(densest)
    return selected


def preprocess(image: np.ndarray, imgsz: int) -> np.ndarray:
    padded = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=image)
    rgb_chw = padded[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(rgb_chw[None], dtype=np.float32) / 255.0


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    area_second = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    denominator = area_first + area_second - intersection
    return intersection / denominator if denominator else 0.0


def compare_detections(pt_boxes: np.ndarray, onnx_boxes: np.ndarray) -> dict:
    remaining = set(range(len(onnx_boxes)))
    matched = 0
    max_box_delta = 0.0
    max_conf_delta = 0.0
    min_iou = 1.0
    for reference in pt_boxes:
        same_class = [index for index in remaining if int(onnx_boxes[index, 5]) == int(reference[5])]
        if not same_class:
            continue
        index = max(same_class, key=lambda candidate: box_iou(reference[:4], onnx_boxes[candidate, :4]))
        candidate = onnx_boxes[index]
        overlap = box_iou(reference[:4], candidate[:4])
        if overlap < MIN_MATCH_IOU:
            continue
        remaining.remove(index)
        matched += 1
        min_iou = min(min_iou, overlap)
        max_box_delta = max(max_box_delta, float(np.max(np.abs(reference[:4] - candidate[:4]))))
        max_conf_delta = max(max_conf_delta, float(abs(reference[4] - candidate[4])))
    passed = (
        matched == len(pt_boxes) == len(onnx_boxes)
        and max_box_delta <= MAX_BOX_DELTA_PX
        and max_conf_delta <= MAX_CONF_DELTA
    )
    return {
        "pt_count": len(pt_boxes),
        "onnx_count": len(onnx_boxes),
        "matched_count": matched,
        "min_matched_iou": min_iou if matched else None,
        "max_box_delta_px": max_box_delta if matched else None,
        "max_confidence_delta": max_conf_delta if matched else None,
        "passed": passed,
    }


def main() -> int:
    args = parse_args()
    if not 0 < args.conf <= 1 or not 0 < args.iou <= 1:
        raise SystemExit("--conf and --iou must be in (0, 1]")
    for model_path in (args.pt, args.onnx):
        if not model_path.is_file():
            raise SystemExit(f"Missing model: {model_path}")
    if args.images is not None:
        if not args.images:
            raise SystemExit("--images needs one or more image paths")
        selected = [(path, "unlabelled", 0) for path in args.images]
    else:
        selected = select_images(args.images_dir, args.labels_dir, args.per_class)
    for path, _, _ in selected:
        if not path.is_file():
            raise SystemExit(f"Missing image: {path}")

    pt = YOLO(str(args.pt))
    pt.model.eval()
    onnx = YOLO(str(args.onnx))
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    details = []
    for path, group, label_count in selected:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"Could not decode image: {path}")
        input_tensor = preprocess(image, args.imgsz)
        with torch.inference_mode():
            raw_pt = pt.model(torch.from_numpy(input_tensor))
        raw_pt = raw_pt[0] if isinstance(raw_pt, (list, tuple)) else raw_pt
        raw_pt = raw_pt.detach().cpu().numpy()
        raw_onnx = session.run(None, {input_name: input_tensor})[0]
        if raw_pt.shape != raw_onnx.shape:
            raise SystemExit(f"Raw output shape differs on {path}: {raw_pt.shape} vs {raw_onnx.shape}")
        raw_delta = np.abs(raw_pt - raw_onnx)
        raw_passed = bool(np.allclose(raw_pt, raw_onnx, rtol=RAW_RTOL, atol=RAW_ATOL))

        prediction_args = dict(imgsz=args.imgsz, conf=args.conf, iou=args.iou, rect=False, device="cpu", verbose=False)
        pt_result = pt.predict(source=str(path), **prediction_args)[0]
        onnx_result = onnx.predict(source=str(path), **prediction_args)[0]
        pt_boxes = pt_result.boxes.data.cpu().numpy()
        onnx_boxes = onnx_result.boxes.data.cpu().numpy()
        detections = compare_detections(pt_boxes, onnx_boxes)
        result = {
            "image": str(path),
            "annotation_group": group,
            "annotation_boxes": label_count,
            "raw_shape": list(raw_pt.shape),
            "raw_allclose": raw_passed,
            "raw_max_abs_delta": float(raw_delta.max()),
            "raw_mean_abs_delta": float(raw_delta.mean()),
            "detections": detections,
            "passed": raw_passed and detections["passed"],
        }
        details.append(result)
        print(f"{'PASS' if result['passed'] else 'FAIL'} {path.name}: "
              f"raw max={result['raw_max_abs_delta']:.6f}, "
              f"detections={detections['pt_count']}/{detections['onnx_count']}")

    passed = all(item["passed"] for item in details)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Numerical and detection parity; not an independent quality evaluation of v2",
        "dataset_note": "Default v1 test images may overlap v2 training images",
        "model": {"pt": str(args.pt), "pt_sha256": sha256(args.pt),
                  "onnx": str(args.onnx), "onnx_sha256": sha256(args.onnx)},
        "versions": {"torch": torch.__version__, "onnxruntime": ort.__version__},
        "settings": {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou,
                     "raw_atol": RAW_ATOL, "raw_rtol": RAW_RTOL,
                     "max_box_delta_px": MAX_BOX_DELTA_PX,
                     "max_confidence_delta": MAX_CONF_DELTA,
                     "min_match_iou": MIN_MATCH_IOU, "device": "cpu", "seed": SEED},
        "summary": {"images": len(details), "images_passed": sum(item["passed"] for item in details),
                    "pt_detections": sum(item["detections"]["pt_count"] for item in details),
                    "onnx_detections": sum(item["detections"]["onnx_count"] for item in details),
                    "passed": passed},
        "images": details,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Report: {args.report}; {report['summary']['images_passed']}/{len(details)} passed")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
