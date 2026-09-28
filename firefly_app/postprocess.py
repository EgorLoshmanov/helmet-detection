from __future__ import annotations

import cv2
import numpy as np


def letterbox(bgr: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    height, width = bgr.shape[:2]
    ratio = min(size / height, size / width)
    rw, rh = round(width * ratio), round(height * ratio)
    left = round((size - rw) / 2 - 0.1)
    top = round((size - rh) / 2 - 0.1)
    resized = cv2.resize(bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, size - rh - top, left, size - rw - left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, ratio, (left, top)


def decode(outputs: list[np.ndarray], shape: tuple[int, int], ratio: float,
           padding: tuple[int, int], confidence: float, iou: float) -> list[dict]:
    # Only the accepted FP16 export: decoded xywh + two class probabilities.
    if len(outputs) != 1 or outputs[0].shape != (1, 6, 8400):
        raise ValueError("Expected FP16 YOLO output (1, 6, 8400); split/INT8 models are unsupported")
    raw = np.asarray(outputs[0], dtype=np.float32)[0].T
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite model output")
    classes = raw[:, 4:].argmax(axis=1)
    scores = raw[np.arange(len(raw)), classes + 4]
    candidates = np.flatnonzero((scores >= confidence) & (raw[:, 2] > 0) & (raw[:, 3] > 0))
    # Bound CPU NMS work on pathological outputs.
    candidates = candidates[np.argsort(-scores[candidates], kind="stable")[:3000]]
    xyxy = np.empty((len(candidates), 4), dtype=np.float32)
    rows = raw[candidates]
    xyxy[:, :2] = rows[:, :2] - rows[:, 2:4] / 2
    xyxy[:, 2:] = rows[:, :2] + rows[:, 2:4] / 2
    kept = []
    remaining = np.arange(len(candidates))
    while len(remaining) and len(kept) < 300:
        first, rest = remaining[0], remaining[1:]
        kept.append(first)
        if not len(rest):
            break
        intersection = np.maximum(0, np.minimum(xyxy[first, 2:], xyxy[rest, 2:]) - np.maximum(xyxy[first, :2], xyxy[rest, :2])).prod(axis=1)
        a = (xyxy[first, 2:] - xyxy[first, :2]).prod()
        b = (xyxy[rest, 2:] - xyxy[rest, :2]).prod(axis=1)
        overlap = intersection / np.maximum(a + b - intersection, 1e-9)
        same_class = classes[candidates[first]] == classes[candidates[rest]]
        remaining = rest[~((overlap > iou) & same_class)]
    height, width = shape
    detections = []
    for index in kept:
        box = xyxy[index].copy()
        box[[0, 2]] = np.clip((box[[0, 2]] - padding[0]) / ratio, 0, width)
        box[[1, 3]] = np.clip((box[[1, 3]] - padding[1]) / ratio, 0, height)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        candidate = candidates[index]
        detections.append({"box": box.tolist(), "class_id": int(classes[candidate]),
                           "confidence": float(scores[candidate])})
    return detections


def in_roi(box: list[float], shape: tuple[int, int], roi: tuple[float, ...]) -> bool:
    """ROI membership uses the centre of the detected head/helmet box."""
    height, width = shape
    x, y = (box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height)
    return roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]
