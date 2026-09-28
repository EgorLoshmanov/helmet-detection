from __future__ import annotations

import hashlib
from pathlib import Path
import time

import cv2

from .postprocess import decode, letterbox


FP16_SHA256 = "72c5fdf379d0f9dced422e27689cb4a344b12b32c84bc188f91a8ce9fa271747"


class RKNNDetector:
    def __init__(self, model: Path, expected_sha256: str) -> None:
        digest = hashlib.sha256()
        with model.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.sha256 = digest.hexdigest()
        if self.sha256 != expected_sha256:
            raise ValueError("Model SHA-256 differs from the accepted FP16 model; see docs/edge-monitor.md")
        from rknnlite.api import RKNNLite
        self.runtime = RKNNLite()
        try:
            if self.runtime.load_rknn(str(model)) != 0:
                raise RuntimeError("RKNN model loading failed")
            if self.runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
                raise RuntimeError("RKNN NPU initialization failed")
        except Exception:
            self.runtime.release()
            raise

    def predict(self, frame, config):
        padded, ratio, padding = letterbox(frame)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)[None]
        started = time.monotonic()
        outputs = self.runtime.inference(inputs=[rgb], data_format=["nhwc"])
        inference_ms = (time.monotonic() - started) * 1000
        if outputs is None:
            raise RuntimeError("NPU returned no outputs")
        detections = decode(outputs, frame.shape[:2], ratio, padding, config.confidence, config.nms_iou)
        return detections, inference_ms

    def close(self) -> None:
        self.runtime.release()
