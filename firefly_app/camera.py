from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import cv2
import numpy as np


@dataclass
class Frame:
    sequence: int
    captured_at: float
    image: np.ndarray
    demo_detections: list[dict] | None = None


class Camera:
    """One latest-frame slot; a blocked read is detected by the supervisor."""

    def __init__(self, source: str, stop: threading.Event, demo: bool = False) -> None:
        self.source = int(source) if source.isdigit() else source
        self.stop = stop
        self.demo = demo
        self.lock = threading.Lock()
        self.latest: Frame | None = None
        self.error = "Waiting for camera"
        self.thread = threading.Thread(target=self._run, name="capture", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def snapshot(self) -> tuple[Frame | None, str]:
        with self.lock:
            return self.latest, self.error

    def _publish(self, sequence: int, image, detections=None) -> None:
        with self.lock:
            self.latest = Frame(sequence, time.monotonic(), image, detections)
            self.error = ""

    def _fail(self, message: str) -> None:
        with self.lock:
            self.error = message
            self.latest = None

    def _run(self) -> None:
        sequence = 0
        if self.demo:
            started = time.monotonic()
            while not self.stop.is_set():
                phase = (time.monotonic() - started) % 40
                if 28 <= phase < 32:
                    self._fail("DEMO: camera disconnected")
                else:
                    image = np.full((540, 960, 3), (26, 22, 18), dtype=np.uint8)
                    detections = []
                    label = "EMPTY SCENE"
                    if phase < 6:
                        box, class_id, label = [15, 160, 100, 255], 1, "NO HELMET OUTSIDE ROI"
                    elif phase < 14 or phase >= 32:
                        box, class_id, label = [420, 180, 510, 280], 1, "NO HELMET INSIDE ROI"
                    elif phase < 22:
                        box, class_id, label = [420, 180, 510, 280], 0, "HELMET INSIDE ROI"
                    else:
                        box = None
                    if box:
                        cv2.circle(image, (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)), 40,
                                   (0, 200, 230) if class_id == 0 else (150, 180, 220), -1)
                        detections = [{"box": box, "class_id": class_id, "confidence": 0.93}]
                    cv2.putText(image, "DEMO / SYNTHETIC DATA", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 110), 2)
                    cv2.putText(image, label, (30, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210, 210, 210), 1)
                    sequence += 1
                    self._publish(sequence, image, detections)
                self.stop.wait(0.1)
            return
        while not self.stop.is_set():
            capture = None
            try:
                if isinstance(self.source, str) and self.source.startswith(("rtsp://", "http://", "https://")):
                    capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
                else:
                    capture = cv2.VideoCapture(self.source)
                if not capture.isOpened():
                    raise RuntimeError("Cannot open camera/source")
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                while not self.stop.is_set():
                    ok, image = capture.read()
                    if not ok or image is None or image.size == 0:
                        raise RuntimeError("Camera frame read failed (or video ended)")
                    sequence += 1
                    self._publish(sequence, image)
            except Exception as exc:
                self._fail(str(exc))
            finally:
                if capture is not None:
                    capture.release()
            self.stop.wait(1.0)
