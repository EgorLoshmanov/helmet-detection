from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import io
import csv
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from .camera import Camera, Frame
from .config import Config, save_config
from .inference import RKNNDetector
from .journal import Journal, utc_now
from .peripheral import Peripheral
from .postprocess import in_roi
from .safety_state import SafetyState


@dataclass
class Result:
    frame: Frame
    completed_at: float
    config_version: int
    detections: list[dict]
    inference_ms: float


class Monitor:
    def __init__(self, config: Config, config_path: Path, source: str, model: Path,
                 expected_sha256: str, data_dir: Path, demo: bool = False,
                 led: Path | None = None, uart: str | None = None, max_sound: float = 10) -> None:
        self.lock = threading.Lock()
        self.config = config
        self.config_path = config_path
        self.config_version = 0
        self.stop = threading.Event()
        self.camera = Camera(source, self.stop, demo)
        self.model = model
        self.expected_sha256 = expected_sha256
        self.demo = demo
        self.journal = Journal(data_dir)
        self.led, self.uart, self.max_sound = led, uart, max_sound
        self.result: Result | None = None
        self.worker_error = "Waiting for detector"
        self.jpeg: bytes | None = None
        self.status = {"state": "FAULT", "reason": "Starting", "mode": "demo" if demo else "rknn_fp16"}
        self.metrics = deque(maxlen=1000)
        self.threads = [threading.Thread(target=self._infer, name="inference", daemon=True),
                        threading.Thread(target=self._supervise, name="supervisor", daemon=True)]

    def start(self) -> None:
        self.camera.start()
        for thread in self.threads:
            thread.start()

    def settings(self) -> dict:
        with self.lock:
            return self.config.to_dict()

    def configure(self, values: dict) -> dict:
        config = Config.from_dict(values)
        with self.lock:
            save_config(self.config_path, config)
            self.config, self.config_version = config, self.config_version + 1
        return config.to_dict()

    def snapshot(self) -> tuple[dict, bytes | None]:
        with self.lock:
            return dict(self.status), self.jpeg

    def metrics_csv(self) -> str:
        with self.lock:
            records = list(self.metrics)
        output = io.StringIO()
        fields = ["timestamp", "sequence", "inference_ms", "frame_to_result_ms", "state", "mode"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
        return output.getvalue()

    def _infer(self) -> None:
        detector = None
        try:
            if not self.demo:
                detector = RKNNDetector(self.model, self.expected_sha256)
            last_sequence = -1
            while not self.stop.is_set():
                frame, error = self.camera.snapshot()
                if frame is None or error or frame.sequence == last_sequence:
                    self.stop.wait(0.01)
                    continue
                with self.lock:
                    config, version = self.config, self.config_version
                last_sequence = frame.sequence
                try:
                    if self.demo:
                        detections = [dict(d) for d in frame.demo_detections if d["confidence"] >= config.confidence]
                        inference_ms = 0.0
                    else:
                        detections, inference_ms = detector.predict(frame.image, config)
                    result = Result(frame, time.monotonic(), version, detections, inference_ms)
                    with self.lock:
                        self.result, self.worker_error = result, ""
                except Exception as exc:
                    with self.lock:
                        self.result, self.worker_error = None, f"Inference failed: {exc}"
                    self.stop.wait(0.2)
        except Exception as exc:
            with self.lock:
                self.worker_error = f"Detector startup failed: {exc}"
        finally:
            if detector is not None:
                detector.close()

    @staticmethod
    def _render(result: Result | None, config: Config, state: str, reason: str, fps: float) -> bytes:
        if result is None or state == "FAULT":
            image = np.full((540, 960, 3), 24, dtype=np.uint8)
        else:
            image = result.frame.image.copy()
            height, width = image.shape[:2]
            left, top, right, bottom = config.roi
            cv2.rectangle(image, (round(left * width), round(top * height)),
                          (round(right * width), round(bottom * height)), (255, 190, 60), 2)
            for detection in result.detections:
                box = [round(x) for x in detection["box"]]
                active = in_roi(detection["box"], image.shape[:2], config.roi)
                color = (60, 210, 90) if detection["class_id"] == 0 else (60, 80, 240)
                if not active:
                    color = (140, 140, 140)
                cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), color, 2)
                label = f"{'helmet' if detection['class_id'] == 0 else 'no_helmet'} {detection['confidence']:.2f}"
                cv2.putText(image, label, (box[0], max(20, box[1] - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.rectangle(image, (0, 55), (image.shape[1], 95), (15, 15, 15), -1)
        color = (60, 80, 240) if state in {"ALARM", "FAULT"} else (80, 220, 170)
        cv2.putText(image, f"{state} | processed FPS {fps:.1f}", (20, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        if reason:
            cv2.putText(image, reason[:100], (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        # Bound panel/event storage even when the camera is 4K.
        if image.shape[1] > 1280:
            image = cv2.resize(image, (1280, round(image.shape[0] * 1280 / image.shape[1])))
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return jpeg.tobytes()

    def _supervise(self) -> None:
        safety = SafetyState()
        peripheral = None
        peripheral_error = ""
        event_id = None
        event_started = 0.0
        episode_key = None
        observed_version = -1
        processed_sequence = -1
        processed_times = deque()
        pending_started = None
        confirmation_seconds = None
        render_key = None
        jpeg = None
        try:
            try:
                peripheral = Peripheral(self.led, self.uart, self.max_sound)
                peripheral.update("FAULT", time.monotonic())
            except Exception as exc:
                peripheral_error = f"Peripheral initialization failed: {exc}"
            while not self.stop.is_set():
                now = time.monotonic()
                frame, camera_error = self.camera.snapshot()
                with self.lock:
                    result, error, config, version = self.result, self.worker_error, self.config, self.config_version
                if observed_version != version:
                    if event_id is not None:
                        self.journal.end(event_id, now - event_started, "settings changed")
                    event_id, episode_key = None, None
                    safety.reset()
                    pending_started = None
                    observed_version = version
                fault = peripheral_error or camera_error or error
                if not fault and (frame is None or now - frame.captured_at > config.max_frame_age_seconds):
                    fault = "Camera stopped producing fresh frames"
                if not fault and (result is None or result.config_version != version):
                    fault = "Waiting for inference with current settings"
                if not fault and (now - result.completed_at > config.max_result_age_seconds or
                                  now - result.frame.captured_at > config.max_frame_age_seconds):
                    fault = "Inference result is stale"
                new_result = result is not None and result.frame.sequence != processed_sequence
                violation = bool(result and any(d["class_id"] == 1 and in_roi(d["box"], result.frame.image.shape[:2], config.roi)
                                               for d in result.detections))
                previous = safety.state
                # Timers advance only on new inference results. Re-reading one detection
                # cannot turn a single frame into a confirmed violation.
                if fault or new_result:
                    state = safety.update(now, violation, fault or None, config.alarm_on_seconds, config.alarm_off_seconds)
                else:
                    state = safety.state
                if state == "PENDING" and previous != "PENDING":
                    pending_started = result.frame.captured_at if result else now
                if state == "ALARM" and previous != "ALARM":
                    confirmation_seconds = now - pending_started if pending_started is not None else None
                if state in {"SAFE", "FAULT"}:
                    pending_started = None
                sound = False
                if peripheral and not peripheral_error:
                    try:
                        sound = peripheral.update(state, now)
                    except Exception as exc:
                        peripheral_error = f"Peripheral write failed: {exc}"
                        state = safety.update(now, False, peripheral_error, config.alarm_on_seconds, config.alarm_off_seconds)
                        try:
                            peripheral.close()
                        except Exception:
                            pass
                        peripheral = None
                if new_result:
                    processed_sequence = result.frame.sequence
                    if not fault:
                        processed_times.append(now)
                        with self.lock:
                            self.metrics.append({"timestamp": utc_now(), "sequence": processed_sequence,
                                "inference_ms": round(result.inference_ms, 3),
                                "frame_to_result_ms": round((result.completed_at - result.frame.captured_at) * 1000, 3),
                                "state": state, "mode": "demo" if self.demo else "rknn_fp16"})
                while processed_times and now - processed_times[0] > 2:
                    processed_times.popleft()
                fps = (len(processed_times) - 1) / (processed_times[-1] - processed_times[0]) if len(processed_times) > 1 else 0.0
                if state == "FAULT":
                    fps = 0.0
                next_render_key = (result.frame.sequence if result and state != "FAULT" else None,
                                   state, safety.reason, version)
                if next_render_key != render_key:
                    jpeg = self._render(result, config, state, safety.reason, fps)
                    render_key = next_render_key
                key = (state, safety.reason if state == "FAULT" else "") if state in {"ALARM", "FAULT"} else None
                if key != episode_key:
                    if event_id is not None:
                        self.journal.end(event_id, now - event_started, f"state changed to {state}")
                    event_id = None
                    if key is not None:
                        event_id = self.journal.start(state, safety.reason or "no_helmet inside ROI",
                            result.detections if result and state == "ALARM" else [], config.to_dict(),
                            jpeg if state == "ALARM" else None, confirmation_seconds if state == "ALARM" else None)
                        event_started = now
                    episode_key = key
                    print(f"{utc_now()} state={state} {safety.reason}", flush=True)
                with self.lock:
                    samples = list(self.metrics)
                    self.jpeg = jpeg
                    self.status = {"state": state, "reason": safety.reason, "mode": "demo" if self.demo else "rknn_fp16",
                        "fps": round(fps, 2), "inference_ms": round(result.inference_ms, 2) if result and not fault else None,
                        "frame_to_result_ms": round((result.completed_at - result.frame.captured_at) * 1000, 2) if result and not fault else None,
                        "frame_age_ms": round((now - frame.captured_at) * 1000, 2) if frame else None,
                        "violations": sum(d["class_id"] == 1 and in_roi(d["box"], result.frame.image.shape[:2], config.roi) for d in result.detections) if result and not fault else 0,
                        "sound_active": bool(sound and self.uart), "event_id": event_id,
                        "model_sha256": None if self.demo else self.expected_sha256,
                        "latency_p95_ms": round(float(np.percentile([r["frame_to_result_ms"] for r in samples], 95)), 2) if samples else None,
                        "supervisor_alive": True}
                self.stop.wait(0.05)
        except Exception as exc:
            with self.lock:
                self.status = {"state": "FAULT", "reason": f"Supervisor failed: {exc}",
                               "mode": "demo" if self.demo else "rknn_fp16", "supervisor_alive": False}
                self.jpeg = None
            self.stop.set()
            print(f"Supervisor failed: {exc}", flush=True)
        finally:
            if event_id is not None:
                try:
                    self.journal.end(event_id, time.monotonic() - event_started, "application stopped")
                except Exception:
                    pass
            if peripheral:
                try:
                    peripheral.close()
                except Exception as exc:
                    print(f"Peripheral cleanup failed: {exc}", flush=True)

    def close(self) -> None:
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=4)
        self.camera.thread.join(timeout=1)
        if not self.threads[1].is_alive():
            self.journal.close()
