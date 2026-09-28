from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from firefly_app.camera import Camera, Frame
from firefly_app.config import Config, load_config, save_config
from firefly_app.inference import FP16_SHA256, RKNNDetector
from firefly_app.journal import Journal
from firefly_app.monitor import Monitor, Result
from firefly_app.peripheral import Peripheral
from firefly_app.postprocess import decode, in_roi, letterbox
from firefly_app.safety_state import SafetyState
from firefly_app.web import make_server


class SafetyTests(unittest.TestCase):
    def test_confirmation_and_hysteresis(self):
        machine = SafetyState()
        update = lambda t, violation, fault=None: machine.update(t, violation, fault, 1, 2)
        self.assertEqual(update(0, False), "SAFE")
        self.assertEqual(update(1, True), "PENDING")
        self.assertEqual(update(1.5, False), "SAFE")
        self.assertEqual(update(2, True), "PENDING")
        self.assertEqual(update(3, True), "ALARM")
        self.assertEqual(update(4, False), "ALARM")
        self.assertEqual(update(5, True), "ALARM")
        self.assertEqual(update(6, False), "ALARM")
        self.assertEqual(update(8, False), "SAFE")

    def test_fault_discards_old_confirmation(self):
        machine = SafetyState()
        machine.update(0, True, None, 1, 2)
        self.assertEqual(machine.update(0.9, True, "camera lost", 1, 2), "FAULT")
        self.assertEqual(machine.update(5, True, None, 1, 2), "PENDING")
        self.assertEqual(machine.update(6, True, None, 1, 2), "ALARM")


class ConfigTests(unittest.TestCase):
    def test_invalid_settings(self):
        for values in [{"confidence": float("nan")}, {"confidence": True}, {"roi": [0, 0, 0, 1]},
                       {"roi": [0, 0, 2, 1]}, {"alarm_on_seconds": 0}, {"unexpected": 1}, []]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config.from_dict(values)

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            config = Config.from_dict({"roi": [0, 0, 1, 1], "confidence": 0.7})
            save_config(path, config)
            self.assertEqual(load_config(path), config)


class DecodeTests(unittest.TestCase):
    def test_nms_and_coordinate_restoration(self):
        image = np.zeros((360, 640, 3), np.uint8)
        padded, ratio, padding = letterbox(image)
        self.assertEqual(padded.shape, (640, 640, 3))
        raw = np.zeros((1, 6, 8400), np.float32)
        # Two overlapping same-class boxes, one other-class box at the same place.
        raw[0, :, 0] = [320, 320, 100, 80, 0.1, 0.95]
        raw[0, :, 1] = [321, 320, 100, 80, 0.1, 0.9]
        raw[0, :, 2] = [320, 320, 100, 80, 0.85, 0.1]
        results = decode([raw], image.shape[:2], ratio, padding, 0.5, 0.7)
        self.assertEqual(len(results), 2)
        self.assertEqual([r["class_id"] for r in results], [1, 0])
        np.testing.assert_allclose(results[0]["box"], [270, 140, 370, 220])
        self.assertTrue(in_roi(results[0]["box"], image.shape[:2], (0.4, 0.3, 0.6, 0.7)))
        self.assertFalse(in_roi([0, 0, 40, 40], image.shape[:2], (0.4, 0.3, 0.6, 0.7)))

    def test_empty_and_invalid_outputs(self):
        raw = np.zeros((1, 6, 8400), np.float32)
        self.assertEqual(decode([raw], (480, 640), 1, (0, 80), 0.5, 0.7), [])
        with self.assertRaises(ValueError):
            decode([raw[:, :4], raw[:, 4:]], (480, 640), 1, (0, 80), 0.5, 0.7)
        raw[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            decode([raw], (480, 640), 1, (0, 80), 0.5, 0.7)

    def test_model_hash_checked_before_runtime_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unaccepted.rknn"
            path.write_bytes(b"not a model")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                RKNNDetector(path, FP16_SHA256)


class JournalTests(unittest.TestCase):
    def test_episodes_retention_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            journal = Journal(path, max_events=2)
            first = journal.start("ALARM", "ROI", [], {}, b"snapshot")
            journal.end(first, 1.2, "safe")
            journal.start("FAULT", "camera", [], {}, None)
            third = journal.start("ALARM", "ROI", [], {}, b"new snapshot")
            self.assertEqual(len(journal.recent()), 2)
            self.assertFalse((path / "snapshots" / f"{first}.jpg").exists())
            self.assertEqual(journal.snapshot(third), b"new snapshot")
            journal.close()
            restarted = Journal(path)
            self.assertTrue(all(row["ended_at"] for row in restarted.recent()))
            self.assertEqual(restarted.recent()[0]["end_reason"], "process interrupted")
            restarted.close()


class PeripheralTests(unittest.TestCase):
    def test_led_fault_pattern_and_sound_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "brightness"
            output = Peripheral(path, None, max_sound=2)
            self.assertTrue(output.update("ALARM", 0))
            self.assertFalse(output.update("ALARM", 2.1))
            self.assertEqual(path.read_text(), "1")
            output.update("FAULT", 3)
            self.assertEqual(path.read_text(), "1")
            output.update("FAULT", 3.5)
            self.assertEqual(path.read_text(), "0")
            self.assertTrue(output.update("ALARM", 4))
            output.close()
            self.assertEqual(path.read_text(), "0")


class WebTests(unittest.TestCase):
    def test_api_validation_and_snapshot_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            monitor = Monitor(Config(), path / "settings.json", "0", path / "model", FP16_SHA256, path, True)
            server = make_server(monitor, "127.0.0.1", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            root = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(root + "/") as response:
                    self.assertIn(b"Helmet Monitor", response.read())
                settings = Config.from_dict({"confidence": 0.65}).to_dict()
                request = Request(root + "/api/settings", json.dumps(settings).encode(), {"Content-Type": "application/json"})
                with urlopen(request) as response:
                    self.assertEqual(json.load(response)["confidence"], 0.65)
                self.assertEqual(load_config(path / "settings.json").confidence, 0.65)
                bad_requests = [
                    (Request(root + "/api/settings", b'{"confidence":-1}', {"Content-Type": "application/json"}), 400),
                    (Request(root + "/api/settings", b'{}', {"Content-Type": "application/json", "Origin": "https://elsewhere.example"}), 403),
                    (Request(root + "/api/settings", b'{}', {"Content-Type": "text/plain"}), 415),
                    (root + "/snapshots/../settings.json", 404),
                    (root + "/frame.jpg", 503)]
                for request, status in bad_requests:
                    with self.subTest(status=status), self.assertRaises(HTTPError) as caught:
                        urlopen(request)
                    self.assertEqual(caught.exception.code, status)
            finally:
                server.shutdown()
                server.server_close()
                monitor.journal.close()


class RecordingSourceTests(unittest.TestCase):
    """A recording must play at its own rate and rewind without faulting.

    Read unpaced, a clip is consumed far faster than real time and every pass ends
    in a read failure, so the confirmation timers never see a violation held for
    the configured interval.
    """

    def test_recording_is_paced_and_rewinds_without_error(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory) / "clip.mp4"
            writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 64))
            self.assertTrue(writer.isOpened())
            for index in range(6):
                writer.write(np.full((64, 64, 3), index * 30, np.uint8))
            writer.release()

            stop = threading.Event()
            camera = Camera(str(clip), stop)
            camera.start()
            try:
                errors, published = 0, 0
                started = time.monotonic()
                while time.monotonic() - started < 1.2:
                    frame, error = camera.snapshot()
                    if error and published:
                        errors += 1
                    if frame is not None:
                        published = max(published, frame.sequence)
                    time.sleep(0.005)
            finally:
                stop.set()
                camera.thread.join(2)
            # 0.5 s of clip in 1.2 s of wall time: it must loop, but stay near 12 fps.
            self.assertGreater(published, 6,
                f"only {published} frames were observable; the recording must play and rewind")
            self.assertLess(published, 30,
                f"{published} frames in 1.2 s: the recording was read faster than its frame rate")
            self.assertEqual(errors, 0, "rewinding must not surface as a camera error")

    def test_unreadable_recording_still_reports_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "broken.mp4"
            broken.write_bytes(b"not a video")
            stop = threading.Event()
            camera = Camera(str(broken), stop)
            camera.start()
            try:
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    if camera.snapshot()[1]:
                        break
                    time.sleep(0.01)
                self.assertTrue(camera.snapshot()[1], "a broken recording must report an error")
                self.assertIsNone(camera.snapshot()[0])
            finally:
                stop.set()
                camera.thread.join(2)


class SupervisorTests(unittest.TestCase):
    def test_one_frame_cannot_confirm_and_stale_stream_faults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = Config.from_dict({"alarm_on_seconds": 0.1, "max_frame_age_seconds": 0.5,
                                       "max_result_age_seconds": 0.5, "roi": [0, 0, 1, 1]})
            monitor = Monitor(config, path / "settings.json", "0", path / "model", FP16_SHA256, path, True)
            frame = Frame(1, time.monotonic(), np.zeros((120, 160, 3), np.uint8))
            detection = {"box": [30, 30, 60, 60], "class_id": 1, "confidence": 0.9}
            with monitor.camera.lock:
                monitor.camera.latest, monitor.camera.error = frame, ""
            with monitor.lock:
                monitor.result, monitor.worker_error = Result(frame, time.monotonic(), 0, [detection], 1), ""
            monitor.threads[1].start()
            try:
                self.wait_state(monitor, "PENDING")
                time.sleep(0.2)
                self.assertEqual(monitor.snapshot()[0]["state"], "PENDING")
                # A second fresh result confirms the time interval.
                next_frame = Frame(2, time.monotonic(), frame.image)
                with monitor.camera.lock:
                    monitor.camera.latest = next_frame
                with monitor.lock:
                    monitor.result = Result(next_frame, time.monotonic(), 0, [detection], 1)
                self.wait_state(monitor, "ALARM")
                self.wait_state(monitor, "FAULT")
                self.assertIn("fresh", monitor.snapshot()[0]["reason"])
                alarms = [r for r in monitor.journal.recent() if r["kind"] == "ALARM"]
                self.assertEqual(len(alarms), 1)
                self.assertIsNotNone(alarms[0]["ended_at"])
            finally:
                monitor.stop.set()
                monitor.threads[1].join(2)
                monitor.journal.close()

    @staticmethod
    def wait_state(monitor, state):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if monitor.snapshot()[0].get("state") == state:
                return
            time.sleep(0.01)
        raise AssertionError(f"Expected {state}, got {monitor.snapshot()[0]}")


if __name__ == "__main__":
    unittest.main()
