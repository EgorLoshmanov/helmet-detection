from __future__ import annotations

import argparse
from pathlib import Path
import signal
import threading

from .config import load_config, save_config
from .inference import FP16_SHA256
from .monitor import Monitor
from .web import make_server


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Firefly safety monitor and web panel")
    parser.add_argument("--demo", action="store_true", help="Synthetic demo, no camera, model or peripherals")
    parser.add_argument("--source", default="0", help="USB camera index, /dev/video path, stream or video file")
    parser.add_argument("--model", type=Path, default=Path("models/helmet_detector_fp16.rknn"))
    parser.add_argument("--expected-sha256", default=FP16_SHA256, help="Accepted FP16 SHA-256; change only after model validation")
    parser.add_argument("--config", type=Path, default=Path("configs/firefly.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("runtime"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--led", type=Path, help="Writable sysfs brightness path (optional)")
    parser.add_argument("--uart", help="Optional ESP32 UART device, 115200 baud; requires pyserial")
    parser.add_argument("--max-sound-seconds", type=float, default=10)
    args = parser.parse_args()
    if args.demo and (args.led or args.uart):
        parser.error("Demo mode cannot drive physical outputs")
    if not 1 <= args.max_sound_seconds <= 60:
        parser.error("--max-sound-seconds must be between 1 and 60")
    if len(args.expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.expected_sha256):
        parser.error("--expected-sha256 must be 64 lowercase hex characters")
    # Browser changes persist locally, leaving the checked-in example untouched.
    local_config = args.data_dir / "settings.json"
    config = load_config(local_config if local_config.exists() else args.config)
    save_config(local_config, config)
    monitor = Monitor(config, local_config, args.source, args.model, args.expected_sha256,
                      args.data_dir, args.demo, args.led, args.uart, args.max_sound_seconds)
    try:
        server = make_server(monitor, args.host, args.port)
    except Exception:
        monitor.journal.close()
        raise
    stopped = threading.Event()

    def handle_signal(signum, frame):
        stopped.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    server_thread = threading.Thread(target=server.serve_forever, name="web", daemon=True)
    server_started = False
    try:
        monitor.start()
        server_thread.start()
        server_started = True
        print(f"Panel: http://{args.host}:{args.port} | {'DEMO (synthetic)' if args.demo else 'RKNN FP16 / NPU core 0'}", flush=True)
        while not stopped.wait(0.2):
            if monitor.stop.is_set():
                return 1
    finally:
        if server_started:
            server.shutdown()
        server.server_close()
        monitor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
