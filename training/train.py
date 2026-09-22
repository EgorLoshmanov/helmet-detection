#!/usr/bin/env python3
"""Train the FORTNITEBALLS YOLO baseline with a recorded configuration."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one epoch on 5% of train data without publishing model weights.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_source = config.pop("model")
    config["data"] = str(Path(config["data"]).resolve())
    config["project"] = str(Path(config["project"]).resolve())

    if args.smoke:
        config.update(
            {
                "epochs": 1,
                "patience": 0,
                "fraction": 0.05,
                "name": "yolov8n_smoke",
                "save_period": -1,
                "plots": False,
                "exist_ok": True,
            }
        )

    if config.get("device") == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable in this process")

    model = YOLO(model_source)
    results = model.train(**config)
    save_dir = Path(results.save_dir)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke" if args.smoke else "baseline",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "model_source": model_source,
        "config": config,
    }
    (save_dir / "environment.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if not args.smoke:
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(save_dir / "weights" / "best.pt", models_dir / "helmet_detector_best.pt")
        shutil.copy2(save_dir / "weights" / "last.pt", models_dir / "helmet_detector_last.pt")

    print(f"training_output={save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
