from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    confidence: float = 0.5
    nms_iou: float = 0.7
    alarm_on_seconds: float = 1.0
    alarm_off_seconds: float = 2.0
    max_frame_age_seconds: float = 2.0
    max_result_age_seconds: float = 2.0
    roi: tuple[float, float, float, float] = (0.15, 0.15, 0.85, 0.9)

    @classmethod
    def from_dict(cls, values: dict) -> "Config":
        if not isinstance(values, dict):
            raise ValueError("Configuration must be an object")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        defaults = asdict(cls())
        defaults.update(values)
        for name in defaults.keys() - {"roi"}:
            value = defaults[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            limits = (0.01, 1.0) if name in {"confidence", "nms_iou"} else (0.1, 60.0)
            if not limits[0] <= value <= limits[1]:
                raise ValueError(f"{name} must be between {limits[0]} and {limits[1]}")
        roi = defaults["roi"]
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            raise ValueError("roi must be [left, top, right, bottom]")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in roi):
            raise ValueError("ROI coordinates must be finite numbers between 0 and 1")
        if roi[0] >= roi[2] or roi[1] >= roi[3]:
            raise ValueError("ROI must have positive width and height")
        defaults["roi"] = tuple(roi)
        return cls(**defaults)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path) -> Config:
    return Config.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_config(path: Path, config: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
