from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Journal:
    """Bounded event history; an event is one ALARM or FAULT episode."""

    def __init__(self, directory: Path, max_events: int = 500) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.snapshots = directory / "snapshots"
        self.snapshots.mkdir(exist_ok=True)
        self.max_events = max_events
        self.lock = threading.Lock()
        self.db = sqlite3.connect(directory / "events.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY, kind TEXT NOT NULL, started_at TEXT NOT NULL,
            ended_at TEXT, duration_seconds REAL, reason TEXT NOT NULL,
            end_reason TEXT, detections TEXT NOT NULL, settings TEXT NOT NULL,
            snapshot TEXT, confirmation_seconds REAL)""")
        # An unclosed event belongs to a previous process, never to this run.
        self.db.execute("UPDATE events SET ended_at=?, end_reason='process interrupted' WHERE ended_at IS NULL", (utc_now(),))
        self.db.commit()

    def start(self, kind: str, reason: str, detections: list, settings: dict,
              jpeg: bytes | None, confirmation_seconds: float | None = None) -> int:
        with self.lock:
            cursor = self.db.execute("""INSERT INTO events
                (kind, started_at, reason, detections, settings, confirmation_seconds)
                VALUES (?, ?, ?, ?, ?, ?)""", (kind, utc_now(), reason,
                json.dumps(detections), json.dumps(settings), confirmation_seconds))
            event_id = cursor.lastrowid
            try:
                if jpeg is not None:
                    name = f"{event_id}.jpg"
                    (self.snapshots / name).write_bytes(jpeg)
                    self.db.execute("UPDATE events SET snapshot=? WHERE id=?", (name, event_id))
                stale = self.db.execute("SELECT id,snapshot FROM events ORDER BY id DESC LIMIT -1 OFFSET ?", (self.max_events,)).fetchall()
                for row in stale:
                    self.db.execute("DELETE FROM events WHERE id=?", (row["id"],))
                    if row["snapshot"]:
                        (self.snapshots / row["snapshot"]).unlink(missing_ok=True)
                self.db.commit()
            except Exception:
                self.db.rollback()
                (self.snapshots / f"{event_id}.jpg").unlink(missing_ok=True)
                raise
            return event_id

    def end(self, event_id: int, duration: float, reason: str) -> None:
        with self.lock:
            self.db.execute("UPDATE events SET ended_at=?,duration_seconds=?,end_reason=? WHERE id=?",
                            (utc_now(), round(duration, 3), reason, event_id))
            self.db.commit()

    def recent(self, limit: int = 100) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(row) for row in rows]

    def snapshot(self, event_id: int) -> bytes | None:
        with self.lock:
            row = self.db.execute("SELECT snapshot FROM events WHERE id=?", (event_id,)).fetchone()
            if not row or not row["snapshot"]:
                return None
            path = self.snapshots / row["snapshot"]
            return path.read_bytes() if path.exists() else None

    def close(self) -> None:
        with self.lock:
            self.db.close()
