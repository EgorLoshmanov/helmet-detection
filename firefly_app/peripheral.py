from __future__ import annotations

from pathlib import Path


class Peripheral:
    """Supervisor-owned outputs; optional UART heartbeat and bounded sound."""

    def __init__(self, led: Path | None, uart: str | None, max_sound: float) -> None:
        self.led = led
        self.serial = None
        self.max_sound = max_sound
        self.last_output = None
        self.last_sent = -100.0
        self.alarm_started = None
        if uart:
            import serial
            self.serial = serial.Serial(uart, 115200, timeout=0.1, write_timeout=0.2)

    def update(self, state: str, now: float) -> bool:
        if state == "ALARM":
            if self.alarm_started is None:
                self.alarm_started = now
        else:
            self.alarm_started = None
        sound = state == "ALARM" and now - self.alarm_started < self.max_sound
        fault = state == "FAULT"
        light = state == "ALARM" or (fault and int(now * 2) % 2 == 0)
        outputs = (sound, fault, light)
        if self.led and (self.last_output is None or light != self.last_output[2]):
            self.led.write_text("1" if light else "0", encoding="ascii")
        if self.serial and (outputs[:2] != (self.last_output or (None, None))[:2] or now - self.last_sent >= 1):
            self.serial.write(f"ALARM:{int(sound)}\nFAULT:{int(fault)}\n".encode("ascii"))
            self.last_sent = now
        self.last_output = outputs
        return sound

    def close(self) -> None:
        try:
            if self.led:
                self.led.write_text("0", encoding="ascii")
        finally:
            if self.serial:
                try:
                    self.serial.write(b"ALARM:0\nFAULT:0\n")
                finally:
                    self.serial.close()
