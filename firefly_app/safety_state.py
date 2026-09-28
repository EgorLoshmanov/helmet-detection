from __future__ import annotations


class SafetyState:
    """Continuous confirmation in monotonic seconds; failures reset confirmation."""

    def __init__(self) -> None:
        self.state = "FAULT"
        self.since: float | None = None
        self.reason = "Waiting for a fresh inference result"

    def reset(self) -> None:
        self.__init__()

    def update(self, now: float, violation: bool, fault: str | None, on: float, off: float) -> str:
        if fault:
            self.state, self.since, self.reason = "FAULT", None, fault
            return self.state
        self.reason = ""
        if self.state == "FAULT":
            self.state, self.since = "SAFE", None
        if violation:
            if self.state == "SAFE":
                self.state, self.since = "PENDING", now
            elif self.state == "ALARM":
                self.since = None
            if self.state == "PENDING" and self.since is not None and now - self.since >= on:
                self.state, self.since = "ALARM", None
        else:
            if self.state == "PENDING":
                self.state, self.since = "SAFE", None
            elif self.state == "ALARM":
                if self.since is None:
                    self.since = now
                if now - self.since >= off:
                    self.state, self.since = "SAFE", None
        return self.state
