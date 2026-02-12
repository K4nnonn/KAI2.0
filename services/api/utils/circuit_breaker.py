from __future__ import annotations

import time


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        cooldown_seconds: int = 60,
        enabled: bool = True,
    ):
        self.name = name
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_seconds = max(1, int(cooldown_seconds))
        self.enabled = enabled
        self._failures = 0
        self._opened_until = 0.0

    def allow(self) -> bool:
        if not self.enabled:
            return True
        now = time.time()
        if now < self._opened_until:
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_until = 0.0

    def record_failure(self) -> None:
        if not self.enabled:
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_until = time.time() + self.cooldown_seconds

    def status(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "failures": self._failures,
            "open_until": self._opened_until,
        }
