from __future__ import annotations

import time
from typing import Dict, Tuple


class TokenBucket:
    def __init__(self, rate_per_minute: float, burst: int, ttl_seconds: int = 900):
        self.rate_per_minute = max(0.1, float(rate_per_minute))
        self.burst = max(1, int(burst))
        self.ttl_seconds = max(60, int(ttl_seconds))
        self._buckets: Dict[str, Tuple[float, float]] = {}

    def _refill(self, tokens: float, last_ts: float, now: float) -> float:
        rate_per_sec = self.rate_per_minute / 60.0
        return min(self.burst, tokens + (now - last_ts) * rate_per_sec)

    def allow(self, key: str) -> bool:
        now = time.time()
        entry = self._buckets.get(key)
        if not entry:
            self._buckets[key] = (self.burst - 1.0, now)
            return True

        tokens, last_ts = entry
        if now - last_ts > self.ttl_seconds:
            self._buckets[key] = (self.burst - 1.0, now)
            return True

        tokens = self._refill(tokens, last_ts, now)
        if tokens < 1.0:
            self._buckets[key] = (tokens, now)
            return False

        self._buckets[key] = (tokens - 1.0, now)
        return True

    def size(self) -> int:
        return len(self._buckets)
