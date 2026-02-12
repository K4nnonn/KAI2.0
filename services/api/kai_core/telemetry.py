"""
Telemetry helpers for Kai platform.

Provides a stable import path for usage logging in both source and binary modes.
In environments without central telemetry, this becomes a no-op logger.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional


def log_openai_usage(source: str, metadata: Optional[Dict[str, Any]] = None, usage: Optional[Dict[str, Any]] = None, latency_ms: Optional[float] = None) -> None:
    """
    Lightweight telemetry hook. Emits a structured log for LLM usage.
    Safe to call even if telemetry backend is absent.
    """
    payload = {
        "source": source,
        "metadata": metadata or {},
        "usage": usage or {},
        "latency_ms": latency_ms,
    }
    try:
        logging.info("[telemetry] %s", json.dumps(payload))
    except Exception:
        # Swallow any logging/serialization issues to avoid breaking the app.
        logging.info("[telemetry] %s", payload)

