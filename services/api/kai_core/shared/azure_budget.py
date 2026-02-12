"""
Centralized Azure usage guardrails.

Purpose:
- Keep Azure calls bounded (cost control).
- Provide a single policy check for all Azure call sites.
"""
from __future__ import annotations

import os
import time
import threading
import logging
from typing import Optional, Tuple

from kai_core.config import get_deployment_mode, is_azure_openai_enabled

_BUDGET_LOCK = threading.Lock()
_BUDGET = {
    "minute_start": time.time(),
    "minute_count": 0,
    "hour_start": time.time(),
    "hour_count": 0,
    "total_count": 0,
}


def _read_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except Exception:
        return default


def _allowed_list(env_name: str) -> list[str]:
    raw = (os.environ.get(env_name) or "").strip()
    if not raw:
        return []
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _azure_budget_limits() -> tuple[bool, int, int, int]:
    allow = os.environ.get("ALLOW_AZURE_FALLBACK", "true").lower() == "true"
    max_per_minute = _read_int("AZURE_LLM_MAX_CALLS_PER_MINUTE", 5)
    max_per_hour = _read_int("AZURE_LLM_MAX_CALLS_PER_HOUR", 50)
    max_total = _read_int("AZURE_LLM_MAX_CALLS_TOTAL", 500)
    return allow, max_per_minute, max_per_hour, max_total


def azure_budget_snapshot() -> dict:
    allow, max_per_minute, max_per_hour, max_total = _azure_budget_limits()
    with _BUDGET_LOCK:
        snapshot = dict(_BUDGET)
    snapshot.update(
        {
            "allow_fallback": allow,
            "max_per_minute": max_per_minute,
            "max_per_hour": max_per_hour,
            "max_total": max_total,
        }
    )
    return snapshot


def allow_azure_usage(
    module: str,
    purpose: Optional[str] = None,
    intent: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Gate Azure calls by policy + budget.
    Returns (allowed, reason_if_denied).
    """
    if not is_azure_openai_enabled():
        return False, "azure_disabled"

    # In LOCAL mode, require explicit opt-in via AZURE_OPENAI_ENABLED.
    if get_deployment_mode() == "LOCAL" and os.environ.get("AZURE_OPENAI_ENABLED") is None:
        return False, "local_mode_requires_explicit_opt_in"

    allowed_modules = _allowed_list("AZURE_OPENAI_ALLOWED_MODULES")
    allowed_purposes = _allowed_list("AZURE_OPENAI_ALLOWED_PURPOSES")
    module_key = (module or "unknown").strip().lower()
    purpose_key = (purpose or "").strip().lower()

    if allowed_modules and module_key not in allowed_modules:
        return False, "module_not_allowlisted"
    if allowed_purposes and purpose_key and purpose_key not in allowed_purposes:
        return False, "purpose_not_allowlisted"

    allow_fallback, max_per_minute, max_per_hour, max_total = _azure_budget_limits()
    if not allow_fallback:
        return False, "azure_fallback_disabled"

    now = time.time()
    with _BUDGET_LOCK:
        if now - _BUDGET["minute_start"] >= 60:
            _BUDGET["minute_start"] = now
            _BUDGET["minute_count"] = 0
        if now - _BUDGET["hour_start"] >= 3600:
            _BUDGET["hour_start"] = now
            _BUDGET["hour_count"] = 0
        if max_total > 0 and _BUDGET["total_count"] >= max_total:
            return False, "azure_budget_total_exceeded"
        if max_per_minute > 0 and _BUDGET["minute_count"] >= max_per_minute:
            return False, "azure_budget_minute_exceeded"
        if max_per_hour > 0 and _BUDGET["hour_count"] >= max_per_hour:
            return False, "azure_budget_hour_exceeded"
        _BUDGET["total_count"] += 1
        _BUDGET["minute_count"] += 1
        _BUDGET["hour_count"] += 1

    logging.info(
        "[azure_budget] allowed module=%s purpose=%s intent=%s",
        module_key,
        purpose_key,
        intent or "",
    )
    return True, None
