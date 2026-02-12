from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.append(str(API_ROOT))

from kai_core.shared import brain_gate


def _iso_to_ts(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def _set_defaults(monkeypatch: pytest.MonkeyPatch, cache_path: Path, mode: str) -> None:
    monkeypatch.setenv("LICENSE_ENFORCEMENT_MODE", mode)
    monkeypatch.setenv("LICENSE_SERVER_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("LICENSE_INSTANCE_ID", "instance-abc")
    monkeypatch.setenv("LICENSE_REFRESH_INTERVAL_HOURS", "24")
    monkeypatch.setenv("LICENSE_RENEW_DAYS_BEFORE_EXP", "3")
    monkeypatch.setenv("LICENSE_GRACE_DAYS", "1")
    monkeypatch.setenv("LICENSE_REQUEST_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("LICENSE_CACHE_PATH", str(cache_path))


def test_mode_off_allows_without_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_defaults(monkeypatch, tmp_path / "cache.json", "off")
    state = brain_gate.refresh_license(force=True)
    allowed, reason = brain_gate.license_allows_request("/api/chat/send")
    assert state["mode"] == "off"
    assert state["valid"] is True
    assert allowed is True
    assert reason is None


def test_mode_soft_invalid_server_allows_with_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_defaults(monkeypatch, tmp_path / "cache.json", "soft")
    state = brain_gate.refresh_license(force=True)
    allowed, reason = brain_gate.license_allows_request("/api/chat/send")
    assert state["valid"] is False
    assert allowed is True
    assert isinstance(reason, str) and len(reason) > 0


def test_mode_hard_invalid_server_without_cache_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_defaults(monkeypatch, tmp_path / "cache.json", "hard")
    state = brain_gate.refresh_license(force=True)
    allowed, reason = brain_gate.license_allows_request("/api/chat/send")
    assert state["valid"] is False
    assert allowed is False
    assert isinstance(reason, str) and len(reason) > 0


def test_mode_hard_uses_cache_within_grace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    _set_defaults(monkeypatch, cache, "hard")

    expired_12_hours_ago = datetime.now(timezone.utc) - timedelta(hours=12)
    cache_payload = {
        "instance_id": "instance-abc",
        "sub": "instance-abc",
        "exp": _iso_to_ts(expired_12_hours_ago),
        "exp_iso": expired_12_hours_ago.replace(tzinfo=timezone.utc).isoformat(),
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    cache.write_text(json.dumps(cache_payload), encoding="utf-8")

    state = brain_gate.refresh_license(force=True)
    allowed, reason = brain_gate.license_allows_request("/api/chat/send")
    assert state["valid"] is True
    assert state["reason"] == "license_grace_active"
    assert allowed is True
    assert reason is None
