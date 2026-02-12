from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from uuid import uuid4

import jwt
import requests

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {
    "mode": "soft",
    "valid": True,
    "reason": None,
    "exp": None,
    "cache_present": False,
    "last_check": None,
    "next_check": None,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_int(value: str | None, default: int) -> int:
    try:
        return int(str(value or default).strip())
    except Exception:
        return default


def _mode() -> str:
    mode = str(os.environ.get("LICENSE_ENFORCEMENT_MODE", "soft")).strip().lower()
    return mode if mode in {"off", "soft", "hard"} else "soft"


def _instance_id() -> str:
    return str(os.environ.get("LICENSE_INSTANCE_ID", "")).strip()


def _license_url() -> str:
    return str(os.environ.get("LICENSE_SERVER_URL", "https://license.kelvinale.com")).strip()


def _refresh_interval_seconds() -> int:
    return max(1, _to_int(os.environ.get("LICENSE_REFRESH_INTERVAL_HOURS"), 24)) * 3600


def _renew_before_seconds() -> int:
    return max(0, _to_int(os.environ.get("LICENSE_RENEW_DAYS_BEFORE_EXP"), 3)) * 86400


def _grace_seconds() -> int:
    return max(0, _to_int(os.environ.get("LICENSE_GRACE_DAYS"), 1)) * 86400


def _request_timeout_seconds() -> int:
    return max(1, _to_int(os.environ.get("LICENSE_REQUEST_TIMEOUT_SECONDS"), 8))


def _cache_path() -> Path:
    raw = str(os.environ.get("LICENSE_CACHE_PATH", "./runtime/license_cache.json")).strip() or "./runtime/license_cache.json"
    return Path(raw)


def _iso_from_ts(exp_ts: int | None) -> str | None:
    if not exp_ts:
        return None
    try:
        return datetime.fromtimestamp(int(exp_ts), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _read_cache() -> dict[str, Any] | None:
    cache_file = _cache_path()
    try:
        if not cache_file.exists():
            return None
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(data: dict[str, Any]) -> None:
    cache_file = _cache_path()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "instance_id": data.get("instance_id"),
        "sub": data.get("sub"),
        "exp": int(data["exp"]) if data.get("exp") is not None else None,
        "exp_iso": data.get("exp_iso"),
        "validated_at": data.get("validated_at"),
    }
    cache_file.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _decode_license_token(token: str, expected_sub: str) -> tuple[bool, str | None, int | None]:
    if not token or not isinstance(token, str):
        return False, "license_token_missing", None
    try:
        claims = jwt.decode(
            token,
            options={"verify_signature": False, "verify_exp": False, "verify_nbf": False, "verify_iat": False},
            algorithms=["HS256", "RS256", "ES256", "none"],
        )
    except Exception:
        return False, "license_token_parse_failed", None

    if not isinstance(claims, dict):
        return False, "license_token_claims_invalid", None

    subject = str(claims.get("sub", "")).strip()
    if subject != expected_sub:
        return False, "license_sub_mismatch", None

    try:
        exp_ts = int(claims.get("exp"))
    except Exception:
        return False, "license_exp_missing", None

    if exp_ts <= int(_utc_now().timestamp()):
        return False, "license_expired", exp_ts
    return True, None, exp_ts


def _request_license_token() -> tuple[bool, str | None, int | None]:
    instance_id = _instance_id()
    if not instance_id:
        return False, "license_instance_id_missing", None

    server_url = _license_url()
    if not server_url:
        return False, "license_server_url_missing", None

    endpoint = urljoin(server_url.rstrip("/") + "/", "license/token")
    payload = {
        "instance_id": instance_id,
        "timestamp": int(_utc_now().timestamp()),
        "nonce": uuid4().hex,
    }
    try:
        response = requests.post(endpoint, json=payload, timeout=_request_timeout_seconds())
    except Exception:
        return False, "license_server_unreachable", None

    if response.status_code != 200:
        return False, f"license_http_{response.status_code}", None

    try:
        body = response.json()
    except Exception:
        return False, "license_response_not_json", None

    token = body.get("token") or body.get("license_token") or body.get("access_token")
    ok, reason, exp_ts = _decode_license_token(token, expected_sub=instance_id)
    if not ok:
        return False, reason, exp_ts
    return True, None, exp_ts


def _effective_status_after_cache(remote_reason: str | None) -> tuple[bool, str | None, int | None, bool]:
    now_ts = int(_utc_now().timestamp())
    grace_seconds = _grace_seconds()
    instance_id = _instance_id()

    cache = _read_cache()
    if not cache:
        return False, remote_reason or "license_unavailable", None, False

    cache_sub = str(cache.get("sub") or cache.get("instance_id") or "").strip()
    try:
        cache_exp = int(cache.get("exp"))
    except Exception:
        cache_exp = None

    if cache_sub != instance_id or cache_exp is None:
        return False, remote_reason or "license_cache_invalid", None, True

    if cache_exp > now_ts:
        return True, "license_cache_active", cache_exp, True

    if grace_seconds > 0 and now_ts <= (cache_exp + grace_seconds):
        return True, "license_grace_active", cache_exp, True

    return False, remote_reason or "license_expired_out_of_grace", cache_exp, True


def refresh_license(force: bool = False) -> dict[str, Any]:
    with _LOCK:
        mode = _mode()
        now = _utc_now()
        now_ts = int(now.timestamp())
        now_iso = now.isoformat()

        _STATE["mode"] = mode
        _STATE["cache_present"] = _cache_path().exists()

        if mode == "off":
            _STATE.update(
                {
                    "valid": True,
                    "reason": None,
                    "exp": None,
                    "last_check": now_iso,
                    "next_check": None,
                }
            )
            return dict(_STATE)

        next_check = _STATE.get("next_check")
        if not force and next_check:
            try:
                if now_ts < int(next_check):
                    return dict(_STATE)
            except Exception:
                pass

        remote_ok, remote_reason, remote_exp = _request_license_token()
        if remote_ok and remote_exp:
            exp_iso = _iso_from_ts(remote_exp)
            _write_cache(
                {
                    "instance_id": _instance_id(),
                    "sub": _instance_id(),
                    "exp": remote_exp,
                    "exp_iso": exp_iso,
                    "validated_at": now_iso,
                }
            )
            refresh_due = now_ts + _refresh_interval_seconds()
            renew_due = remote_exp - _renew_before_seconds()
            _STATE.update(
                {
                    "valid": True,
                    "reason": None,
                    "exp": remote_exp,
                    "cache_present": True,
                    "last_check": now_iso,
                    "next_check": min(refresh_due, renew_due) if renew_due > now_ts else now_ts,
                }
            )
            return dict(_STATE)

        cache_valid, cache_reason, cache_exp, cache_present = _effective_status_after_cache(remote_reason)
        _STATE.update(
            {
                "valid": bool(cache_valid),
                "reason": cache_reason or remote_reason or "license_unavailable",
                "exp": cache_exp,
                "cache_present": bool(cache_present),
                "last_check": now_iso,
                "next_check": now_ts + _refresh_interval_seconds(),
            }
        )
        return dict(_STATE)


def license_status() -> dict[str, Any]:
    state = refresh_license(force=False)
    exp_ts = state.get("exp")
    return {
        "mode": state.get("mode"),
        "valid": bool(state.get("valid")),
        "reason": state.get("reason"),
        "exp": _iso_from_ts(exp_ts),
        "cache_present": bool(state.get("cache_present")),
        "last_check": state.get("last_check"),
        "next_check": _iso_from_ts(state.get("next_check")),
    }


def license_allows_request(path: str) -> tuple[bool, str | None]:
    state = refresh_license(force=False)
    mode = str(state.get("mode", "soft"))
    if mode == "off":
        return True, None

    if state.get("valid"):
        reason = state.get("reason")
        if mode == "soft" and reason:
            return True, str(reason)
        return True, None

    reason = str(state.get("reason") or "license_invalid")
    if mode == "soft":
        return True, reason
    return False, reason
