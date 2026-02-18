from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.append(str(API_ROOT))


def _import_app(monkeypatch: pytest.MonkeyPatch):
    # Ensure licensing cannot block the chat endpoint during tests.
    monkeypatch.setenv("LICENSE_ENFORCEMENT_MODE", "off")
    monkeypatch.setenv("LICENSE_SERVER_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("LICENSE_INSTANCE_ID", "test-instance")
    monkeypatch.setenv("LICENSE_CACHE_PATH", str(API_ROOT / ".." / ".." / "runtime" / "license_cache_test.json"))
    # Keep auth deterministic for UI/API tests that may hit /api/auth/verify.
    monkeypatch.setenv("KAI_ACCESS_PASSWORD", "testpw")

    import importlib

    # Import after env is set so module-level config reads the right values.
    main = importlib.import_module("main")
    return main.app


def test_send_perf_action_without_tool_output_is_fast_and_non_hallucinatory(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Regression guard:
    - When a user asks for performance actions WITHOUT tool_output, but an account is already selected,
      the system should NOT call an LLM (latency + hallucination risk).
    - Instead, it should return a deterministic, advisor-grade response that mentions key levers and
      avoids implying SA360 is disconnected.
    """
    app = _import_app(monkeypatch)
    client = TestClient(app)

    session_id = "test-session-perf-advice"

    # Seed a default account so `has_account_context` is true without requiring SA360 OAuth.
    save = client.post(
        "/api/sa360/default-account",
        json={
            "session_id": session_id,
            "customer_id": "7902313748",
            "account_name": "Havas_Shell_GoogleAds_US_Mobility Loyalty",
        },
    )
    assert save.status_code == 200

    msg = "Based on the current account context, give me 3 levers to improve performance. Keep it concise and actionable."
    t0 = time.perf_counter()
    resp = client.post(
        "/api/chat/send",
        json={
            "message": msg,
            "ai_enabled": True,
            "session_id": session_id,
        },
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("role") == "assistant"
    assert body.get("model") == "rules_fast_performance_advice"

    reply = (body.get("reply") or "").lower()
    # Intent-to-goal: must mention multiple optimization levers (mirrors spec intent goals).
    must_have_any = ("budget", "bid", "bidding", "keyword", "creative", "landing", "cpa", "cpc", "ctr")
    hits = [k for k in must_have_any if k in reply]
    assert len(hits) >= 2

    # Quality: do not falsely claim SA360 is disconnected.
    assert "not connected" not in reply
    assert "has not been connected" not in reply

    # Performance: should be fast (no LLM). Keep threshold generous for cold-start.
    assert elapsed_ms < 2000, f"expected fast deterministic reply; got {elapsed_ms:.1f}ms"

