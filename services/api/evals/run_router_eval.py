from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

import requests


def _bool_env(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "y"}:
        return True
    if raw in {"0", "false", "no", "n"}:
        return False
    return default


def _load_cases(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return payload["cases"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Invalid router eval file format.")


def _compare_expected(actual: Dict[str, Any], expected: Dict[str, Any]) -> list[str]:
    mismatches = []
    for key, value in expected.items():
        if actual.get(key) != value:
            mismatches.append(f"{key}: expected {value}, got {actual.get(key)}")
    return mismatches


def main() -> int:
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
    cases_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("evals", "router_cases.json")
    strict = _bool_env("EVAL_STRICT", True)

    cases = _load_cases(cases_path)
    results = []
    failures = 0

    for case in cases:
        name = case.get("name", "unnamed_case")
        message = case.get("message", "")
        expected = case.get("expected", {})
        payload = {
            "message": message,
            "account_name": case.get("account_name"),
            "customer_ids": case.get("customer_ids") or [],
        }
        start = time.time()
        try:
            resp = requests.post(f"{backend_url}/api/chat/route", json=payload, timeout=30)
        except Exception as exc:
            failures += 1
            results.append(
                {"name": name, "status": "fail", "detail": f"request_error: {exc}"}
            )
            continue
        duration_ms = int((time.time() - start) * 1000)
        if resp.status_code < 200 or resp.status_code >= 300:
            failures += 1
            results.append(
                {
                    "name": name,
                    "status": "fail",
                    "detail": f"status={resp.status_code} body={resp.text[:400]}",
                }
            )
            continue

        body = resp.json()
        mismatches = _compare_expected(body, expected)
        status = "pass" if not mismatches else "fail"
        if status == "fail":
            failures += 1
        results.append(
            {
                "name": name,
                "status": status,
                "duration_ms": duration_ms,
                "mismatches": mismatches,
            }
        )

    summary = {
        "backend_url": backend_url,
        "cases": len(results),
        "failures": failures,
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    if failures and strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
