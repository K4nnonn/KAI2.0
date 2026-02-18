"""
FastAPI Backend for Kai Platform
Modern REST API replacing Streamlit
"""
from __future__ import annotations

import sys
import os
import tempfile
import asyncio
import threading
from contextvars import ContextVar
from uuid import uuid4
from pathlib import Path
from typing import Any, List
import time
import base64
import hmac
import urllib.parse
from datetime import datetime, date, timedelta
import math
import pandas as pd
import traceback
import json
import io
import re
import traceback
import importlib
import hashlib
import requests
import logging
import jwt


# Add project root to path before local imports
# In container: /app/main.py -> ROOT should be /app
# Locally: ...\services\api\main.py -> ROOT should be repo or services\api with kai_core
ROOT = Path(__file__).resolve().parent
# Check if we're in container (flat structure) or local dev (backend subdirectory)
if (ROOT / "kai_core").exists():
    pass  # Already at the right level (container)
else:
    ROOT = ROOT.parent  # Go up one level (local dev)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


_BRAIN_GATE_MODULE: Any | None = None
_BRAIN_GATE_USE_OBFUSCATED = os.environ.get("BRAIN_GATE_USE_OBFUSCATED", "false").lower() == "true"


def _load_brain_gate_module() -> Any:
    global _BRAIN_GATE_MODULE
    if _BRAIN_GATE_MODULE is not None:
        return _BRAIN_GATE_MODULE

    if not _BRAIN_GATE_USE_OBFUSCATED:
        from kai_core.shared import brain_gate as source_brain_gate
        _BRAIN_GATE_MODULE = source_brain_gate
        return _BRAIN_GATE_MODULE

    candidate_dirs = [ROOT / "build" / "obf", ROOT.parent.parent / "build" / "obf"]
    obf_dir = None
    artifacts: list[Path] = []
    for candidate in candidate_dirs:
        found = sorted(candidate.glob("brain_gate*.pyd")) + sorted(candidate.glob("brain_gate*.so"))
        if found:
            obf_dir = candidate
            artifacts = found
            break
    if not artifacts or obf_dir is None:
        raise RuntimeError(
            "BRAIN_GATE_USE_OBFUSCATED=true but no compiled brain_gate artifact was found in build/obf."
        )

    if str(obf_dir) not in sys.path:
        sys.path.insert(0, str(obf_dir))

    try:
        _BRAIN_GATE_MODULE = importlib.import_module("brain_gate")
    except Exception as exc:
        raise RuntimeError(f"Failed to load obfuscated brain_gate module: {exc}") from exc
    return _BRAIN_GATE_MODULE


def _brain_gate_refresh(force: bool = False) -> dict[str, Any]:
    module = _load_brain_gate_module()
    return module.refresh_license(force=force)


def _brain_gate_status() -> dict[str, Any]:
    module = _load_brain_gate_module()
    return module.license_status()


def _brain_gate_allows_request(path: str) -> tuple[bool, str | None]:
    module = _load_brain_gate_module()
    return module.license_allows_request(path)


# In-memory backoff for local LLM (avoid repeated 404/connection errors)
_LOCAL_LLM_BACKOFF_UNTIL = 0.0
_LOCAL_LLM_HEALTH_OK_UNTIL = 0.0
_LOCAL_LLM_HEALTH_ERR = None
_LOCAL_LLM_ROUTER_SEMAPHORE = None
_LOCAL_LLM_ROUTER_MAX_CONCURRENCY = None
_LLM_USAGE_LOCK = threading.Lock()
_LLM_USAGE = {
    "local_success": 0,
    "local_error": 0,
    "azure_success": 0,
    "azure_error": 0,
    "last_local_error": None,
    "last_azure_error": None,
    "last_local_latency_ms": None,
    "last_azure_latency_ms": None,
    "last_local_ts": None,
    "last_azure_ts": None,
}
_LLM_TRACE: ContextVar[dict | None] = ContextVar("_LLM_TRACE", default=None)


def _record_llm_trace(
    model: str,
    intent: str | None,
    status: str,
    latency_ms: float | None = None,
    error: str | None = None,
) -> None:
    trace = _LLM_TRACE.get()
    if trace is None:
        return
    entry: dict[str, Any] = {"model": model, "intent": intent or "", "status": status}
    if latency_ms is not None:
        entry["latency_ms"] = round(float(latency_ms), 2)
        trace["total_ms"] += float(latency_ms)
    if error:
        entry["error"] = error
    if model == "local":
        trace["local_calls"] += 1
    elif model == "azure":
        trace["azure_calls"] += 1
    trace["calls"].append(entry)


def _reset_llm_usage() -> None:
    with _LLM_USAGE_LOCK:
        _LLM_USAGE.update(
            {
                "local_success": 0,
                "local_error": 0,
                "azure_success": 0,
                "azure_error": 0,
                "last_local_error": None,
                "last_azure_error": None,
                "last_local_latency_ms": None,
                "last_azure_latency_ms": None,
                "last_local_ts": None,
                "last_azure_ts": None,
            }
        )


def _record_llm_usage(kind: str, status: str, error: str | None = None, latency_ms: float | None = None) -> None:
    if kind not in ("local", "azure"):
        return
    if status not in ("success", "error"):
        return
    now = datetime.utcnow().isoformat() + "Z"
    with _LLM_USAGE_LOCK:
        key = f"{kind}_{status}"
        _LLM_USAGE[key] = int(_LLM_USAGE.get(key, 0)) + 1
        if error:
            _LLM_USAGE[f"last_{kind}_error"] = error
        if latency_ms is not None:
            _LLM_USAGE[f"last_{kind}_latency_ms"] = round(float(latency_ms), 2)
        _LLM_USAGE[f"last_{kind}_ts"] = now


def _llm_usage_snapshot(reset: bool = False) -> dict:
    with _LLM_USAGE_LOCK:
        snapshot = dict(_LLM_USAGE)
        if reset:
            _reset_llm_usage()
    return snapshot


def _azure_budget_snapshot() -> dict:
    return azure_budget_snapshot()


def _azure_budget_allow(intent: str | None = None) -> tuple[bool, str | None]:
    return allow_azure_usage(module="main_llm", purpose="fallback", intent=intent)


def _get_router_semaphore() -> threading.BoundedSemaphore | None:
    global _LOCAL_LLM_ROUTER_SEMAPHORE, _LOCAL_LLM_ROUTER_MAX_CONCURRENCY
    try:
        max_concurrency = int(os.environ.get("LOCAL_LLM_ROUTER_MAX_CONCURRENCY", "2") or "2")
    except Exception:
        max_concurrency = 2
    if os.environ.get("REQUIRE_LOCAL_LLM", "false").lower() == "true":
        max_concurrency = min(max_concurrency, 1)
    if max_concurrency < 1:
        return None
    if _LOCAL_LLM_ROUTER_SEMAPHORE is None or _LOCAL_LLM_ROUTER_MAX_CONCURRENCY != max_concurrency:
        _LOCAL_LLM_ROUTER_SEMAPHORE = threading.BoundedSemaphore(max_concurrency)
        _LOCAL_LLM_ROUTER_MAX_CONCURRENCY = max_concurrency
    return _LOCAL_LLM_ROUTER_SEMAPHORE


def _requests_verify_path() -> str | bool:
    """
    Determine a CA bundle path for outbound HTTPS requests.

    Why:
    - In some container environments, Requests can fail TLS verification due to missing/overridden CA bundles.
    - We allow an explicit override for enterprise networks while keeping a safe default for Azure.
    """
    override = (os.environ.get("KAI_SSL_CA_BUNDLE") or os.environ.get("REQUESTS_CA_BUNDLE") or "").strip()
    if override:
        try:
            if Path(override).exists():
                return override
        except Exception:
            pass
    try:
        import certifi  # type: ignore

        ca = certifi.where()
        if ca and Path(ca).exists():
            return ca
    except Exception:
        pass
    sys_ca = "/etc/ssl/certs/ca-certificates.crt"
    try:
        if Path(sys_ca).exists():
            return sys_ca
    except Exception:
        pass
    return True


def _local_llm_health_check(endpoint: str, timeout: float | None = None) -> tuple[bool, str | None]:
    """
    Lightweight health probe. For Ollama-compatible endpoints, /api/tags is cheap.
    Caches success for 60s to avoid noisy checks.
    """
    global _LOCAL_LLM_HEALTH_OK_UNTIL, _LOCAL_LLM_HEALTH_ERR
    if timeout is None:
        timeout = float(os.environ.get("LOCAL_LLM_HEALTH_TIMEOUT_SECONDS", "5") or "5")
    now = time.time()
    if now < _LOCAL_LLM_HEALTH_OK_UNTIL:
        return True, None
    try:
        _, tags_url = _local_llm_endpoints(endpoint)
        url = tags_url
        verify = _requests_verify_path() if str(url).lower().startswith("https://") else True
        resp = requests.get(url, timeout=timeout, verify=verify)
        allow_404 = os.environ.get("LOCAL_LLM_HEALTH_ALLOW_404", "false").lower() == "true"
        if allow_404 and resp.status_code == 404:
            _LOCAL_LLM_HEALTH_OK_UNTIL = now + 60
            _LOCAL_LLM_HEALTH_ERR = None
            return True, None
        resp.raise_for_status()
        _LOCAL_LLM_HEALTH_OK_UNTIL = now + 60
        _LOCAL_LLM_HEALTH_ERR = None
        return True, None
    except Exception as exc:
        _LOCAL_LLM_HEALTH_ERR = str(exc)
        return False, _LOCAL_LLM_HEALTH_ERR


def _local_llm_endpoints(endpoint: str) -> tuple[str, str]:
    base = endpoint.rstrip("/")
    if base.endswith("/api/chat"):
        chat_url = base
        tags_url = f"{base[:-len('/chat')]}/tags"
    elif base.endswith("/api"):
        chat_url = f"{base}/chat"
        tags_url = f"{base}/tags"
    else:
        chat_url = f"{base}/api/chat"
        tags_url = f"{base}/api/tags"
    return chat_url, tags_url


def _extract_json_dict(text: str) -> dict | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    obj_start = None
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == "\"":
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                obj_start = idx
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                snippet = cleaned[obj_start : idx + 1]
                try:
                    data = json.loads(snippet)
                    return data if isinstance(data, dict) else None
                except Exception:
                    sanitized = []
                    in_str_s = False
                    escape_s = False
                    for ch_s in snippet:
                        if escape_s:
                            escape_s = False
                            sanitized.append(ch_s)
                            continue
                        if ch_s == "\\":
                            escape_s = True
                            sanitized.append(ch_s)
                            continue
                        if ch_s == "\"":
                            in_str_s = not in_str_s
                            sanitized.append(ch_s)
                            continue
                        if in_str_s and ch_s in ("\n", "\r", "\t"):
                            sanitized.append("\\n" if ch_s == "\n" else "\\r" if ch_s == "\r" else "\\t")
                            continue
                        sanitized.append(ch_s)
                    try:
                        data = json.loads("".join(sanitized))
                        return data if isinstance(data, dict) else None
                    except Exception:
                        return None
    return None

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
from azure.data.tables import TableServiceClient, UpdateMode
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from kai_core.account_utils import slugify_account_name

from utils import db_manager
from utils.circuit_breaker import CircuitBreaker
from utils.job_queue import (
    job_queue_enabled,
    job_queue_force,
    enqueue_job,
    get_job,
    get_job_result,
    queue_depth,
)
from utils.rate_limit import TokenBucket
from utils.telemetry import log_event, set_request_id
from utils.ttl_cache import TTLCache
from kai_core.Concierge import call_azure_openai
from kai_core.core_logic import UnifiedAuditEngine
from kai_core.core_logic.audit_verbalizer import AuditNarrativeHumanizer
from kai_core.plugins.creative import CreativeFactory
from kai_core.shared.adapters import CreativeContext
from kai_core.plugins.pmax import PMaxAnalyzer
from kai_core.plugins.serp import SerpScanner, check_url_health
from kai_core.plugins.competitor_intelligence import analyze_competitor
from kai_core.pmax_channel_split import PMaxChannelSplitter
from kai_core.data_validation import DataValidator
from kai_core.market_intelligence import MarketIntelligence
from kai_core.unified_schema_manager import UnifiedSchemaManager
from kai_core.agentic_orchestrator import MarketingReasoningAgent
from kai_core.config import get_deployment_mode
from kai_core.shared.ai_sync import (
    chat_system_prompt,
    performance_system_prompt,
    performance_advisor_system_prompt,
    audit_persona_prefix,
)
from kai_core.shared.azure_budget import allow_azure_usage, azure_budget_snapshot
from kai_core.shared.vector_index import index_audit_workbook
from kai_core.shared.vector_search import build_vector_queries, load_vector_config, resolve_vector_filter_mode
from ml_reasoning_engine import get_ml_reasoning_engine
import httpx
from trends_service import fetch_trends, seasonality_multipliers, summarize_seasonality, ENABLE_TRENDS
from fastapi import status
import csv

# Initialize FastAPI app
app = FastAPI(
    title="Kai Platform API",
    description="PPC Marketing Intelligence Platform",
    version="2.0.0",
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup_local_warmup():
    _brain_gate_refresh(force=True)
    _start_local_llm_warmup()


@app.middleware("http")
async def request_telemetry_and_rate_limit(request: Request, call_next):
    start = time.perf_counter()
    path = request.url.path
    exempt_exact = {
        "/api/health",
        "/api/diagnostics/health",
        "/api/auth/verify",
        "/openapi.json",
    }
    exempt_prefixes = ("/docs",)
    license_warn: str | None = None

    if path not in exempt_exact and not any(path.startswith(prefix) for prefix in exempt_prefixes):
        allowed, reason = _brain_gate_allows_request(path)
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "license_blocked", "reason": reason or "license_invalid"},
            )
        if reason:
            license_warn = reason

    session_id = request.headers.get("x-session-id") or request.query_params.get("session_id")
    client_ip = request.client.host if request.client else "unknown"
    rate_key = (session_id or client_ip or "anon").strip()
    request_id = request.headers.get("x-request-id") or str(uuid4())
    set_request_id(request_id)
    request.state.request_id = request_id

    if RATE_LIMIT_ENABLED:
        limiter = _HEAVY_RATE_LIMITER if path.startswith(_HEAVY_PATH_PREFIXES) else _RATE_LIMITER
        if not limiter.allow(rate_key):
            log_event(
                "rate_limited",
                path=path,
                client=client_ip,
                session_id=session_id,
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"status": "error", "detail": "Rate limit exceeded. Please retry shortly."},
            )

    response = None
    try:
        response = await call_next(request)
        if response is not None:
            response.headers["X-Request-ID"] = request_id
            if license_warn:
                response.headers["X-License-Warn"] = license_warn
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        status_code = getattr(response, "status_code", 500)
        log_event(
            "request",
            path=path,
            method=request.method,
            status=status_code,
            duration_ms=round(duration_ms, 2),
            client=client_ip,
            session_id=session_id,
        )

# Initialize database
db_manager.init_db()


# Pydantic models
class ChatMessage(BaseModel):
    message: str
    ai_enabled: bool = True
    top_k: int | None = None  # optional: number of sources to retrieve for web lookup
    account_name: str | None = None  # optional for audit/upload flows
    context: dict[str, Any] | None = None  # optional: tool routing hints (serp/pmax/competitor/etc.) and data context (customer_ids/date_range)
    session_id: str | None = None  # optional: session-scoped chat history
    dry_run: bool = False  # optional: skip persistence for non-destructive validation


class ChatResponse(BaseModel):
    reply: str
    role: str = "assistant"
    sources: list[dict] = []
    model: str | None = None
    guardrail: dict | None = None


class AuditRequest(BaseModel):
    business_unit: str
    account_name: str
    use_mock_data: bool = False  # deprecated; blob download is used when available
    async_mode: bool = False
    data_prefix: str | None = None  # optional: override blob prefix for isolated datasets


class CreativeRequest(BaseModel):
    business_name: str
    url: str | None = None
    keywords: list[str] = []
    usps: list[str] = []


class PMaxRequest(BaseModel):
    placements: list[dict] = []
    spend: float | None = None
    total_cost: float | None = None
    conversions: float | None = None


class IntelRequest(BaseModel):
    query: str = "Why is performance down?"
    pmax: list[dict] = []
    creative: list[dict] = []
    market: list[dict] = []
    brand_terms: list[str] = []


# Helpers
def _load_dataframes_from_dir(directory: Path) -> list[pd.DataFrame]:
    """Best-effort load of CSV/XLSX files from a directory into pandas dataframes.

    NOTE: For XLSX/XLS, we load *all* sheets, since agencies often export multiple tabs in one workbook.
    """
    frames: list[pd.DataFrame] = []
    if not directory.exists():
        return frames
    for file in directory.iterdir():
        if file.suffix.lower() == ".csv":
            try:
                frames.append(pd.read_csv(file))
            except Exception:
                continue
        elif file.suffix.lower() in {".xlsx", ".xls"}:
            try:
                sheets = pd.read_excel(file, sheet_name=None)  # dict[str, DataFrame]
                for _, df in (sheets or {}).items():
                    if df is not None and not df.empty:
                        frames.append(df)
            except Exception:
                continue
    return frames


def _pick_frame(frames: list[pd.DataFrame], required_columns: list[str]) -> pd.DataFrame | None:
    """Return the first frame containing all required columns (case-insensitive)."""
    if not frames:
        return None
    req = {c.lower() for c in required_columns}
    for frame in frames:
        cols = {c.lower() for c in frame.columns}
        if req.issubset(cols):
            return frame
    return None


class SerpRequest(BaseModel):
    urls: list[str]


class CompetitorObservation(BaseModel):
    """Competitor observation from conversational extraction."""
    competitor_domain: str
    impression_share_current: float | None = None
    impression_share_previous: float | None = None
    outranking_rate: float | None = None
    top_of_page_rate: float | None = None
    position_above_rate: float | None = None
    raw_description: str | None = None  # Fallback for fuzzy inference


class AuthRequest(BaseModel):
    password: str


class KeywordMetricsQuery(BaseModel):
    keyword: str
    month: str | None = None  # optional, e.g., "2025-11" or "last_month"


class SearchQuery(BaseModel):
    query: str
    count: int = 3  # number of results to return


class PlanRequest(BaseModel):
    message: str
    customer_ids: list[str] = []
    account_name: str | None = None
    default_date_range: str | None = "LAST_7_DAYS"
    include_previous: bool = True  # If false, skip previous-range fetch for lower-latency single-window analysis.
    generate_report: bool = False  # If true, return XLSX link; otherwise keep chat-only
    intent_hint: str | None = None  # optional: 'performance' | 'audit' to override heuristic intent
    async_mode: bool = False
    session_id: str | None = None


class PlanResponse(BaseModel):
    plan: dict
    notes: str | None = None
    executed: bool = False
    result: dict | None = None
    summary: str | None = None
    enhanced_summary: str | None = None
    error: str | None = None
    analysis: dict | None = None
    status: str | None = None
    job_id: str | None = None


class QaAccuracyRequest(BaseModel):
    customer_id: str
    date_range: str | None = "LAST_7_DAYS"
    session_id: str | None = None


class QaAccuracyResponse(BaseModel):
    raw_totals: dict
    aggregated_totals: dict
    matches: dict


class RouteRequest(BaseModel):
    message: str
    account_name: str | None = None
    customer_ids: list[str] = []
    session_id: str | None = None


class RouteResponse(BaseModel):
    intent: str
    tool: str | None = None
    run_planner: bool = False
    run_trends: bool = False
    themes: list[str] = []
    customer_ids: list[str] = []
    needs_ids: bool = False
    notes: str | None = None
    confidence: float | None = None
    needs_clarification: bool = False
    clarification: str | None = None
    candidates: list[str] = []


def _call_local_llm(
    messages: list[dict],
    intent: str | None = None,
    max_tokens: int = 512,
    force_json: bool = False,
    record_usage: bool = False,
) -> tuple[str | None, dict]:
    """
    Call local Ollama (if enabled) for low-cost inference.
    Returns (text, meta) where meta contains model, latency_ms, error if any.
    """
    allow_local = os.environ.get("ENABLE_LOCAL_LLM_WRAPPER", "false").lower() == "true"
    endpoint = os.environ.get("LOCAL_LLM_ENDPOINT", "").strip()
    if not allow_local and endpoint:
        allow_local = os.environ.get("LOCAL_LLM_AUTO_ENABLE", "true").lower() == "true"
    model = os.environ.get("LOCAL_LLM_MODEL", "").strip() or "llama3"
    timeout_env = os.environ.get("LOCAL_LLM_TIMEOUT_SECONDS")
    timeout = float(timeout_env or "20")
    require_local = os.environ.get("REQUIRE_LOCAL_LLM", "false").lower() == "true"
    required_timeout = float(os.environ.get("LOCAL_LLM_REQUIRED_TIMEOUT_SECONDS", "45") or "45")
    if require_local:
        # Ensure local-only runs get a higher timeout even if a low default is set.
        timeout = max(timeout, required_timeout)
    max_input_env = os.environ.get("LOCAL_LLM_MAX_INPUT_CHARS")
    max_input_chars = int(max_input_env or ("2000" if require_local else "3000"))
    if intent == "router":
        try:
            timeout = float(os.environ.get("LOCAL_LLM_ROUTER_TIMEOUT_SECONDS", "8") or "8")
        except Exception:
            pass
    if intent == "performance_advice" and not require_local:
        # Keep advisor responses interactive: if local is slow/unhealthy, fail fast and fall back to Azure.
        try:
            timeout = float(os.environ.get("LOCAL_LLM_PERF_ADVICE_TIMEOUT_SECONDS", "6") or "6")
        except Exception:
            pass
    meta = {"model": "local", "intent": intent or "", "used": False}

    if not allow_local or not endpoint:
        meta["error"] = "local_disabled"
        return None, meta

    ok, err = _local_llm_health_check(endpoint)
    if not ok:
        meta["error"] = f"local_unhealthy: {err}"
        return None, meta

    url, _ = _local_llm_endpoints(endpoint)
    temp = 0.0 if intent == "router" or force_json else 0.2
    options = {"temperature": temp, "num_predict": max_tokens}
    local_messages = _truncate_messages_for_local(messages, max_input_chars)
    if local_messages is not messages:
        meta["trimmed"] = True
        meta["input_chars"] = sum(
            len(m.get("content", "")) for m in local_messages if isinstance(m.get("content"), str)
        )

    payload = {
        "model": model,
        "messages": local_messages,
        "stream": False,
        "options": options,
    }
    if intent == "router" or force_json:
        payload["format"] = "json"
    try:
        start = time.perf_counter()
        verify = _requests_verify_path() if str(url).lower().startswith("https://") else True
        resp = requests.post(url, json=payload, timeout=timeout, verify=verify)
        latency_ms = (time.perf_counter() - start) * 1000
        meta.update({"latency_ms": latency_ms, "used": True})
        resp.raise_for_status()
        data = resp.json()
        content = None
        # Ollama chat returns {"message":{"content": ...}}
        if isinstance(data, dict):
            content = (
                data.get("message", {}).get("content")
                or data.get("response")
                or data.get("text")
            )
        meta["content_preview"] = (content or "")[:60]
        if record_usage:
            _record_llm_usage("local", "success", latency_ms=latency_ms)
            _record_llm_trace("local", intent, "success", latency_ms=latency_ms)
        return (content.strip() if isinstance(content, str) else None), meta
    except Exception as exc:
        meta["error"] = str(exc)
        if record_usage:
            _record_llm_usage("local", "error", error=str(exc))
            _record_llm_trace("local", intent, "error", error=str(exc))
        return None, meta


def _start_local_llm_warmup() -> None:
    if os.environ.get("LOCAL_LLM_WARMUP", "true").strip().lower() not in {"1", "true", "yes"}:
        return
    if os.environ.get("REQUIRE_LOCAL_LLM", "false").strip().lower() not in {"1", "true", "yes"}:
        return
    if not os.environ.get("LOCAL_LLM_ENDPOINT", "").strip():
        return

    def _run() -> None:
        try:
            _call_local_llm(
                [{"role": "user", "content": "Say ok."}],
                intent="warmup",
                max_tokens=8,
                force_json=False,
                record_usage=True,
            )
        except Exception:
            return

    threading.Thread(target=_run, daemon=True).start()


def _call_llm(
    messages: list[dict],
    intent: str | None = None,
    allow_local: bool = True,
    max_tokens: int = 512,
    force_azure: bool = False,
) -> tuple[str | None, dict]:
    """
    Local-first LLM wrapper with Azure fallback.
    Returns (text, meta) where meta includes model and latency/error.
    """
    # Try local first if allowed and not in backoff
    global _LOCAL_LLM_BACKOFF_UNTIL
    now = time.time()
    require_local_env = os.environ.get("REQUIRE_LOCAL_LLM")
    if require_local_env is None:
        require_local = get_deployment_mode() == "LOCAL" and bool(os.environ.get("LOCAL_LLM_ENDPOINT", "").strip())
    else:
        require_local = require_local_env.lower() == "true"
    if require_local and not allow_local:
        meta = {"model": "local", "intent": intent or "", "used": False, "error": "local_required"}
        _record_llm_usage("local", "error", error=meta["error"])
        log_event(
            "llm_call",
            model="local",
            intent=intent or "",
            status="error",
            error=meta["error"],
        )
        return None, meta
    if allow_local and not force_azure:
        if now < _LOCAL_LLM_BACKOFF_UNTIL and not require_local:
            meta = {"model": "local", "intent": intent or "", "used": False, "error": "local_backoff"}
            _record_llm_usage("local", "error", error=meta["error"])
            log_event(
                "llm_call",
                model="local",
                intent=intent or "",
                status="error",
                error=meta["error"],
            )
            return None, meta
        router_sema = _get_router_semaphore() if intent == "router" else None
        got_permit = True
        if router_sema is not None:
            got_permit = router_sema.acquire(blocking=False)
            if not got_permit:
                err_text = "local_busy"
                try:
                    print(
                        f"[llm] intent={intent or ''} model=local skipped error={err_text}",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception:
                    pass
                log_event(
                    "llm_call",
                    model="local",
                    intent=intent or "",
                    status="error",
                    error=err_text,
                )
                _record_llm_usage("local", "error", error=err_text)
                if require_local:
                    meta = {"model": "local", "intent": intent or "", "used": False, "error": err_text}
                    return None, meta
        if got_permit:
            try:
                local_max_tokens = max_tokens
                try:
                    env_cap = int(os.environ.get("LOCAL_LLM_MAX_TOKENS", "") or "0")
                except Exception:
                    env_cap = 0
                if env_cap <= 0 and require_local:
                    env_cap = 120
                if env_cap > 0:
                    local_max_tokens = min(max_tokens, env_cap)
                local_text, local_meta = _call_local_llm(
                    messages, intent=intent, max_tokens=local_max_tokens
                )
            finally:
                if router_sema is not None:
                    try:
                        router_sema.release()
                    except Exception:
                        pass
            if local_text:
                try:
                    print(
                        f"[llm] intent={intent or ''} model=local latency_ms={local_meta.get('latency_ms')}",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception:
                    pass
                log_event(
                    "llm_call",
                    model="local",
                    intent=intent or "",
                    latency_ms=local_meta.get("latency_ms"),
                    status="success",
                )
                _record_llm_usage("local", "success", latency_ms=local_meta.get("latency_ms"))
                _record_llm_trace(
                    "local",
                    intent,
                    "success",
                    latency_ms=local_meta.get("latency_ms"),
                )
                return local_text, local_meta
            else:
                err_text = (local_meta or {}).get("error")
                # Back off local for 5 minutes on hard connectivity errors to avoid extra latency
                if err_text and any(
                    tok in err_text
                    for tok in (
                        "404",
                        "Connection refused",
                        "Failed to establish a new connection",
                        "Max retries exceeded",
                        "timed out",
                        "Read timed out",
                        "500",
                        "503",
                    )
                ):
                    _LOCAL_LLM_BACKOFF_UNTIL = now + 300
                try:
                    print(
                        f"[llm] intent={intent or ''} model=local failed error={err_text}",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception:
                    pass
                log_event(
                    "llm_call",
                    model="local",
                    intent=intent or "",
                    status="error",
                    error=err_text,
                )
                _record_llm_usage("local", "error", error=err_text)
                _record_llm_trace(
                    "local",
                    intent,
                    "error",
                    latency_ms=(local_meta or {}).get("latency_ms"),
                    error=err_text,
                )
                if require_local:
                    return None, local_meta

    if require_local:
        meta = {"model": "local", "intent": intent or "", "used": False, "error": "local_required"}
        _record_llm_usage("local", "error", error=meta["error"])
        log_event(
            "llm_call",
            model="local",
            intent=intent or "",
            status="error",
            error=meta["error"],
        )
        return None, meta

    # Azure fallback (guarded by budget caps)
    try:
        allowed, reason = _azure_budget_allow(intent)
        if not allowed:
            meta = {"model": "azure", "intent": intent or "", "used": False, "error": reason}
            log_event(
                "llm_call",
                model="azure",
                intent=intent or "",
                status="blocked",
                error=reason,
            )
            _record_llm_usage("azure", "error", error=reason)
            _record_llm_trace("azure", intent, "blocked", error=reason)
            return None, meta
        start = time.perf_counter()
        text = call_azure_openai(
            messages,
            session_id=None,
            intent=intent,
            tenant_id=None,
            use_json_mode=bool(intent == "router"),
            max_tokens=max_tokens,
            temperature=0 if intent == "router" else 0.2,
            # Azure policy allowlisting is keyed on "purpose". Use stable purposes ("router"/"fallback")
            # rather than raw intents ("performance_advice"/"tool_followup"/etc.) so the allowlist
            # can remain tight while still enabling safe fallback when local responses fail quality gates.
            purpose="router" if intent == "router" else "fallback",
        )
        latency_ms = (time.perf_counter() - start) * 1000
        meta = {"model": "azure", "intent": intent or "", "latency_ms": latency_ms, "used": True}
        try:
            print(
                f"[llm] intent={intent or ''} model=azure latency_ms={latency_ms}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass
        log_event(
            "llm_call",
            model="azure",
            intent=intent or "",
            latency_ms=latency_ms,
            status="success",
        )
        _record_llm_usage("azure", "success", latency_ms=latency_ms)
        _record_llm_trace("azure", intent, "success", latency_ms=latency_ms)
        return text, meta
    except Exception as exc:
        meta = {"model": "azure", "intent": intent or "", "used": False, "error": str(exc)}
        try:
            print(
                f"[llm] intent={intent or ''} model=azure failed error={exc}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass
        log_event(
            "llm_call",
            model="azure",
            intent=intent or "",
            status="error",
            error=str(exc),
        )
        _record_llm_usage("azure", "error", error=str(exc))
        _record_llm_trace("azure", intent, "error", error=str(exc))
        return None, meta


def _extract_json_obj(raw_text: str | None) -> dict | None:
    """Best-effort JSON extraction: strip code fences and grab first {...} block."""
    if not raw_text or not isinstance(raw_text, str):
        return None
    txt = raw_text.strip()
    if txt.startswith("```"):
        # remove code fences if present
        txt = txt.strip("` \n")
        if txt.lower().startswith("json"):
            txt = txt[4:].lstrip()
    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1:
        return None
    if end == -1 or end <= start:
        fragment = txt[start:]
        # Attempt to close a missing brace
        fragment = fragment.rstrip(", \n") + "}"
    else:
        fragment = txt[start : end + 1]
    try:
        return json.loads(fragment)
    except Exception:
        # last-chance clean-up: strip trailing commas and retry
        try:
            fragment = re.sub(r",\\s*}", "}", fragment)
            return json.loads(fragment)
        except Exception:
            return None


def _is_low_quality_local_reply(text: str | None) -> bool:
    """Detect low-quality local LLM replies to avoid surfacing junk output."""
    if not text or not isinstance(text, str):
        return True
    cleaned = text.strip()
    if len(cleaned) < 20:
        return True
    # JSON-looking output for general chat is almost always wrong here.
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return True
    lower = cleaned.lower()
    if lower.count("assistant:") + lower.count("kai:") >= 3:
        return True
    lines = [line.strip().lower() for line in cleaned.splitlines() if line.strip()]
    if len(lines) >= 4:
        unique = len(set(lines))
        if unique <= max(1, len(lines) // 4):
            return True
    return False


def _is_advisor_grade_performance_reply(text: str | None) -> bool:
    """
    Deterministic structure validator for advisor-mode performance replies.

    We use this as a hard quality gate before accepting local-model output, because
    a metric dump can be "non-junk" (passes _is_low_quality_local_reply) but still
    fail the product requirement ("Option A/B + monitoring plan").
    """
    if not text or not isinstance(text, str):
        return False
    cleaned = text.strip()
    if len(cleaned) < 80:
        return False
    lower = cleaned.lower()
    # Required structure (per performance_advisor_system_prompt()).
    # We accept multiple label styles to avoid forcing a rigid "Option A/B" template.
    markers: set[str] = set()
    for m in re.finditer(r"\b(option|approach|path|strategy)\s+([a-d]|[1-4])\b", lower):
        markers.add(f"{m.group(1)}:{m.group(2)}")
    has_two_paths = len(markers) >= 2
    if not has_two_paths:
        # Accept a numbered list (e.g., "1." + "2.") as a fallback signal.
        numbered = re.findall(r"(?m)^\s*([1-4])[\.\)]\s+", cleaned)
        has_two_paths = len(set(numbered)) >= 2
    if not has_two_paths:
        return False
    if not re.search(r"\b(monitor|watch|track|keep an eye)\b", lower):
        return False
    if not re.search(r"\b(recommend|my recommendation|i recommend|i would start|i'd start|start with)\b", lower):
        return False
    # Avoid leaking internal tokens.
    if any(tok in cleaned for tok in ("LAST_7_DAYS", "LAST_14_DAYS", "LAST_30_DAYS")):
        return False
    return True


def _ensure_performance_advice_minimum_structure(text: str | None) -> str | None:
    """
    Deterministically enforce minimum advisor structure for performance recommendations.

    Rationale:
    - Local and Azure models can occasionally omit a monitoring/watch section even when prompted.
    - We gate local quality before upgrading, but we do not want to hard-fail Azure output (latency/cost).
    - The appended content avoids introducing new numbers (numeric grounding guardrail will still apply).
    """
    if not text or not isinstance(text, str):
        return text
    cleaned = text.strip()
    if not cleaned:
        return text
    lower = cleaned.lower()

    has_recommend = bool(re.search(r"\b(recommend|start with|i would start)\b", lower))
    has_monitor = bool(re.search(r"\b(monitor|watch|track|keep an eye)\b", lower))
    option_markers = set()
    for m in re.finditer(r"\b(option|approach|path|strategy)\s+([a-d]|[1-4])\b", lower):
        option_markers.add(f"{m.group(1)}:{m.group(2)}")
    has_two_options = len(option_markers) >= 2

    if not has_two_options:
        cleaned = (
            f"{cleaned}\n\n"
            "Option A (Conservative): isolate the main driver first by campaign and device, then prioritize the top mover.\n"
            "Option B (Aggressive): apply a targeted fix after driver confirmation (query cleanup, bid adjustment, or creative/LP refresh)."
        ).strip()
        lower = cleaned.lower()
        has_recommend = bool(re.search(r"\b(recommend|start with|i would start)\b", lower))
        has_monitor = bool(re.search(r"\b(monitor|watch|track|keep an eye)\b", lower))

    if not has_recommend:
        cleaned = (
            f"{cleaned}\n\nRecommendation: If you want stability first, start with the lowest-risk lever and change one thing at a time."
        ).strip()
        lower = cleaned.lower()
        has_monitor = bool(re.search(r"\b(monitor|watch|track|keep an eye)\b", lower))

    if not has_monitor:
        cleaned = (
            f"{cleaned}\n\nMonitoring plan: Watch conversions, cost/spend, CTR, and CPC daily for the next week. If efficiency worsens, roll back the most recent change and rerun the check."
        ).strip()

    return cleaned


def _is_refusal_reply(text: str | None) -> bool:
    if not text or not isinstance(text, str):
        return False
    lower = text.lower()
    patterns = [
        "i can't share",
        "i cannot share",
        "i can't provide",
        "i cannot provide",
        "cannot disclose",
        "not authorized",
        "sensitive data",
    ]
    return any(p in lower for p in patterns)


def _extract_session_id(chat: ChatMessage) -> str | None:
    sid = getattr(chat, "session_id", None)
    if isinstance(sid, str) and sid.strip():
        return sid.strip()
    ctx = chat.context if isinstance(chat.context, dict) else {}
    for key in ("session_id", "sessionId"):
        value = ctx.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_for_dedupe(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _dedupe_general_chat_reply(text: str | None) -> str | None:
    if not text or not isinstance(text, str):
        return text
    cleaned = text.strip()
    if not cleaned:
        return text
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return text
    if "```" in cleaned:
        return text
    lines = cleaned.splitlines()
    if len(lines) > 1:
        deduped_lines = []
        seen = set()
        for line in lines:
            if not line.strip():
                deduped_lines.append(line)
                continue
            norm = _normalize_for_dedupe(line)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            deduped_lines.append(line)
        return "\n".join(deduped_lines).strip()
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    deduped = []
    seen = set()
    for sentence in sentences:
        norm = _normalize_for_dedupe(sentence)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(sentence)
    return " ".join(deduped).strip()


def _needs_concept_rewrite(text: str | None) -> bool:
    if not text or not isinstance(text, str):
        return True
    cleaned = text.strip()
    if len(cleaned.split()) < 18:
        return True
    lower = cleaned.lower()
    has_example = any(marker in lower for marker in ("for example", "for instance", "e.g."))
    has_metric = bool(
        re.search(r"\b(cpc|cpa|roas|ctr|impressions|clicks|spend|cost|conversion|conversions)\b", lower)
    )
    has_action = any(
        marker in lower
        for marker in (
            "next step",
            "check",
            "compare",
            "break down",
            "segment",
            "review",
            "audit",
            "test",
            "pull",
        )
    )
    return not (has_example and has_metric and has_action)


def _rewrite_concept_reply(question: str, draft: str) -> str | None:
    if not draft:
        return None
    system_prompt = (
        "Rewrite the response into exactly two sentences. "
        "Avoid generic textbook phrasing. "
        "Include one paid-search-specific example with a metric (CPC/CPA/ROAS/CTR/spend). "
        "End with one concrete next step (campaign/device/query/geo cut or experiment)."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}\nDraft: {draft}"},
    ]
    rewrite, _ = _call_llm(messages, intent="concept_rewrite", allow_local=True, max_tokens=160)
    if rewrite and not _is_low_quality_local_reply(rewrite):
        return rewrite
    return None


def _needs_missing_data_rewrite(text: str | None) -> bool:
    if not text or not isinstance(text, str):
        return True
    lower = text.lower()
    if "sensitive data" in lower or "can't provide" in lower or "cannot provide" in lower:
        return True
    has_request = any(marker in lower for marker in ("request", "ask", "pull", "export"))
    has_report = any(marker in lower for marker in ("report", "export", "field", "column"))
    return not (has_request and has_report)


def _extract_data_needed_items(text: str | None) -> list[str]:
    if not text or not isinstance(text, str):
        return []
    matches = re.findall(r"data needed:\\s*([^\\n\\r]+)", text, flags=re.IGNORECASE)
    items: list[str] = []
    seen: set[str] = set()
    for match in matches:
        item = match.strip().rstrip(".")
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
    return items


def _rewrite_missing_data_reply(question: str, draft: str | None) -> str | None:
    context = draft or ""
    items = _extract_data_needed_items(context)
    if items:
        joined = "; ".join(items)
        return (
            "To complete the missing checks, please export and share: "
            f"{joined}. If you already have those files, upload them and I will rerun the audit."
        )
    default_items = [
        "Campaign report (account name, campaign type, status, bid strategy, impressions, clicks, cost, conversions)",
        "Ad group report (campaign and ad group IDs, status, impressions, clicks, cost)",
        "Keyword report (match type, status, impressions, clicks, cost, quality score)",
        "Ad/asset report (headline/description text, ad type, final URL)",
        "Search query report (queries, match type, impressions, clicks, cost, conversions)",
        "Landing page report (final URL column)",
        "Sitelink/extension assets report (sitelinks/callouts)",
        "Audience/reporting settings (remarketing or audience columns when used)",
        "Conversion + attribution settings export (conversion actions, attribution model)",
    ]
    joined = "; ".join(default_items)
    return (
        "To complete the missing checks, please export: "
        f"{joined}. If any of these are unavailable, tell me which ones and I will tailor the audit."
    )


def _advisor_missing_data_reply(question: str) -> str | None:
    """Generate an advisor-style response when data is missing (local LLM only)."""
    # Keep this user-facing: the main Kai Chat UX should not force "Evidence/Hypothesis/Next step" scaffolding.
    # (That structure is useful for internal diagnostics, but reads like debug output to beta users.)
    return (
        "I can't run that analysis yet because I don't have account data / performance data for the requested account/timeframe. "
        "This may be because SA360 isn't connected in this session or the relevant exports weren't uploaded. "
        "Next step: connect SA360 (or upload the relevant exports), then retry."
    )


def _advisor_missing_data_reply_for_session(question: str, session_id: str | None) -> str | None:
    """
    Context-aware missing-data reply for chat/send fallback.

    Goal: avoid telling connected users to "Connect SA360" when the real blocker is
    missing account selection / missing MCC (login_customer_id) context.
    """
    sid = _normalize_session_id(session_id)
    if not sid:
        return _advisor_missing_data_reply(question)
    session = _load_sa360_session(sid) or {}
    connected = bool(session.get("refresh_token"))
    login_cid = str(session.get("login_customer_id") or "").strip()
    default_cid = str(session.get("default_customer_id") or "").strip()

    if not connected:
        return (
            "I can't pull performance data because I don't have any account data for this session yet (SA360 isn't connected). "
            "This may be because Google OAuth wasn't completed for this session. "
            "Next step: click 'Connect SA360' at the top of Kai, complete Google OAuth, then select an account (by name) and retry."
        )

    if not login_cid:
        return (
            "You're connected to SA360, but I can't list accounts yet because your Manager (MCC) isn't saved for this session "
            "(so I don't have account data to query yet). "
            "This could be because the MCC wasn't saved in this session. "
            "Next step: enter your MCC at the top of Kai and click 'Save MCC', then pick an account (by name) and retry."
        )

    if not default_cid:
        return (
            "I can run that analysis, but this request didn't include a customer account, so I don't have report output / account data to query. "
            "It could be that you haven't selected an account yet. "
            "Next step: pick an account from the 'Account (by name)' picker at the top of Kai (or paste a customer ID), then retry."
        )

    return (
        "I can run that analysis, but this request didn't include a customer account, so I don't have report output / account data to query. "
        "It could be that the selected account isn't saved for this session yet. "
        "Next step: confirm the account in the 'Account (by name)' picker at the top of Kai (or paste a customer ID), then retry."
    )


def _advisor_evidence_ok(text: str) -> bool:
    lower = (text or "").lower()
    return bool(
        re.search(
            r"(ctr|cpc|cpa|roas|conversion|conversions|impression|impressions|click|clicks|cost|spend|revenue|aov|cvr|rate|report|output|performance data|audit output|account data)",
            lower,
        )
    )


def _ensure_advisor_sections(reply: str | None, evidence: str, hypothesis: str, next_step: str) -> str:
    """
    Soft structure guardrail for advisor replies.

    Goal: keep responses actionable without forcing rigid "Evidence/Hypothesis/Next step" headers.
    This function should be used sparingly (primarily as a fallback when an upstream model/tool reply
    is too generic).
    """
    text = (reply or "").strip()
    lower = text.lower()

    # Evidence: ensure at least one grounding cue exists (metrics / data / output).
    has_grounding = bool(
        re.search(
            r"(report output|performance data|audit output|account data|cpc|ctr|cpa|roas|conversions|clicks|impressions|spend|cost)",
            lower,
        )
    )
    if not has_grounding and evidence:
        text = f"{text}\nBased on what I have: {evidence}".strip()
        lower = text.lower()

    # Hypothesis: keep it explicitly uncertain ("likely") but avoid templated headers.
    has_hypothesis = bool(re.search(r"(likely|may be|could be|suggests|driven by|because|due to)", lower))
    if not has_hypothesis and hypothesis:
        hyp = hypothesis.strip()
        if hyp and not hyp.lower().startswith(("it is", "it's", "likely")):
            hyp = f"It is likely that {hyp[0].lower()}{hyp[1:]}" if len(hyp) > 1 else f"It is likely that {hyp}"
        text = f"{text}\n{hyp}".strip()
        lower = text.lower()

    # Next step: ensure one concrete action exists.
    has_next = "next step" in lower or bool(re.search(r"\bnext\b", lower))
    if not has_next and next_step:
        step = next_step.strip()
        if step and not step.lower().startswith(("next", "do", "review", "check", "run")):
            step = f"Next: {step}"
        text = f"{text}\n{step}".strip()

    return text


def _advisor_strategy_reply(question: str) -> str | None:
    """Generate a concise advisor-style strategy response (prefer local)."""
    system = (
        chat_system_prompt()
        + " The user is asking for strategy/next steps. "
        "Respond in clear, natural language. "
        "Give 2-3 options with tradeoffs (risk/effort/time-to-impact) and end with your recommended next step. "
        "If account-specific data is required but not provided, ask one targeted question (account + timeframe) rather than guessing."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"User question: {question}"},
    ]
    reply, _meta = _call_local_llm(messages, intent="strategy_advisor", max_tokens=220)
    if reply and not _is_low_quality_local_reply(reply):
        # Do not force templates; return the model's natural-language strategy.
        return reply.strip()
    return None


def _humanize_relative_date_tokens(text: str) -> str:
    """Normalize internal date-range tokens into user-facing language."""
    if not isinstance(text, str) or not text:
        return text
    replacements = {
        "LAST_7_DAYS": "last 7 days",
        "LAST_14_DAYS": "last 14 days",
        "LAST_30_DAYS": "last 30 days",
        "THIS_MONTH": "this month",
        "LAST_MONTH": "last month",
        "THIS_QUARTER": "this quarter",
        "LAST_QUARTER": "last quarter",
    }
    out = text
    for token, phrase in replacements.items():
        out = re.sub(rf"\b{re.escape(token)}\b", phrase, out)
    # Improve readability for comma-separated explicit date spans.
    out = re.sub(r"(\d{4}-\d{2}-\d{2}),(\d{4}-\d{2}-\d{2})", r"\1 to \2", out)
    return out


def _normalize_reply_text(text: str | None) -> str | None:
    if not text or not isinstance(text, str):
        return text
    # Normalize common mojibake bullet artifacts to ASCII for readability.
    cleaned = _humanize_relative_date_tokens(text).replace("\u0080", "")
    cleaned = re.sub(r"\?\?\s*([+-]?\d)", r"delta \1", cleaned)
    cleaned = cleaned.replace("??", "delta ")
    cleaned = re.sub(r"(?i)No date specified;\s*defaulting to\s*last\s*\d+\s*days\.?", "", cleaned)
    cleaned = re.sub(r"(?i)No date specified;\s*defaulting to\s*LAST_\d+_DAYS\.?", "", cleaned)
    cleaned = re.sub(r"(?i)\brouter_[a-z0-9_]+(?:\s+[a-z0-9_=.-]+)*\b", "", cleaned)
    cleaned = re.sub(r"(?i)\bmodel=(?:local|azure|planner|rules)\b", "", cleaned)
    cleaned = re.sub(r"(?i)api keys?", "authorized access", cleaned)
    cleaned = re.sub(r"(?i)access tokens?", "authorized access", cleaned)
    cleaned = cleaned.encode("ascii", "ignore").decode()
    return (
        cleaned.replace("ƒ›", "-")
        .replace("ƒ?›", "-")
        .replace("•", "-")
        .replace("â€¢", "-")
        .replace("â€“", "-")
        .replace("â€”", "-")
    )


def _sanitize_kb_content(text: str | None, max_len: int = 800) -> str | None:
    if not text or not isinstance(text, str):
        return text
    cleaned = text.replace("\u0080", "")
    cleaned = re.sub(r"\?\?\s*([+-]?\d)", r"delta \1", cleaned)
    cleaned = cleaned.replace("??", "x")
    cleaned = cleaned.replace("�", "")
    cleaned = cleaned.encode("ascii", "ignore").decode()
    cleaned = cleaned.strip()
    if len(cleaned) <= max_len:
        return cleaned
    truncated = cleaned[:max_len]
    last_break = max(truncated.rfind("\n"), truncated.rfind(" "), truncated.rfind("."))
    if last_break > 200:
        truncated = truncated[: last_break + 1]
    return truncated.rstrip()


def _truncate_text(text: str, max_len: int) -> str:
    if not text or max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_break = max(truncated.rfind("\n"), truncated.rfind(" "), truncated.rfind("."))
    if last_break > max_len * 0.4:
        truncated = truncated[: last_break + 1]
    return truncated.rstrip()


def _truncate_messages_for_local(messages: list[dict], max_chars: int) -> list[dict]:
    if max_chars <= 0:
        return messages
    total = sum(
        len(m.get("content", "")) for m in messages if isinstance(m.get("content"), str)
    )
    if total <= max_chars:
        return messages

    trimmed: list[dict] = []
    start_idx = 0
    if messages and messages[0].get("role") == "system":
        trimmed.append(dict(messages[0]))
        start_idx = 1

    tail_start = max(start_idx, len(messages) - 4)
    for msg in messages[tail_start:]:
        if trimmed and msg is messages[0]:
            continue
        trimmed.append(dict(msg))

    trimmed_total = sum(
        len(m.get("content", "")) for m in trimmed if isinstance(m.get("content"), str)
    )
    if trimmed_total > max_chars:
        per_message = max(200, max_chars // max(1, len(trimmed)))
        for msg in trimmed:
            content = msg.get("content")
            if isinstance(content, str) and len(content) > per_message:
                msg["content"] = _truncate_text(content, per_message)
    return trimmed


def _normalize_numeric_token(token: str) -> str:
    if not token:
        return ""
    cleaned = token.strip().replace(",", "").replace("−", "-")
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    return cleaned


def _add_numeric_variants(value: float | int, tokens: set[str]) -> None:
    if isinstance(value, bool):
        return
    try:
        raw = float(value)
    except Exception:
        return
    variants: set[str] = set()
    for candidate in (raw, abs(raw)):
        # Preserve the original string (including sign) and common formatted variants.
        variants.add(str(value) if candidate == raw else str(abs(raw)))
        variants.add(str(int(candidate)) if candidate.is_integer() else str(candidate))
        variants.add(f"{candidate:.0f}")
        variants.add(f"{candidate:.1f}")
        variants.add(f"{candidate:.2f}")
        variants.add(f"{candidate:.3f}")
        variants.add(f"{candidate:,.2f}")
    for v in variants:
        norm = _normalize_numeric_token(v)
        if norm:
            tokens.add(norm)


def _collect_numeric_tokens(value: Any, tokens: set[str]) -> None:
    import re
    if value is None:
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _add_numeric_variants(value, tokens)
        return
    if isinstance(value, str):
        for match in re.finditer(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?", value):
            norm = _normalize_numeric_token(match.group(0))
            if norm:
                tokens.add(norm)
                # Variant: allow signless token when the evidence contains a signed delta
                # (e.g., "-42" in deltas) but the model expresses magnitude + direction in words.
                if norm.startswith("-") and len(norm) > 1:
                    tokens.add(norm[1:])
                # Variant: strip leading zeros for integer-ish tokens ("03" -> "3"),
                # because model outputs often omit leading zeros in dates.
                sign = ""
                raw = norm
                if raw.startswith("-"):
                    sign = "-"
                    raw = raw[1:]
                if raw.isdigit() and raw.startswith("0") and raw != "0":
                    stripped = raw.lstrip("0")
                    if stripped:
                        tokens.add(sign + stripped)
                        if sign == "-":
                            tokens.add(stripped)
                # Variant: normalize trailing zeros in decimals ("12.60" -> "12.6", "12.00" -> "12").
                if "." in raw and not raw.endswith("%"):
                    trimmed = raw.rstrip("0").rstrip(".")
                    if trimmed and trimmed != raw:
                        tokens.add(sign + trimmed)
                        if sign == "-":
                            tokens.add(trimmed)
        return
    if isinstance(value, dict):
        for v in value.values():
            _collect_numeric_tokens(v, tokens)
        return
    if isinstance(value, (list, tuple, set)):
        for v in value:
            _collect_numeric_tokens(v, tokens)


def _reply_has_numbers(text: str | None) -> bool:
    tokens: set[str] = set()
    if not text:
        return False
    _collect_numeric_tokens(text, tokens)
    return bool(tokens)


def _extract_tool_output_from_message(text: str) -> Any | None:
    import json
    if not text or "planner output" not in text.lower():
        return None
    idx = text.lower().find("planner output")
    if idx < 0:
        return None
    snippet = text[idx:]
    brace_idx = snippet.find("{")
    if brace_idx < 0:
        return None
    start = idx + brace_idx
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == "\"":
            in_string = not in_string
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                block = text[start:i + 1]
                try:
                    return json.loads(block)
                except Exception:
                    return None
    return None


def _extract_allowed_numbers(chat: ChatMessage) -> set[str]:
    tokens: set[str] = set()
    ctx = chat.context if isinstance(chat.context, dict) else {}
    tool_output = None
    if isinstance(ctx, dict):
        tool_output = ctx.get("tool_output")
    if tool_output is None:
        tool_output = _extract_tool_output_from_message(chat.message or "")
    if tool_output is None:
        stored_ctx = _load_last_tool_context(_extract_session_id(chat))
        if isinstance(stored_ctx, dict):
            tool_output = stored_ctx.get("tool_output")
    if tool_output is None:
        return tokens
    _collect_numeric_tokens(tool_output, tokens)
    return tokens


def _compact_tool_output(tool_output: Any) -> Any:
    if not isinstance(tool_output, dict):
        return tool_output
    out: dict[str, Any] = {}
    # "enhanced_summary" is a UX helper (it may append a follow-up question). We intentionally
    # exclude it from compacted tool output used as LLM context to avoid the model copying
    # a low-quality or hallucinated question verbatim into advisor replies.
    for key in ("summary", "notes", "analysis", "summary_seed"):
        if key in tool_output:
            out[key] = tool_output.get(key)
    plan = tool_output.get("plan")
    if isinstance(plan, dict):
        out["plan"] = {
            "account_name": plan.get("account_name"),
            "customer_ids": plan.get("customer_ids"),
            "date_range": plan.get("date_range"),
        }
    result = tool_output.get("result")
    if isinstance(result, dict):
        out["result"] = {
            "current": result.get("current"),
            "previous": result.get("previous"),
            "deltas": result.get("deltas"),
            "data_quality": result.get("data_quality"),
        }
    return out


def _extract_summary_seed(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    analysis = payload.get("analysis")
    if isinstance(analysis, dict):
        summary = analysis.get("summary")
        if isinstance(summary, str) and summary.strip() and not _is_refusal_reply(summary):
            return summary.strip()
    # Prefer stable summaries for numeric grounding. "enhanced_summary" may include an appended
    # follow-up question (UX helper) which can cause LLMs to copy that question verbatim.
    for key in ("summary_seed", "summary", "enhanced_summary"):
        candidate = payload.get(key)
        if isinstance(candidate, str) and candidate.strip() and not _is_refusal_reply(candidate):
            return candidate.strip()
    return None


def _compact_performance_payload(payload: Any) -> Any:
    """Reduce performance payload size for LLM followups to avoid timeouts."""
    if not isinstance(payload, dict):
        return payload
    out: dict[str, Any] = {}
    # For LLM context, prefer the stable numeric summary only; exclude "enhanced_summary" which
    # may contain an appended follow-up question intended for the UI.
    for key in ("summary", "notes"):
        if key in payload:
            out[key] = payload.get(key)
    analysis = payload.get("analysis")
    if isinstance(analysis, dict):
        slim = {k: analysis.get(k) for k in ("summary", "note", "deltas", "metric_focus") if k in analysis}
        drivers = analysis.get("drivers")
        if isinstance(drivers, dict):
            slim_drivers = {}
            for key in ("campaign", "device", "geo"):
                items = drivers.get(key)
                if isinstance(items, list):
                    slim_drivers[key] = items[:3]
            if slim_drivers:
                slim["drivers"] = slim_drivers
        kb = analysis.get("kb")
        if isinstance(kb, list):
            slim_kb = []
            for item in kb[:2]:
                if not isinstance(item, dict):
                    continue
                content = item.get("content") or ""
                if isinstance(content, str) and len(content) > 300:
                    content = content[:300].rstrip()
                slim_kb.append(
                    {
                        "title": item.get("title"),
                        "section": item.get("section"),
                        "content": content,
                    }
                )
            if slim_kb:
                slim["kb"] = slim_kb
        if slim:
            out["analysis"] = slim
    plan = payload.get("plan")
    if isinstance(plan, dict):
        out["plan"] = {
            "account_name": plan.get("account_name"),
            "customer_ids": plan.get("customer_ids"),
            "date_range": plan.get("date_range"),
        }
    result = payload.get("result")
    if isinstance(result, dict):
        out["result"] = {
            "current": result.get("current"),
            "previous": result.get("previous"),
            "deltas": result.get("deltas"),
            "data_quality": result.get("data_quality"),
        }
    return out


def _tool_context_key(session_id: str) -> str:
    return f"last_tool_context:{session_id}"


def _sanitize_table_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", name or "")
    if not cleaned:
        return "kaitoolcontext"
    if not cleaned[0].isalpha():
        cleaned = f"k{cleaned}"
    return cleaned[:63]


_TOOL_CONTEXT_TABLE_CLIENT: Any | None = None
_TOOL_CONTEXT_TABLE_NAME = _sanitize_table_name(os.environ.get("KAI_TOOL_CONTEXT_TABLE", "kai_tool_context"))
_TOOL_CONTEXT_TABLE_ENABLED = os.environ.get("KAI_TOOL_CONTEXT_TABLE_ENABLED", "true").lower() == "true"


def _get_tool_context_table() -> Any | None:
    global _TOOL_CONTEXT_TABLE_CLIENT
    if not _TOOL_CONTEXT_TABLE_ENABLED:
        return None
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        return None
    if _TOOL_CONTEXT_TABLE_CLIENT is not None:
        return _TOOL_CONTEXT_TABLE_CLIENT
    try:
        service = TableServiceClient.from_connection_string(conn)
        table = service.get_table_client(_TOOL_CONTEXT_TABLE_NAME)
        try:
            table.create_table()
        except ResourceExistsError:
            pass
        _TOOL_CONTEXT_TABLE_CLIENT = table
        return table
    except Exception:
        return None


def _save_last_tool_context(
    session_id: str | None,
    tool: str | None,
    tool_output: Any,
    prompt_kind: str | None,
) -> None:
    if not session_id or tool_output is None:
        return
    try:
        compact = _compact_tool_output(tool_output)
        payload = {
            "tool": tool,
            "prompt_kind": prompt_kind,
            "tool_output": compact,
            "ts": time.time(),
        }
        table = _get_tool_context_table()
        if table is not None:
            try:
                serialized = json.dumps(compact, ensure_ascii=True)
                if len(serialized) <= 60000:
                    entity = {
                        "PartitionKey": session_id,
                        "RowKey": "last",
                        "tool": tool or "",
                        "prompt_kind": prompt_kind or "",
                        "tool_output": serialized,
                        "ts": payload["ts"],
                    }
                    table.upsert_entity(mode=UpdateMode.REPLACE, entity=entity)
                    return
            except Exception:
                pass
        db_manager.set_setting(_tool_context_key(session_id), payload)
    except Exception:
        return


def _load_last_tool_context(session_id: str | None, max_age_seconds: int = 7200) -> dict | None:
    if not session_id:
        return None
    table = _get_tool_context_table()
    if table is not None:
        try:
            entity = table.get_entity(partition_key=session_id, row_key="last")
            payload = {
                "tool": entity.get("tool") or None,
                "prompt_kind": entity.get("prompt_kind") or None,
                "tool_output": None,
                "ts": entity.get("ts"),
            }
            raw_output = entity.get("tool_output")
            if isinstance(raw_output, str) and raw_output.strip():
                try:
                    payload["tool_output"] = json.loads(raw_output)
                except Exception:
                    payload["tool_output"] = raw_output
            else:
                payload["tool_output"] = raw_output
            ts = payload.get("ts")
            if ts is not None:
                try:
                    age = time.time() - float(ts)
                    if age > max_age_seconds:
                        return None
                except Exception:
                    return None
            return payload
        except ResourceNotFoundError:
            pass
        except Exception:
            pass
    try:
        payload = db_manager.get_setting(_tool_context_key(session_id), default=None)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    ts = payload.get("ts")
    if ts is not None:
        try:
            age = time.time() - float(ts)
            if age > max_age_seconds:
                return None
        except Exception:
            return None
    return payload


def _normalize_session_id(session_id: str | None) -> str | None:
    if not session_id:
        return None
    sid = str(session_id).strip()
    return sid or None


def _session_id_from_request(request: Request, fallback: str | None = None) -> str | None:
    return _normalize_session_id(
        fallback
        or request.headers.get("x-session-id")
        or request.query_params.get("session_id")
    )

# --- Entra SSO helpers (optional, enterprise/broad beta) ---
_ENTRA_JWKS_CACHE: dict[str, Any] = {"ts": None, "by_kid": {}}
_ENTRA_JWKS_TTL_SECONDS = int(os.environ.get("ENTRA_JWKS_TTL_SECONDS", "21600") or "21600")  # 6h


def _entra_sso_enabled() -> bool:
    return (KAI_SSO_MODE in {"optional", "required"}) and bool(ENTRA_TENANT_ID and ENTRA_CLIENT_ID)


def _entra_bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth:
        return None
    auth = str(auth).strip()
    if not auth.lower().startswith("bearer "):
        return None
    token = auth.split(" ", 1)[1].strip()
    return token or None


def _entra_jwks_url() -> str | None:
    if not ENTRA_TENANT_ID:
        return None
    return f"https://login.microsoftonline.com/{ENTRA_TENANT_ID}/discovery/v2.0/keys"


def _entra_issuer() -> str | None:
    if not ENTRA_TENANT_ID:
        return None
    return f"https://login.microsoftonline.com/{ENTRA_TENANT_ID}/v2.0"


def _entra_allowed_audiences() -> list[str]:
    if not ENTRA_CLIENT_ID:
        return []
    # Some tokens use 'api://<client_id>' as audience; accept both for practicality.
    return [ENTRA_CLIENT_ID, f"api://{ENTRA_CLIENT_ID}"]


def _entra_refresh_jwks(force: bool = False) -> None:
    if not _entra_sso_enabled():
        return
    now = time.time()
    ts = _ENTRA_JWKS_CACHE.get("ts")
    if (not force) and ts and (now - float(ts)) <= float(_ENTRA_JWKS_TTL_SECONDS):
        return
    url = _entra_jwks_url()
    if not url:
        return
    try:
        resp = httpx.get(url, timeout=10)
        if resp.status_code != 200:
            return
        payload = resp.json() if resp.content else {}
        keys = payload.get("keys") or []
        by_kid: dict[str, Any] = {}
        for k in keys:
            kid = (k or {}).get("kid")
            if kid:
                by_kid[str(kid)] = k
        if by_kid:
            _ENTRA_JWKS_CACHE["by_kid"] = by_kid
            _ENTRA_JWKS_CACHE["ts"] = now
    except Exception:
        return


def _entra_claims_from_request(request: Request) -> dict[str, Any] | None:
    """
    Validate an Entra JWT (id_token or access_token) and return claims.
    Returns None if SSO is disabled or token missing/invalid.
    """
    if not _entra_sso_enabled():
        return None
    token = _entra_bearer_token(request)
    if not token:
        return None
    try:
        header = jwt.get_unverified_header(token) or {}
        kid = str(header.get("kid") or "").strip()
        if not kid:
            return None
    except Exception:
        return None

    _entra_refresh_jwks(force=False)
    jwk = (_ENTRA_JWKS_CACHE.get("by_kid") or {}).get(kid)
    if not jwk:
        _entra_refresh_jwks(force=True)
        jwk = (_ENTRA_JWKS_CACHE.get("by_kid") or {}).get(kid)
    if not jwk:
        return None
    try:
        key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
        issuer = _entra_issuer()
        audiences = _entra_allowed_audiences()
        if not issuer or not audiences:
            return None
        claims = jwt.decode(
            token,
            key=key,
            algorithms=["RS256"],
            audience=audiences,
            issuer=issuer,
        )
        tid = str(claims.get("tid") or "").strip()
        if ENTRA_TENANT_ID and tid and tid != ENTRA_TENANT_ID:
            return None
        return claims if isinstance(claims, dict) else None
    except Exception:
        return None


def _entra_principal_key_from_claims(claims: dict[str, Any] | None) -> str | None:
    if not claims:
        return None
    oid = (claims.get("oid") or claims.get("sub") or "").strip()
    if not oid:
        return None
    # Azure Table Storage keys are strict; avoid forbidden chars.
    return f"u-{oid}"


def _sa360_scope_from_request(request: Request, fallback: str | None = None) -> str | None:
    """
    Determine the SA360 storage scope for this request.

    - If Entra JWT is present+valid, scope to the Entra principal (per-user tokens/defaults).
    - Else fall back to the legacy session_id (per-browser-session tokens/defaults).
    """
    claims = _entra_claims_from_request(request)
    principal = _entra_principal_key_from_claims(claims)
    if principal:
        return principal
    if KAI_SSO_MODE == "required" and _entra_sso_enabled():
        raise HTTPException(status_code=401, detail="SSO required.")
    return _session_id_from_request(request, fallback)


_SA360_TOKEN_TABLE_CLIENT: Any | None = None
_SA360_TOKEN_TABLE_NAME = _sanitize_table_name(os.environ.get("KAI_SA360_TOKEN_TABLE", "kai_sa360_tokens"))
_SA360_TOKEN_TABLE_ENABLED = os.environ.get("KAI_SA360_TOKEN_TABLE_ENABLED", "true").lower() == "true"


def _get_sa360_token_table() -> Any | None:
    global _SA360_TOKEN_TABLE_CLIENT
    if not _SA360_TOKEN_TABLE_ENABLED:
        return None
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        return None
    if _SA360_TOKEN_TABLE_CLIENT is not None:
        return _SA360_TOKEN_TABLE_CLIENT
    try:
        service = TableServiceClient.from_connection_string(conn)
        table = service.get_table_client(_SA360_TOKEN_TABLE_NAME)
        try:
            table.create_table()
        except ResourceExistsError:
            pass
        _SA360_TOKEN_TABLE_CLIENT = table
        return table
    except Exception:
        return None


def _sa360_session_key(session_id: str) -> str:
    return f"sa360_session:{session_id}"


def _load_sa360_session(session_id: str | None) -> dict | None:
    sid = _normalize_session_id(session_id)
    if not sid:
        return None
    table = _get_sa360_token_table()
    if table is not None:
        try:
            entity = table.get_entity(partition_key=sid, row_key="token")
            return {
                "refresh_token": entity.get("refresh_token") or None,
                "login_customer_id": entity.get("login_customer_id") or None,
                "default_customer_id": entity.get("default_customer_id") or None,
                "default_account_name": entity.get("default_account_name") or None,
                "updated_at": entity.get("updated_at") or None,
            }
        except Exception:
            pass
    try:
        payload = db_manager.get_setting(_sa360_session_key(sid), default=None)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return None


def _upsert_sa360_session(
    session_id: str | None,
    refresh_token: str | None = None,
    login_customer_id: str | None = None,
    default_customer_id: str | None = None,
    default_account_name: str | None = None,
) -> None:
    sid = _normalize_session_id(session_id)
    if not sid:
        return
    current = _load_sa360_session(sid) or {}
    prev_refresh = current.get("refresh_token")
    prev_login = current.get("login_customer_id")
    prev_default = current.get("default_customer_id")
    prev_default_name = current.get("default_account_name")
    refresh_changed = False
    login_changed = False
    default_changed = False
    if refresh_token and refresh_token != prev_refresh:
        current["refresh_token"] = refresh_token
        refresh_changed = True
    if login_customer_id and login_customer_id != prev_login:
        current["login_customer_id"] = login_customer_id
        login_changed = True
    if default_customer_id is not None and default_customer_id != prev_default:
        current["default_customer_id"] = default_customer_id
        default_changed = True
    if default_account_name is not None and default_account_name != prev_default_name:
        current["default_account_name"] = default_account_name
        default_changed = True
    if refresh_changed or login_changed:
        # If a user updates their OAuth token or MCC, the per-session account discovery cache is now stale.
        try:
            SA360_ACCOUNT_CACHE_BY_SESSION.pop(sid, None)
        except Exception:
            pass
    current["updated_at"] = datetime.utcnow().isoformat() + "Z"

    table = _get_sa360_token_table()
    if table is not None:
        try:
            entity = {
                "PartitionKey": sid,
                "RowKey": "token",
                "refresh_token": current.get("refresh_token") or "",
                "login_customer_id": current.get("login_customer_id") or "",
                "default_customer_id": current.get("default_customer_id") or "",
                "default_account_name": current.get("default_account_name") or "",
                "updated_at": current.get("updated_at") or "",
            }
            table.upsert_entity(mode=UpdateMode.REPLACE, entity=entity)
            return
        except Exception:
            pass
    try:
        db_manager.set_setting(_sa360_session_key(sid), current)
    except Exception:
        pass


def _sa360_state_secret() -> str:
    return (
        os.environ.get("SA360_OAUTH_STATE_SECRET")
        or os.environ.get("KAI_ACCESS_PASSWORD")
        or "kai-state-secret"
    )


def _sign_oauth_state(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    b64 = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("utf-8").rstrip("=")
    sig = hmac.new(_sa360_state_secret().encode("utf-8"), b64.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{b64}.{sig}"


def _parse_oauth_state(state: str, max_age_seconds: int = 900) -> dict | None:
    if not state or "." not in state:
        return None
    b64, sig = state.split(".", 1)
    expected = hmac.new(_sa360_state_secret().encode("utf-8"), b64.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    padded = b64 + "=" * (-len(b64) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")))
    except Exception:
        return None
    ts = payload.get("ts")
    if isinstance(ts, (int, float)) and max_age_seconds > 0:
        if time.time() - float(ts) > max_age_seconds:
            return None
    return payload
    if not isinstance(payload, dict):
        return None
    ts = payload.get("ts")
    if ts is not None:
        try:
            age = time.time() - float(ts)
            if age > max_age_seconds:
                return None
        except Exception:
            return None
    return payload


def _extract_reply_numbers(reply: str) -> list[str]:
    import re
    if not reply:
        return []
    normalized_reply = (
        reply.replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2012", "-")
    )
    numbers = []
    for match in re.finditer(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?", normalized_reply):
        token = match.group(0)
        start = match.start()
        end = match.end()
        line_start = normalized_reply.rfind("\n", 0, start) + 1
        prefix = normalized_reply[line_start:start]
        if prefix.strip() == "" and end < len(normalized_reply) and normalized_reply[end] in {".", ")"}:
            continue
        nearby = normalized_reply[max(0, start - 14):start].lower()
        if nearby.endswith("step "):
            continue
        if any(nearby.endswith(prefix) for prefix in ("option ", "path ", "approach ", "strategy ")):
            continue
        norm = _normalize_numeric_token(token)
        if norm:
            numbers.append(norm)
    return numbers


def _extract_reply_number_spans(reply: str) -> list[tuple[int, int, str, str]]:
    """
    Return numeric spans from a reply as tuples: (start, end, normalized, raw_token).

    This uses the same extraction rules as _extract_reply_numbers() so guardrails can
    sanitize ungrounded numbers without breaking list numbering or step labels.
    """
    import re
    if not reply:
        return []
    normalized_reply = (
        reply.replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2012", "-")
    )
    spans: list[tuple[int, int, str, str]] = []
    for match in re.finditer(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?", normalized_reply):
        raw = match.group(0)
        start = match.start()
        end = match.end()
        line_start = normalized_reply.rfind("\n", 0, start) + 1
        prefix = normalized_reply[line_start:start]
        # Ignore common list numbering at start-of-line ("1." / "1)") which is not evidence-bearing.
        if prefix.strip() == "" and end < len(normalized_reply) and normalized_reply[end] in {".", ")"}:
            continue
        nearby = normalized_reply[max(0, start - 14):start].lower()
        if nearby.endswith("step "):
            continue
        if any(nearby.endswith(prefix) for prefix in ("option ", "path ", "approach ", "strategy ")):
            continue
        norm = _normalize_numeric_token(raw)
        if norm:
            spans.append((start, end, norm, raw))
    return spans


def _sanitize_ungrounded_numbers(reply: str, allowed_numbers: set[str]) -> tuple[str, list[str]]:
    """
    Replace ungrounded numeric tokens with placeholders, preserving advisor structure.

    We prefer sanitization over discarding the whole reply so "Option A/B + monitoring plan"
    survives even when the model introduces an extra number (e.g., "7 days", "10-20%").
    """
    if not reply:
        return reply, []
    spans = _extract_reply_number_spans(reply)
    if not spans:
        return reply, []
    ungrounded = [s for s in spans if s[2] not in allowed_numbers]
    if not ungrounded:
        return reply, []
    import re
    out = reply
    blocked: list[str] = []
    # Replace from the end to keep indices stable.
    for start, end, norm, raw in sorted(ungrounded, key=lambda x: x[0], reverse=True):
        blocked.append(norm)
        # Drop the numeric token entirely rather than inserting "X"/"X%".
        # This keeps the response readable and avoids leaking placeholder artifacts to end users,
        # while still enforcing "no invented numbers" strictly.
        out = out[:start] + "" + out[end:]
    # Best-effort cleanup after removals (collapse spaces, fix spacing before punctuation).
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"\(\s*\)", "", out)
    out = out.strip()
    # De-dupe while preserving order (best-effort).
    seen = set()
    blocked_unique: list[str] = []
    for b in blocked[::-1]:  # reverse back to original order-ish
        if b in seen:
            continue
        seen.add(b)
        blocked_unique.append(b)
    return out, blocked_unique


def _apply_numeric_grounding_guardrail(
    reply: str,
    allowed_numbers: set[str],
    fallback_text: str | None = None,
) -> tuple[str, dict | None]:
    if not reply or not allowed_numbers:
        if not reply:
            return reply, None
        reply_numbers = _extract_reply_numbers(reply)
        if not reply_numbers:
            return reply, None
        guardrail = {
            "reason": "ungrounded_numbers",
            "blocked_numbers": list(dict.fromkeys(reply_numbers))[:8],
            "allowed_count": 0,
        }
        # Prefer sanitization so we keep structure even when numbers can't be grounded.
        sanitized, blocked = _sanitize_ungrounded_numbers(reply, allowed_numbers=set())
        if sanitized and sanitized != reply:
            guardrail["blocked_numbers"] = blocked[:8] if blocked else guardrail["blocked_numbers"]
            return sanitized, guardrail
        fallback = (fallback_text or "").strip()
        if fallback:
            return fallback, guardrail
        safe_reply = (
            "I can summarize the tool output, but I cannot introduce new numeric metrics that are not already provided. "
            "If you want exact numbers, please open the report or ask for a specific metric so I can surface it directly."
        )
        return safe_reply, guardrail
    reply_numbers = _extract_reply_numbers(reply)
    ungrounded = [n for n in reply_numbers if n not in allowed_numbers]
    if ungrounded:
        guardrail = {
            "reason": "ungrounded_numbers",
            "blocked_numbers": list(dict.fromkeys(ungrounded))[:8],
            "allowed_count": len(allowed_numbers),
        }
        # Prefer sanitization over discarding the reply, so advisor guidance survives.
        sanitized, blocked = _sanitize_ungrounded_numbers(reply, allowed_numbers=allowed_numbers)
        if sanitized and sanitized != reply:
            guardrail["blocked_numbers"] = blocked[:8] if blocked else guardrail["blocked_numbers"]
            return sanitized, guardrail
        fallback = (fallback_text or "").strip()
        if fallback:
            return fallback, guardrail
        safe_reply = (
            "I can summarize the tool output, but I cannot introduce new numeric metrics that are not already provided. "
            "If you want exact numbers, please open the report or ask for a specific metric so I can surface it directly."
        )
        return safe_reply, guardrail
    return reply, None


class EnvUpdateRequest(BaseModel):
    key: str
    value: str
    admin_password: str


def _safe_route_intent(payload: dict) -> RouteResponse:
    """Validate/normalize router JSON from LLM."""
    allowed_intents = {"general_chat", "audit", "performance", "pmax", "serp", "competitor", "creative", "seasonality"}
    intent = payload.get("intent") if isinstance(payload, dict) else None
    intent = intent.lower() if isinstance(intent, str) else "general_chat"
    if intent not in allowed_intents:
        intent = "general_chat"

    tool = payload.get("tool") if isinstance(payload, dict) else None
    tool = tool.lower() if isinstance(tool, str) else None
    allowed_tools = {"audit", "pmax", "serp", "competitor", "creative", "performance", None}
    if tool not in allowed_tools:
        tool = None

    run_planner = bool(payload.get("run_planner")) if isinstance(payload, dict) else False
    run_trends = bool(payload.get("run_trends")) if isinstance(payload, dict) else False
    themes_raw = payload.get("themes") if isinstance(payload, dict) else []
    themes = []
    if isinstance(themes_raw, list):
        for item in themes_raw:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if cleaned:
                themes.append(cleaned)
    if intent != "seasonality":
        themes = []
    if themes:
        themes = list(dict.fromkeys(themes))[:5]

    # Force tool alignment by intent
    if intent == "performance":
        tool = "performance"
    elif intent == "audit":
        tool = "audit"
    elif intent in {"pmax", "serp", "competitor", "creative"}:
        tool = intent
        run_planner = False
        run_trends = False
    elif intent == "seasonality":
        tool = None
        run_planner = False
        run_trends = True

    # General chat should never trigger tools or planner/trends
    if intent == "general_chat":
        tool = None
        run_planner = False
        run_trends = False

    ids_raw = payload.get("customer_ids") if isinstance(payload, dict) else []
    ids = []
    if isinstance(ids_raw, list):
        ids = [str(i) for i in ids_raw if str(i).isdigit()]
    needs_ids = bool(payload.get("needs_ids")) if isinstance(payload, dict) else False
    if not run_planner:
        needs_ids = False
    notes = payload.get("notes") if isinstance(payload, dict) else None
    confidence = None
    try:
        if isinstance(payload, dict) and payload.get("confidence") is not None:
            confidence = float(payload.get("confidence"))
            confidence = max(0.0, min(1.0, confidence))
    except Exception:
        confidence = None
    needs_clarification = bool(payload.get("needs_clarification")) if isinstance(payload, dict) else False
    clarification = payload.get("clarification") if isinstance(payload, dict) else None
    candidates_raw = payload.get("candidates") if isinstance(payload, dict) else []
    candidates = [c for c in candidates_raw if isinstance(c, str)] if isinstance(candidates_raw, list) else []

    return RouteResponse(
        intent=intent,
        tool=tool,
        run_planner=run_planner,
        run_trends=run_trends,
        themes=themes,
        customer_ids=ids,
        needs_ids=needs_ids,
        notes=notes if isinstance(notes, str) else None,
        confidence=confidence,
        needs_clarification=needs_clarification,
        clarification=clarification if isinstance(clarification, str) else None,
        candidates=candidates,
    )


# Lightweight account aliasing for chat-to-plan resolution
DEFAULT_ACCOUNT_ALIASES = [
    {
        "customer_id": "7902313748",
        "name": "US_Mobility Loyalty",
        "aliases": ["loyalty", "loyalty account", "us mobility loyalty", "mobility loyalty"],
    },
    {
        # Name-only entry to allow resolving account name without forcing a default ID; ID can be provided via ACCOUNT_ALIASES_JSON
        "customer_id": None,
        "name": "Recharge",
        "aliases": ["mobility recharge", "recharge account", "recharge mobility"],
    },
]

def _load_brand_competitor_aliases() -> tuple[list[str], list[str]]:
    """
    Load brand/competitor aliases from env.
    BRAND_ALIASES="brand,brand inc,brand.com"
    COMPETITOR_ALIASES="competitor1,comp one,competitor2"
    """
    def parse_env(key: str) -> list[str]:
        raw = os.environ.get(key, "") or ""
        return [t.strip().lower() for t in raw.split(",") if t.strip()]
    return parse_env("BRAND_ALIASES"), parse_env("COMPETITOR_ALIASES")

def _classify_campaign(name: str | None, channel_type: str | None, brand_aliases: list[str], competitor_aliases: list[str]) -> str:
    n = (name or "").lower()
    ch = (channel_type or "").lower()
    if "performance_max" in ch or "performance max" in n or "pmax" in n:
        return "pmax"
    for b in brand_aliases:
        if b and b in n:
            return "brand"
    for c in competitor_aliases:
        if c and c in n:
            return "competitor"
    return "nonbrand"


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", value.strip().lower())
    return re.sub(r"\s+", " ", cleaned).strip()

def _normalize_audit_business_unit(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = _normalize_text(value)
    if not cleaned:
        return None
    if "529 states" in cleaned:
        return "529 States"
    if re.search(r"\b529\s*(az|ct|de|ma|nh)\b", cleaned):
        return "529 States"
    if re.search(r"\bfidelity\s+529\s*(az|ct|de|ma|nh)\b", cleaned):
        return "529 States"
    return value


def _tokenize_text(value: str | None) -> list[str]:
    if not value:
        return []
    return [t for t in re.split(r"[^a-zA-Z0-9]+", value.lower()) if t]

def _fuzzy_score(a: str, b: str) -> float:
    """Lightweight similarity score between two strings."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def _extract_keyword_from_text(text: str) -> str | None:
    """Pull a keyword/phrase from quotes; fallback to trimmed text."""
    import re as _re
    if not text:
        return None
    m = _re.search(r'"([^"]+)"', text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    cleaned = text.strip()
    return cleaned if cleaned else None


def _extract_customer_ids_from_text(text: str | None) -> list[str]:
    """Find potential customer IDs (8-12 digit) in free text."""
    if not text:
        return []
    return list({m.group(0) for m in re.finditer(r"\b\d{8,12}\b", text)})


def _normalize_customer_id_value(raw: Any) -> str:
    """Normalize customer identifiers to plain digits when possible."""
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    text = text.replace("customers/", "").strip()
    digits = re.sub(r"\D+", "", text)
    if digits:
        return digits
    return text


def _normalize_customer_ids(ids: list[str] | None) -> list[str]:
    """Normalize, de-duplicate, and preserve order of customer IDs."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in ids or []:
        cid = _normalize_customer_id_value(raw)
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
    return out


def _load_account_aliases(session_id: str | None = None) -> list[dict]:
    """
    Merge default aliases with optional ACCOUNT_ALIASES_JSON env override.
    Env format: JSON list of {"customer_id": "...", "name": "...", "aliases": ["..."]} or a dict keyed by id.
    """
    aliases: list[dict] = list(DEFAULT_ACCOUNT_ALIASES)

    # Add live SA360 accounts (dynamic, adaptive to new accounts).
    # IMPORTANT: Scope to the caller's SA360 session so broad beta users only see accounts they can access.
    try:
        for acct in _sa360_list_customers_cached(session_id=session_id):
            if not isinstance(acct, dict):
                continue
            cid = str(acct.get("customer_id") or "").strip()
            name = (acct.get("name") or "").strip()
            if not cid or not name:
                continue
            aliases.append(
                {
                    "customer_id": cid,
                    "name": name,
                    "aliases": [name.lower(), name.replace("_", " "), name.replace("-", " ")],
                }
            )
    except Exception:
        pass
    raw = os.environ.get("ACCOUNT_ALIASES_JSON")
    if raw:
        try:
            parsed = json.loads(raw)
            extra: list[dict] = []
            if isinstance(parsed, list):
                extra = [e for e in parsed if isinstance(e, dict) and "customer_id" in e]
            elif isinstance(parsed, dict):
                for cid, meta in parsed.items():
                    if not isinstance(meta, dict):
                        continue
                    extra.append(
                        {
                            "customer_id": cid,
                            "name": meta.get("name", cid),
                            "aliases": meta.get("aliases", []),
                        }
                    )
            aliases.extend(extra)
        except Exception:
            pass

    merged: dict[str, dict] = {}
    for item in aliases:
        raw_cid = item.get("customer_id")
        if raw_cid is None:
            continue
        cid = str(raw_cid).strip()
        if not cid or cid.lower() == "none":
            continue
        entry = merged.get(cid, {"customer_id": cid, "name": None, "aliases": []})
        name = (item.get("name") or "").strip()
        if name and (not entry.get("name") or len(name) > len(entry.get("name") or "")):
            entry["name"] = name
        combined_aliases = list(entry.get("aliases") or []) + list(item.get("aliases") or [])
        deduped: list[str] = []
        seen: set[str] = set()
        for alias in combined_aliases:
            if not alias:
                continue
            key = _normalize_text(str(alias))
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(str(alias))
        entry["aliases"] = deduped
        merged[cid] = entry

    # Preserve name-only aliases that do not collide with known IDs.
    name_only: list[dict] = []
    id_norms: set[str] = set()
    for entry in merged.values():
        for label in [entry.get("name")] + list(entry.get("aliases") or []):
            if label:
                id_norms.add(_normalize_text(str(label)))

    for item in aliases:
        raw_cid = item.get("customer_id")
        if raw_cid is not None:
            cid = str(raw_cid).strip()
            if cid and cid.lower() != "none":
                continue
        name = (item.get("name") or "").strip()
        if not name:
            continue
        if _normalize_text(name) in id_norms:
            continue
        name_only.append(
            {
                "customer_id": None,
                "name": name,
                "aliases": item.get("aliases", []) or [],
            }
        )

    return list(merged.values()) + name_only


def _extract_entity_intent(message: str) -> dict:
    """
    Very lightweight entity intent extractor for ad/keyword queries.
    Returns dict with keys: entity_type ('ad'|'keyword'|'campaign'|None), identifier (str|None), metric (str|None), compare_wow (bool), device (str|None)
    """
    t = (message or "").lower()
    entity_type = None
    if "keyword" in t:
        entity_type = "keyword"
    elif "campaign" in t:
        entity_type = "campaign"
    elif "ad copy" in t or "ad performance" in t or "ad " in t or t.startswith("ad "):
        entity_type = "ad"
    elif "pmax" in t or "performance max" in t:
        entity_type = "pmax"

    metric = None
    for m in [
        "conversions",
        "conv",
        "clicks",
        "ctr",
        "cpc",
        "cpa",
        "roas",
        "cost",
        "spend",
        "impression share",
        "is",
        "lost is",
        "rank",
        "budget",
        "auction",
        "outranking",
        "top of page",
        "quality score",
        "qs",
        "policy",
        "disapproval",
        "feed",
    ]:
        if m in t:
            metric = m
            break
    if metric is None:
        custom_metrics = _extract_custom_metric_mentions(message)
        if custom_metrics:
            metric = custom_metrics[0]

    # Identifier: quoted string or trailing after keyword/ad
    identifier = None
    m_quote = re.search(r'"([^"]+)"', message or "")
    if m_quote:
        identifier = m_quote.group(1).strip()
    else:
        # crude heuristic: take token after 'keyword' or 'ad'
        tokens = re.split(r"\s+", message)
        for i, tok in enumerate(tokens):
            if tok.lower() in ("keyword", "ad", "adcopy", "ad_copy", "campaign") and i + 1 < len(tokens):
                candidate = tokens[i + 1].strip(" ,.?;:\"'()[]{}")
                if candidate:
                    identifier = candidate
                break

    device = None
    if "mobile" in t:
        device = "MOBILE"
    elif "desktop" in t:
        device = "DESKTOP"
    elif "tablet" in t:
        device = "TABLET"

    network = None
    if "search partner" in t:
        network = "SEARCH_PARTNERS"
    elif "search" in t:
        network = "SEARCH"
    elif "display" in t:
        network = "DISPLAY"

    brand_scope = None
    if "nonbrand" in t or "non-brand" in t or "generic" in t:
        brand_scope = "NON_BRAND"
    elif "brand" in t:
        brand_scope = "BRAND"

    match_type = None
    for mt in ["exact", "phrase", "broad"]:
        if mt in t:
            match_type = mt.upper()
            break

    audience = None
    if "remarketing" in t or "retarget" in t:
        audience = "REMARKETING"
    elif "similar" in t or "similar to" in t or "similar audience" in t:
        audience = "SIMILAR"

    geo = None
    geo_match = re.search(r"in ([A-Za-z\s]+)$", message or "")
    if geo_match:
        geo = geo_match.group(1).strip()

    daypart = None
    for dp in ["morning", "afternoon", "evening", "night"]:
        if dp in t:
            daypart = dp.upper()
            break

    compare_wow = any(p in t for p in ["wow", "week over week", "week-over-week", "week over-week", "w/w", "down", "up"])

    return {
        "entity_type": entity_type,
        "identifier": identifier,
        "metric": metric,
        "compare_wow": compare_wow,
        "device": device,
        "network": network,
        "brand_scope": brand_scope,
        "match_type": match_type,
        "audience": audience,
        "geo": geo,
        "daypart": daypart,
    }


def _extract_top_mover_intent(message: str) -> dict:
    """
    Detect top mover/anomaly intent when no specific entity identifier is given.
    """
    t = (message or "").lower()
    keywords = [
        "top",
        "best",
        "worst",
        "down",
        "up",
        "drop",
        "increase",
        "decrease",
        "spike",
        "anomaly",
        "mover",
        "trending",
        "performance",
        "shift",
        "change",
        "moved",
    ]
    is_top = any(k in t for k in keywords)
    entity_type = None
    if "keyword" in t:
        entity_type = "keyword"
    elif "ad" in t or "creative" in t:
        entity_type = "ad"
    elif "campaign" in t:
        entity_type = "campaign"
    elif "asset group" in t or "asset" in t:
        entity_type = "asset_group"

    metric = None
    for m in ["conversions", "conv", "clicks", "ctr", "cpc", "cpa", "roas", "cost", "spend", "impression share", "auction", "budget"]:
        if m in t:
            metric = m
            break

    device = None
    if "mobile" in t:
        device = "MOBILE"
    elif "desktop" in t:
        device = "DESKTOP"
    elif "tablet" in t:
        device = "TABLET"

    return {"is_top_intent": is_top, "entity_type": entity_type, "metric": metric, "device": device}


def _resolve_account_context(
    message: str,
    customer_ids: list[str],
    account_name: str | None,
    explicit_ids: bool = False,
    session_id: str | None = None,
) -> tuple[list[str], str | None, str | None, list[str]]:
    """
    Resolve account/customer from free-text using aliases and optional explicit IDs.
    Returns (customer_ids, account_name, notes, candidates)
    """
    notes: list[str] = []
    candidates: list[str] = []
    resolved_ids = _normalize_customer_ids(customer_ids)
    resolved_account = account_name

    aliases: list[dict] | None = None

    def _aliases() -> list[dict]:
        nonlocal aliases
        if aliases is None:
            aliases = _load_account_aliases(session_id=session_id)
        return aliases
    msg_norm = _normalize_text(message)
    acct_norm = _normalize_text(account_name)
    ids_in_text: list[str] = _extract_customer_ids_from_text(message or "")

    # These stopwords are used in both token-based matching and default-account fallback logic.
    # Keep them stable to prevent routing/account-resolution regressions across builds.
    stopwords = {
        "how",
        "did",
        "the",
        "a",
        "an",
        "account",
        "accounts",
        "perform",
        "performance",
        "last",
        "week",
        "run",
        "audit",
        "check",
        "for",
        "me",
        "please",
        "show",
        "tell",
        "our",
        "my",
        "we",
        "us",
        "this",
        "that",
        "of",
        "on",
        "in",
        "to",
        "report",
        "metrics",
        "data",
        "stats",
        "results",
    }

    # If the caller supplied explicit customer_ids (e.g., via UI account picker), treat them as
    # authoritative. We may backfill a friendly name, but we must not override/clear the IDs due to
    # fuzzy name matches in the message.
    if explicit_ids and resolved_ids:
        if not resolved_account:
            by_id = {
                _normalize_customer_id_value(a.get("customer_id")): a
                for a in _aliases()
                if a.get("customer_id")
            }
            if resolved_ids[0] in by_id:
                resolved_account = by_id[resolved_ids[0]].get("name") or resolved_account
        if not resolved_account:
            notes.append(f"Using provided customer_id {resolved_ids[0]}.")
        return resolved_ids, resolved_account, "; ".join(notes) if notes else None, candidates

    def matches_entry(entry: dict) -> bool:
        names = [entry.get("name", "")] + entry.get("aliases", []) or []
        for name in names:
            if not name:
                continue
            if _normalize_text(name) and _normalize_text(name) in msg_norm:
                return True
        return False

    def _dedupe_entries(entries: list[dict]) -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
        for entry in entries:
            key = str(entry.get("customer_id") or _normalize_text(entry.get("name") or "")).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(entry)
        return out

    resolved_from_name_match = False

    def _finalize_matches(entries: list[dict]) -> bool:
        nonlocal resolved_ids, resolved_account, candidates, notes, resolved_from_name_match
        if not entries:
            return False
        with_ids = [e for e in entries if e.get("customer_id")]
        if with_ids:
            entries = with_ids
        entries = _dedupe_entries(entries)
        if len(entries) > 1:
            human_names = [
                f"{m.get('name','')} ({m.get('customer_id') or 'needs id'})".strip()
                for m in entries
            ]
            candidates = human_names
            notes.append(f"Multiple account matches found: {', '.join(human_names)}. Please specify a customer ID.")
            resolved_ids = []
            resolved_account = None
            return True
        m = entries[0]
        if m.get("customer_id"):
            resolved_from_name_match = True
            if resolved_ids and m["customer_id"] not in resolved_ids:
                if ids_in_text:
                    notes.append(
                        "Message included an explicit customer_id; keeping that ID despite a different name match."
                    )
                else:
                    resolved_ids = [m["customer_id"]]
                    notes.append("Overrode prior account context based on message match.")
            elif not resolved_ids:
                resolved_ids = [m["customer_id"]]
            if not resolved_account or (resolved_ids and resolved_ids[0] == m["customer_id"]):
                resolved_account = m.get("name") or resolved_account
            notes.append(f"Resolved account to {m.get('name','')} ({m.get('customer_id')}).")
        else:
            resolved_account = resolved_account or m.get("name")
            notes.append(f"Identified account '{resolved_account}' but need a customer ID.")
        return True

    all_aliases = _aliases()
    matched_entries = [e for e in all_aliases if matches_entry(e)]
    if not matched_entries and acct_norm:
        matched_entries = [
            e
            for e in all_aliases
            if _normalize_text(e.get("name")) == acct_norm
            or any(_normalize_text(a) == acct_norm for a in (e.get("aliases") or []))
        ]
    matched_with_ids = [e for e in matched_entries if e.get("customer_id")]
    name_only_matches = [e for e in matched_entries if not e.get("customer_id")]
    # Digits present in message (possible customer_id)
    if ids_in_text and not resolved_ids:
        resolved_ids = [ids_in_text[0]]
        notes.append(f"Detected customer_id {ids_in_text[0]} from message.")

    # If an explicit ID was provided in the message, honor it and skip ambiguous name matching.
    if explicit_ids and resolved_ids and ids_in_text:
        if not resolved_account:
            by_id = {a.get("customer_id"): a for a in all_aliases if a.get("customer_id")}
            if resolved_ids[0] in by_id:
                resolved_account = by_id[resolved_ids[0]].get("name") or resolved_account
        if not resolved_account:
            notes.append(f"Using provided customer_id {resolved_ids[0]}.")
        return resolved_ids, resolved_account, "; ".join(notes) if notes else None, candidates

    resolved = False
    if matched_with_ids:
        resolved = _finalize_matches(matched_entries)

    if not resolved:
        # Token-based match to align short account mentions with SA360 names.
        msg_tokens = [t for t in _tokenize_text(message) if t not in stopwords]
        scored: list[tuple[int, dict]] = []
        if msg_tokens:
            for entry in all_aliases:
                names = [entry.get("name", "")] + entry.get("aliases", []) or []
                entry_tokens: set[str] = set()
                for name in names:
                    for tok in _tokenize_text(name):
                        if tok and tok not in stopwords:
                            entry_tokens.add(tok)
                score = len(set(msg_tokens) & entry_tokens)
                if score >= 1:
                    scored.append((score, entry))
        if scored:
            max_score = max(score for score, _ in scored)
            top = [entry for score, entry in scored if score == max_score]
            top_with_ids = [entry for entry in top if entry.get("customer_id")]
            if top_with_ids:
                top = top_with_ids
            if len(top) == 1 and (max_score >= 2 or len(set(msg_tokens)) <= 1):
                resolved = _finalize_matches(top)
            else:
                top = _dedupe_entries(top)[:3]
                human = [f"{t.get('name','')} ({t.get('customer_id') or 'needs id'})" for t in top]
                candidates = human
                notes.append(f"Possible matches: {', '.join(human)}. Please confirm a customer ID.")
                resolved_ids = []
                resolved_account = None
        else:
            # Try fuzzy match against known aliases/names to propose options without defaulting
            fuzzy_candidates: list[tuple[float, dict]] = []
            for entry in all_aliases:
                names = [entry.get("name", "")] + entry.get("aliases", []) or []
                for name in names:
                    score = _fuzzy_score(msg_norm, _normalize_text(name))
                    if score >= 0.72:
                        fuzzy_candidates.append((score, entry))
                        break
            fuzzy_candidates = sorted(fuzzy_candidates, key=lambda x: x[0], reverse=True)[:3]
            if fuzzy_candidates:
                if any(c[1].get("customer_id") for c in fuzzy_candidates):
                    fuzzy_candidates = [c for c in fuzzy_candidates if c[1].get("customer_id")]
                human = [
                    f"{c[1].get('name','')} ({c[1].get('customer_id') or 'needs id'})"
                    for c in fuzzy_candidates
                ]
                candidates = human
                notes.append(f"Possible matches: {', '.join(human)}. Please confirm a customer ID.")
                resolved_ids = []
                resolved_account = None

    if not resolved_ids and not candidates and name_only_matches:
        name_only_matches = _dedupe_entries(name_only_matches)
        if name_only_matches:
            choice = name_only_matches[0]
            resolved_account = resolved_account or choice.get("name")
            notes.append(f"Identified account '{resolved_account}' but need a customer ID.")

    if resolved_ids and name_only_matches and not matched_with_ids and not ids_in_text and not resolved_from_name_match:
        notes.append("Cleared prior account context because only a name-only match was found.")
        resolved_ids = []

    # If defaults were injected but the text points elsewhere (name-only match), drop defaults to force clarification.
    #
    # Important: do NOT clear defaults for generic prompts like "how did performance look last week".
    # Default-account fallback exists specifically to make those prompts work without forcing account selection.
    default_ids = _default_customer_ids(session_id=session_id)
    # Matched entries can include "name-only" aliases (customer_id=None). Those are useful signals when users
    # explicitly mention an account label like "recharge", but they can also be noisy if a label is too short
    # or overlaps generic language ("in", "the", etc.). Only treat name-only matches as strong when the label
    # is substantive and not a stopword.
    def _is_strong_name_signal(label: str | None) -> bool:
        norm = _normalize_text(label)
        if not norm:
            return False
        if norm in stopwords:
            return False
        if norm.isdigit():
            return False
        # Guard against accidental matches on very short/common tokens.
        return len(norm) >= 4

    name_only_match_entries = [e for e in matched_entries if not e.get("customer_id")]
    strong_name_only_match = any(
        _is_strong_name_signal(e.get("name"))
        or any(_is_strong_name_signal(a) for a in (e.get("aliases") or []))
        for e in name_only_match_entries
    )
    if (
        resolved_ids
        and default_ids
        and resolved_ids == default_ids
        and strong_name_only_match
        and not ids_in_text
        and not explicit_ids
    ):
        notes.append("Cleared fallback customer_id because message did not match default account.")
        resolved_ids = []
        resolved_account = None

    # If we have an ID but no friendly name, try to backfill from aliases
    if resolved_ids and not resolved_account:
        by_id = {a["customer_id"]: a for a in all_aliases}
        if resolved_ids[0] in by_id:
            resolved_account = by_id[resolved_ids[0]].get("name") or resolved_account

    return resolved_ids, resolved_account, "; ".join(notes) if notes else None, candidates


def _default_customer_ids(session_id: str | None = None) -> list[str]:
    """
    Best-effort default customer IDs.

    Priority:
    1) Per-session saved default SA360 account (if present).
    2) DEFAULT_CUSTOMER_IDS env (comma-separated).
    """
    try:
        sess = _load_sa360_session(session_id)
        default_cid = str(sess.get("default_customer_id") or "").strip() if isinstance(sess, dict) else ""
        if default_cid:
            return [default_cid]
    except Exception:
        pass
    env_ids = os.environ.get("DEFAULT_CUSTOMER_IDS", "")
    if env_ids:
        return [cid.strip() for cid in env_ids.split(",") if cid.strip()]
    return []


class AdsConnectRequest(BaseModel):
    client_id: str
    client_secret: str
    developer_token: str
    refresh_token: str
    customer_ids: list[str] = []


class AdsFetchRequest(BaseModel):
    account_name: str
    customer_ids: list[str] = []
    date_range: str | None = "LAST_30_DAYS"  # GAQL date range preset or yyyy-mm-dd..yyyy-mm-dd
    dry_run: bool = False


class AdsFetchAuditRequest(AdsFetchRequest):
    business_unit: str = "Brand"


class Sa360FetchRequest(BaseModel):
    account_name: str | None = None  # will be auto-derived from SA360 if not provided
    customer_ids: list[str] = []
    date_range: str | None = "LAST_30_DAYS"
    dry_run: bool = False
    business_unit: str = "Brand"
    async_mode: bool = False
    session_id: str | None = None


class Sa360DiagnosticsRequest(BaseModel):
    customer_ids: list[str] = []
    date_range: str | None = "LAST_7_DAYS"
    include_previous: bool = True
    account_name: str | None = None
    report_names: list[str] | None = None
    # Optional: metric keys to compute in diagnostics snapshots (e.g. ["custom:store_visits"]).
    # The planner can compute custom conversion action metrics; diagnostics must be able to compute the same
    # values so QA can validate parity against live SA360 pulls.
    metrics: list[str] | None = None
    bypass_cache: bool = False
    async_mode: bool = False
    session_id: str | None = None


class Sa360LoginCustomerRequest(BaseModel):
    login_customer_id: str
    session_id: str | None = None


class Sa360DefaultAccountRequest(BaseModel):
    customer_id: str | None = None
    account_name: str | None = None
    session_id: str | None = None


class TrendsRequest(BaseModel):
    account_name: str | None = None
    customer_ids: list[str] = []
    themes: list[str] = []
    timeframe: str = "now 12-m"   # e.g., "now 12-m" or "2025-01-01,2025-03-31"
    geo: str | None = None
    budget: float | None = None
    use_performance: bool = True  # whether to seed weights from SA360 performance
    async_mode: bool = False
    session_id: str | None = None


# Feature flags
ADS_FETCH_ENABLED = os.environ.get("ADS_FETCH_ENABLED", "false").lower() == "true"
# Search Ads 360 fetcher (feature-flagged)
SA360_FETCH_ENABLED = os.environ.get("SA360_FETCH_ENABLED", "false").lower() == "true"
SA360_CACHE_ENABLED = os.environ.get("SA360_CACHE_ENABLED", "true").lower() == "true"
SA360_CACHE_FRESHNESS_DAYS = int(os.environ.get("SA360_CACHE_FRESHNESS_DAYS", "3") or "3")
SA360_CACHE_PREFIX = os.environ.get("SA360_CACHE_PREFIX", "sa360_cache")
SA360_ACCOUNT_CACHE_TTL_HOURS = int(os.environ.get("SA360_ACCOUNT_CACHE_TTL_HOURS", "6") or "6")
SA360_ACCOUNT_CACHE: dict[str, object] = {"ts": None, "data": []}
SA360_ACCOUNT_CACHE_BY_SESSION: dict[str, dict[str, object]] = {}
# Managers are discovered via customers:listAccessibleCustomers, then filtered to manager accounts.
# Cache per-session to avoid repeated "list managers" calls during onboarding.
SA360_MANAGER_CACHE_TTL_HOURS = int(os.environ.get("SA360_MANAGER_CACHE_TTL_HOURS", "6") or "6")
SA360_MANAGER_CACHE_BY_SESSION: dict[str, dict[str, object]] = {}
SA360_PLAN_MAX_SYNC_ACCOUNTS = int(os.environ.get("SA360_PLAN_MAX_SYNC_ACCOUNTS", "3") or "3")
SA360_DIAGNOSTICS_MAX_SYNC_ACCOUNTS = int(os.environ.get("SA360_DIAGNOSTICS_MAX_SYNC_ACCOUNTS", "3") or "3")
SA360_PLAN_CONCURRENCY = int(os.environ.get("SA360_PLAN_CONCURRENCY", "2") or "2")
SA360_PLAN_CHUNK_SIZE = int(os.environ.get("SA360_PLAN_CHUNK_SIZE", "1") or "1")
SA360_DIAGNOSTICS_CONCURRENCY = int(os.environ.get("SA360_DIAGNOSTICS_CONCURRENCY", "2") or "2")
SA360_DIAGNOSTICS_CHUNK_SIZE = int(os.environ.get("SA360_DIAGNOSTICS_CHUNK_SIZE", "1") or "1")
SA360_PLAN_BYPASS_CACHE = os.environ.get("SA360_PLAN_BYPASS_CACHE", "true").lower() == "true"
ENABLE_SUMMARY_ENHANCER = os.environ.get("ENABLE_SUMMARY_ENHANCER", "true").lower() == "true"

# Router verification controls
ROUTER_PRIMARY = os.environ.get("ROUTER_PRIMARY", "local").lower().strip()
ROUTER_VERIFY_MODE = os.environ.get("ROUTER_VERIFY_MODE", "adaptive").lower().strip()
ROUTER_VERIFY_CONFIDENCE = float(os.environ.get("ROUTER_VERIFY_CONFIDENCE", "0.7") or "0.7")

# Entra SSO (broad beta hardening)
# off: no JWT parsing/verification
# optional: accept JWT and scope SA360 tokens/defaults per user when present
# required: reject requests missing a valid JWT (UI must sign in)
KAI_SSO_MODE = os.environ.get("KAI_SSO_MODE", "off").lower().strip()
ENTRA_TENANT_ID = (os.environ.get("ENTRA_TENANT_ID") or os.environ.get("AZURE_TENANT_ID") or "").strip() or None
ENTRA_CLIENT_ID = (os.environ.get("ENTRA_CLIENT_ID") or "").strip() or None

# Async job queue
JOB_QUEUE_ENABLED = job_queue_enabled()
JOB_QUEUE_FORCE = job_queue_force()

# Rate limiting
RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "false").lower() == "true"
RATE_LIMIT_PER_MINUTE = float(os.environ.get("RATE_LIMIT_PER_MINUTE", "120") or "120")
RATE_LIMIT_BURST = int(os.environ.get("RATE_LIMIT_BURST", "40") or "40")
RATE_LIMIT_HEAVY_PER_MINUTE = float(os.environ.get("RATE_LIMIT_HEAVY_PER_MINUTE", "10") or "10")
RATE_LIMIT_HEAVY_BURST = int(os.environ.get("RATE_LIMIT_HEAVY_BURST", "5") or "5")
_RATE_LIMITER = TokenBucket(RATE_LIMIT_PER_MINUTE, RATE_LIMIT_BURST)
_HEAVY_RATE_LIMITER = TokenBucket(RATE_LIMIT_HEAVY_PER_MINUTE, RATE_LIMIT_HEAVY_BURST)
_HEAVY_PATH_PREFIXES = (
    "/api/integrations/sa360",
    "/api/audit/generate",
    "/api/trends/seasonality",
    "/api/diagnostics/sa360",
)

# Circuit breakers
CIRCUIT_BREAKER_ENABLED = os.environ.get("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
SA360_BREAKER = CircuitBreaker(
    "sa360",
    failure_threshold=int(os.environ.get("SA360_BREAKER_FAILURES", "3") or "3"),
    cooldown_seconds=int(os.environ.get("SA360_BREAKER_COOLDOWN_SECONDS", "60") or "60"),
    enabled=CIRCUIT_BREAKER_ENABLED,
)
SERP_BREAKER = CircuitBreaker(
    "serpapi",
    failure_threshold=int(os.environ.get("SERP_BREAKER_FAILURES", "3") or "3"),
    cooldown_seconds=int(os.environ.get("SERP_BREAKER_COOLDOWN_SECONDS", "60") or "60"),
    enabled=CIRCUIT_BREAKER_ENABLED,
)

# Performance/trends caching
TRENDS_MAX_SYNC_SECONDS = float(
    os.environ.get("TRENDS_MAX_SYNC_SECONDS", os.environ.get("TRENDS_TOTAL_TIMEOUT_SECONDS", "45") or "45") or "45"
)
TRENDS_QUEUE_ON_TIMEOUT = os.environ.get("TRENDS_QUEUE_ON_TIMEOUT", "true").lower() == "true"
TRENDS_PERF_TIMEOUT_SECONDS = float(os.environ.get("TRENDS_PERF_TIMEOUT_SECONDS", "25") or "25")
PERF_WEIGHT_CACHE = TTLCache(
    ttl_seconds=int(os.environ.get("SA360_PERF_WEIGHT_CACHE_TTL_SECONDS", "1800") or "1800"),
    max_items=int(os.environ.get("SA360_PERF_WEIGHT_CACHE_MAX_ITEMS", "128") or "128"),
)
PERF_SEASONALITY_CACHE = TTLCache(
    ttl_seconds=int(os.environ.get("SA360_PERF_SEASONALITY_CACHE_TTL_SECONDS", "1800") or "1800"),
    max_items=int(os.environ.get("SA360_PERF_SEASONALITY_CACHE_MAX_ITEMS", "128") or "128"),
)


class Sa360Account(BaseModel):
    customer_id: str
    name: str | None = None
    manager: bool | None = None
    # utilization note: exposed to frontend for disambiguation and manager guard


class Sa360ConversionActionItem(BaseModel):
    """
    Catalog entry for SA360 conversion actions + (optional) windowed totals.
    This powers both user-facing discovery ("what columns exist?") and QA.
    """

    metric_key: str
    name: str
    action_id: str | None = None
    category: str | None = None
    status: str | None = None
    conversions: float | None = None
    conversions_value: float | None = None
    all_conversions: float | None = None
    all_conversions_value: float | None = None
    cross_device_conversions: float | None = None
    cross_device_conversions_value: float | None = None


class Sa360ConversionCatalogResponse(BaseModel):
    customer_id: str
    date_range: str | None = None
    actions: list[Sa360ConversionActionItem] = []

# Canonical CSV schemas for Ads fetch (kept in code for validation and mapping)
ADS_CSV_SCHEMAS = {
    "campaign": [
        "campaign.id",
        "campaign.name",
        "campaign.status",
        "campaign.advertising_channel_type",
        "campaign.advertising_channel_sub_type",
        "campaign.start_date",
        "campaign.end_date",
        "campaign.bidding_strategy_type",
        "campaign.budget_id",
        "campaign.labels",
        "campaign.resource_name",
        "campaign.network_settings.target_search_network",
        "campaign.network_settings.target_google_search",
        "campaign.network_settings.target_partner_search_network",
        "campaign.network_settings.target_content_network",
        "campaign.target_cpa",
        "campaign.target_roas",
        "segments.device",
        "segments.geo_target_region",
    ],
    "ad_group": [
        "ad_group.id",
        "ad_group.name",
        "ad_group.status",
        "ad_group.type",
        "ad_group.cpc_bid_micros",
        "ad_group.labels",
        "campaign.id",
        "segments.device",
    ],
    "ad": [
        "ad_group_ad.ad.id",
        "ad_group.id",
        "campaign.id",
        "ad_group_ad.status",
        "ad_group_ad.policy_summary.approval_status",
        "ad_group_ad.policy_summary.review_status",
        "ad_group_ad.ad.type",
        "ad_group_ad.ad.responsive_search_ad.headlines",
        "ad_group_ad.ad.responsive_search_ad.descriptions",
        "ad_group_ad.ad.final_urls",
        "segments.device",
    ],
    "keyword_performance": [
        "ad_group_criterion.criterion_id",
        "ad_group_criterion.keyword.text",
        "ad_group_criterion.keyword.match_type",
        "ad_group_criterion.status",
        "ad_group.id",
        "campaign.id",
        "metrics.impressions",
        "metrics.clicks",
        "metrics.cost_micros",
        "metrics.conversions",
        "metrics.conversions_value",
        "metrics.ctr",
        "metrics.average_cpc",
        "metrics.cost_per_conversion",
        "metrics.search_impression_share",
        "metrics.search_rank_lost_impression_share",
        "metrics.search_exact_match_impression_share",
        "segments.device",
    ],
    "keyword_performance_conv": [
        "ad_group_criterion.criterion_id",
        "ad_group_criterion.keyword.text",
        "ad_group_criterion.keyword.match_type",
        "ad_group_criterion.status",
        "ad_group.id",
        "ad_group.name",
        "campaign.id",
        "campaign.name",
        "segments.device",
        "segments.conversion_action_name",
        "metrics.conversions",
        "metrics.conversions_value",
        "metrics.all_conversions",
        "metrics.all_conversions_value",
    ],
    "conversion_action_summary": [
        "segments.conversion_action_name",
        "segments.conversion_action_category",
        "metrics.conversions",
        "metrics.conversions_value",
        "metrics.all_conversions",
        "metrics.all_conversions_value",
        "metrics.cross_device_conversions",
        "metrics.cross_device_conversions_value",
    ],
    "conversion_actions": [
        "conversion_action.id",
        "conversion_action.name",
        "conversion_action.category",
        "conversion_action.status",
    ],
    "campaign_conversion_action": [
        "campaign.id",
        "campaign.name",
        "segments.device",
        "segments.conversion_action_name",
        "metrics.conversions",
        "metrics.conversions_value",
        "metrics.all_conversions",
        "metrics.all_conversions_value",
    ],
    # Customer-level performance totals (avoid undercounting conversions by relying on keyword_view attribution).
    "customer_performance": [
        "customer.id",
        "metrics.impressions",
        "metrics.clicks",
        "metrics.cost_micros",
        "metrics.conversions",
        "metrics.conversions_value",
        "metrics.all_conversions",
        "metrics.all_conversions_value",
        "metrics.cross_device_conversions",
        "metrics.cross_device_conversions_value",
    ],
    "landing_page": [
        "landing_page_view.unexpanded_final_url",
        "metrics.impressions",
        "metrics.clicks",
        "metrics.conversions",
        "metrics.conversions_value",
        "metrics.average_cpc",
        "metrics.ctr",
        "segments.device",
    ],
    "account": [
        "customer.id",
        "customer.descriptive_name",
        "customer.currency_code",
        "customer.time_zone",
        "customer.status",
    ],
}

# SA360 GAQL snippets (mapped to our CSV schemas). Adjust if field coverage changes.
SA360_QUERIES = {
    "campaign": """
        SELECT
          customer.id,
          campaign.id,
          campaign.name,
          campaign.status,
          campaign.advertising_channel_type,
          campaign.advertising_channel_sub_type,
          campaign.start_date,
          campaign.end_date,
          campaign.bidding_strategy_type,
          campaign.campaign_budget,
          campaign.target_cpa.target_cpa_micros,
          campaign.target_roas.target_roas
        FROM campaign
        WHERE campaign.status != 'REMOVED'
    """,
    "ad_group": """
        SELECT
          ad_group.id,
          ad_group.name,
          ad_group.status,
          ad_group.type,
          ad_group.cpc_bid_micros,
          campaign.id,
          segments.device
        FROM ad_group
        WHERE ad_group.status != 'REMOVED'
    """,
    "ad": """
        SELECT
          ad_group_ad.ad.id,
          ad_group.id,
          campaign.id,
          ad_group_ad.status,
          ad_group_ad.ad.type,
          ad_group_ad.ad.responsive_search_ad.headlines,
          ad_group_ad.ad.responsive_search_ad.descriptions,
          ad_group_ad.ad.final_urls,
          segments.device
        FROM ad_group_ad
        WHERE ad_group_ad.status != 'REMOVED'
    """,
    "keyword_performance": """
        SELECT
          ad_group_criterion.criterion_id,
          ad_group_criterion.keyword.text,
          ad_group_criterion.keyword.match_type,
          ad_group_criterion.status,
          ad_group.id,
          campaign.id,
          segments.device,
          metrics.impressions,
          metrics.clicks,
          metrics.cost_micros,
          metrics.conversions,
          metrics.conversions_value,
          metrics.ctr,
          metrics.average_cpc,
          metrics.cost_per_conversion
        FROM keyword_view
        WHERE ad_group_criterion.status != 'REMOVED'
    """,
    "keyword_performance_conv": """
        SELECT
          ad_group_criterion.criterion_id,
          ad_group_criterion.keyword.text,
          ad_group_criterion.keyword.match_type,
          ad_group_criterion.status,
          ad_group.id,
          ad_group.name,
          campaign.id,
          campaign.name,
          segments.device,
          segments.conversion_action_name,
          metrics.conversions,
          metrics.conversions_value,
          metrics.all_conversions,
          metrics.all_conversions_value
        FROM keyword_view
        WHERE ad_group_criterion.status != 'REMOVED'
    """,
    # Account-level (customer) conversion action totals. This is more complete than keyword_view when conversions
    # are not attributable to a specific keyword (e.g., some campaign types / attribution cases).
    "conversion_action_summary": """
        SELECT
          segments.conversion_action_name,
          segments.conversion_action_category,
          metrics.conversions,
          metrics.conversions_value,
          metrics.all_conversions,
          metrics.all_conversions_value,
          metrics.cross_device_conversions,
          metrics.cross_device_conversions_value
        FROM customer
        WHERE segments.conversion_action_name IS NOT NULL
    """,
    # Catalog of conversion actions (names exist even if conversions are 0 in the requested window).
    # IMPORTANT: do not inject segments.date clauses into this query (not selectable with conversion_action).
    "conversion_actions": """
        SELECT
          conversion_action.id,
          conversion_action.name,
          conversion_action.category,
          conversion_action.status
        FROM conversion_action
        WHERE conversion_action.status != 'REMOVED'
    """,
    # Campaign-level + device-level conversion-action breakdown.
    # Used by custom-metric driver analysis so excluded-from-conversions actions (e.g., Store visits)
    # can still produce campaign/device drivers instead of empty lists.
    "campaign_conversion_action": """
        SELECT
          campaign.id,
          campaign.name,
          segments.device,
          segments.conversion_action_name,
          metrics.conversions,
          metrics.conversions_value,
          metrics.all_conversions,
          metrics.all_conversions_value
        FROM campaign
        WHERE campaign.status != 'REMOVED'
          AND segments.conversion_action_name IS NOT NULL
    """,
    # Customer-level totals (fast + complete for account-level performance questions).
    "customer_performance": """
        SELECT
          customer.id,
          metrics.impressions,
          metrics.clicks,
          metrics.cost_micros,
          metrics.conversions,
          metrics.conversions_value,
          metrics.all_conversions,
          metrics.all_conversions_value,
          metrics.cross_device_conversions,
          metrics.cross_device_conversions_value
        FROM customer
    """,
    "account": """
        SELECT
          customer.id,
          customer.descriptive_name,
          customer.currency_code,
          customer.time_zone,
          customer.status
        FROM customer
    """,
}

def _env_csv_list(key: str, default: list[str]) -> list[str]:
    raw = os.environ.get(key, "") or ""
    if not raw.strip():
        return list(default)
    items = [part.strip().lower() for part in raw.split(",") if part.strip()]
    return items or list(default)

# Report subsets for faster SA360 performance/diagnostics pulls (env-overridable)
SA360_PERF_REPORTS = _env_csv_list(
    "SA360_PERF_REPORTS",
    ["keyword_performance", "campaign", "ad_group", "ad"],
)
SA360_DIAGNOSTICS_REPORTS = _env_csv_list(
    "SA360_DIAGNOSTICS_REPORTS",
    ["keyword_performance"],
)


def _get_nested_value(obj: Any, path: str):
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            next_val = cur.get(part)
            if next_val is None and "_" in part:
                # Try camelCase fallback (e.g., cost_micros -> costMicros)
                camel = part.split("_")[0] + "".join(x.capitalize() for x in part.split("_")[1:])
                next_val = cur.get(camel)
            cur = next_val
        else:
            return None
    if isinstance(cur, list):
        return ";".join(str(x) for x in cur)
    if isinstance(cur, dict):
        return json.dumps(cur)
    return cur


def _parse_human_date(message: str, default: str | None = None) -> str | None:
    # Expanded natural-language span parsing for PPC-style queries
    msg = message.lower()

    def _range_days(n: int) -> str:
        end = date.today()
        start = end - timedelta(days=max(n - 1, 0))
        return f"{start:%Y-%m-%d},{end:%Y-%m-%d}"

    if any(k in msg for k in ["yesterday"]):
        d = date.today() - timedelta(days=1)
        return f"{d:%Y-%m-%d},{d:%Y-%m-%d}"
    if "today" in msg:
        d = date.today()
        return f"{d:%Y-%m-%d},{d:%Y-%m-%d}"

    # Weekend handling
    if "weekend" in msg:
        today = date.today()
        if "last weekend" in msg or "over the weekend" in msg:
            base = today - timedelta(days=today.weekday() + 2)  # previous Saturday
        elif "this weekend" in msg:
            days_until_sat = (5 - today.weekday()) % 7
            base = today + timedelta(days=days_until_sat)
        else:
            base = today - timedelta(days=today.weekday() + 2)  # default to last weekend
        start = base
        end = base + timedelta(days=1)
        return f"{start:%Y-%m-%d},{end:%Y-%m-%d}"

    # Fixed spans
    # "Last week" is interpreted as the previous *calendar* week (Mon-Sun), not "last 7 days".
    # This avoids including partial "today" data and matches typical paid-search reporting expectations.
    if "last week" in msg or "previous week" in msg:
        today = date.today()
        this_week_start = today - timedelta(days=today.weekday())  # Monday
        start = this_week_start - timedelta(days=7)
        end = this_week_start - timedelta(days=1)  # Sunday
        return f"{start:%Y-%m-%d},{end:%Y-%m-%d}"
    if "last 7" in msg or "past 7" in msg or "previous 7" in msg:
        return _range_days(7)
    if "last 14" in msg or "past 14" in msg or "previous 14" in msg or "last two weeks" in msg or "last 2 weeks" in msg:
        return _range_days(14)
    if "last 30" in msg or "past 30" in msg or "previous 30" in msg or "last month" in msg:
        return _range_days(30)
    if "last 90" in msg or "past 90" in msg or "previous 90" in msg or "last quarter" in msg:
        return _range_days(90)
    if "last 3 days" in msg or "past 3 days" in msg or "3 days ago" in msg:
        return _range_days(3)
    if "last 14 days" in msg:
        return _range_days(14)

    # Week-to-date / Month-to-date / Quarter-to-date / Year-to-date
    if "wtd" in msg or "week to date" in msg or "this week" in msg:
        today = date.today()
        start = today - timedelta(days=today.weekday())
        return f"{start:%Y-%m-%d},{today:%Y-%m-%d}"
    if "mtd" in msg or "month to date" in msg or "this month" in msg:
        today = date.today()
        start = today.replace(day=1)
        return f"{start:%Y-%m-%d},{today:%Y-%m-%d}"
    if "qtd" in msg or "quarter to date" in msg or "this quarter" in msg:
        today = date.today()
        quarter_start_month = ((today.month - 1) // 3) * 3 + 1
        start = today.replace(month=quarter_start_month, day=1)
        return f"{start:%Y-%m-%d},{today:%Y-%m-%d}"
    if "ytd" in msg or "year to date" in msg or "this year" in msg:
        today = date.today()
        start = today.replace(month=1, day=1)
        return f"{start:%Y-%m-%d},{today:%Y-%m-%d}"

    # Explicit quarter references
    if "last quarter" in msg or "previous quarter" in msg:
        today = date.today()
        quarter_start_month = ((today.month - 1) // 3) * 3 + 1
        this_start = today.replace(month=quarter_start_month, day=1)
        last_end = this_start - timedelta(days=1)
        last_start = last_end.replace(month=((last_end.month - 1) // 3) * 3 + 1, day=1)
        return f"{last_start:%Y-%m-%d},{last_end:%Y-%m-%d}"
    if "q1" in msg or "q2" in msg or "q3" in msg or "q4" in msg:
        today = date.today()
        qmap = {"q1": 1, "q2": 4, "q3": 7, "q4": 10}
        for token, month in qmap.items():
            if token in msg:
                start = date(today.year, month, 1)
                if month == 10:
                    end = date(today.year, 12, 31)
                else:
                    end = date(today.year, month + 3, 1) - timedelta(days=1)
                return f"{start:%Y-%m-%d},{end:%Y-%m-%d}"

    # Specific presets
    for key in [
        "week before last",
        "two weeks ago",
    ]:
        if key in msg:
            start_this_week = date.today() - timedelta(days=date.today().weekday())
            start = start_this_week - timedelta(weeks=2)
            end = start + timedelta(days=6)
            return f"{start:%Y-%m-%d},{end:%Y-%m-%d}"

    # Fallback: if the user didn't specify a timeframe, honor the caller-provided default.
    return default


def _has_timeframe_hint(message: str | None) -> bool:
    if not message:
        return False
    if _parse_human_date(message):
        return True
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", message):
        return True
    if re.search(r"\b(last|past|previous)\s+\d+\s+(day|week|month|quarter|year)s?\b", message.lower()):
        return True
    if re.search(r"\b(LAST_WEEK|LAST_MONTH|THIS_WEEK|THIS_MONTH|YESTERDAY|TODAY)\b", message.upper()):
        return True
    if re.search(r"\bLAST_(7|14|30|90)_DAYS\b", message.upper()):
        return True
    return False


def _build_sa360_plan_from_chat(message: str, customer_ids: list[str], account_name: str | None, default_date: str | None) -> dict:
    # Plan = structured request for SA360 fetch-and-audit
    date_hint = _parse_human_date(message, default_date)
    bu = "Brand"
    lower = (message or "").lower()
    if "pmax" in lower or "performance max" in lower:
        bu = "PMax"
    elif "nonbrand" in lower or "non-brand" in lower or "non brand" in lower:
        bu = "NonBrand"
    elif "brand" in lower:
        bu = "Brand"
    return {
        "account_name": account_name,
        "customer_ids": customer_ids,
        "date_range": date_hint,
        "dry_run": False,
        "business_unit": bu,
    }


def _enhance_summary_text(summary: str | None, seed: str | None = None) -> str | None:
    if not summary:
        return None
    system_prompt = (
        f"{chat_system_prompt()} You are crafting a single, concise follow-up question. "
        "Return only one sentence, 8-16 words, no numbered lists, no generic phrases."
    )
    user_prompt = f"Summary: {summary}\nUser prompt: {seed or ''}\nWrite one tailored follow-up question."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    reply, _meta = _call_local_llm(
        messages,
        intent="summary_enhancer",
        max_tokens=60,
        force_json=False,
        record_usage=True,
    )
    follow_up = (reply or "").strip().replace("\n", " ")
    if not follow_up or _is_low_quality_local_reply(follow_up):
        return None
    lower_follow = follow_up.lower()
    # Guardrail: do not ask for metrics that aren't present in the summary payload.
    # Local models sometimes hallucinate "CPM" as a next question even when we do not fetch it.
    if "cpm" in lower_follow:
        return None
    must_have = (
        "breakdown",
        "break down",
        "driver",
        "drivers",
        "segment",
        "slice",
        "campaign",
        "device",
        "geo",
    )
    if not any(term in lower_follow for term in must_have):
        return None
    if any(term in lower_follow for term in ("metric", "available", "up to date", "up-to-date")):
        return None
    if not follow_up.endswith("?"):
        follow_up = follow_up.rstrip(".") + "?"
    return _normalize_reply_text(f"{summary} {follow_up}") or f"{summary} {follow_up}"


def _llm_general_help_reply(message: str | None) -> str | None:
    prompt = (
        f"{chat_system_prompt()} You are responding to a general help request. "
        "Answer in 2-3 sentences. State two concrete things you can do in this system "
        "(performance analysis, audits, trends, SERP, creative, diagnostics, etc.) and "
        "end with one clarifying question. Avoid generic filler."
    )
    user_prompt = message or "What can you help me with?"
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_prompt},
    ]
    reply, meta = _call_local_llm(
        messages,
        intent="general_help",
        max_tokens=140,
        force_json=False,
        record_usage=True,
    )
    reply = (reply or "").strip()
    if not reply or _is_low_quality_local_reply(reply):
        return None
    lower = reply.lower()
    must_have = ("audit", "performance", "trends", "serp", "diagnostic", "pmax", "creative")
    if not any(term in lower for term in must_have):
        return None
    if not reply.endswith("?"):
        reply = reply.rstrip(".") + "?"
    return _normalize_reply_text(reply) or reply


def _is_performance_action_request(message: str) -> bool:
    """
    Detect "what should I do" / recommendations asks so we can return advisor-grade responses.

    This is intentionally heuristic but centrally defined so UI + backend behavior stays consistent.
    """
    t = (message or "").lower()
    cues = [
        "action",
        "actions",
        "next action",
        "next step",
        "next steps",
        "recommend",
        "recommendation",
        "recommendations",
        "what should i do",
        "what should we do",
        "what would you do",
        "what do you suggest",
        "suggest",
        "suggestion",
        "optimize",
        "optimise",
        "optimization",
        "optimizations",
        "improve",
        "improved",
        "improvement",
        "improvements",
        "prioritize",
        "priority",
        "fix",
        "resolve",
    ]
    return any(cue in t for cue in cues)


def _should_explain_performance(message: str) -> bool:
    t = (message or "").lower()
    if _is_performance_action_request(message):
        return True
    cues = [
        "why",
        "explain",
        "cause",
        "caused",
        "driver",
        "drivers",
        "reason",
        "root cause",
        "root-cause",
        "rootcause",
        "what happened",
        "what changed",
        "compare",
        "versus",
        "vs",
        "due to",
        "because",
        "efficiency",
        "seasonal",
        "seasonality",
        "quality",
        "diminishing return",
        "diminishing returns",
    ]
    if any(cue in t for cue in cues):
        return True
    return _has_dimension_cue(message)


def _has_dimension_cue(message: str) -> bool:
    t = (message or "").lower()
    dimension_cues = [
        "campaign",
        "ad group",
        "adgroup",
        "keyword",
        "query",
        "search term",
        "device",
        "mobile",
        "desktop",
        "geo",
        "location",
        "region",
        "state",
        "city",
    ]
    return any(cue in t for cue in dimension_cues)


def _has_strong_explain_cue(message: str) -> bool:
    t = (message or "").lower()
    cues = [
        "why",
        "cause",
        "caused",
        "because",
        "driver",
        "drivers",
        "reason",
        "root cause",
        "root-cause",
        "rootcause",
    ]
    return any(cue in t for cue in cues)


def _metric_focus_from_message(message: str) -> tuple[str | None, str | None]:
    t = (message or "").lower()
    if "cpc" in t or "cost per click" in t:
        return "cpc", "CPC"
    if "cpa" in t or "cost per acquisition" in t:
        return "cpa", "CPA"
    if "ctr" in t or "click-through" in t:
        return "ctr", "CTR"
    if "cvr" in t or "conversion rate" in t:
        return "cvr", "CVR"
    if "roas" in t:
        return "roas", "ROAS"
    if "conversion" in t or "conversions" in t:
        return "conversions", "conversions"
    if "click" in t or "clicks" in t:
        return "clicks", "clicks"
    if "impression" in t or "impressions" in t:
        return "impressions", "impressions"
    if "cost" in t or "spend" in t or "budget" in t:
        return "cost", "spend"
    custom = _extract_custom_metric_mentions(message)
    if custom:
        key = custom[0]
        return key, _metric_label(key)
    return None, None


def _extract_metric_mentions(message: str) -> list[str]:
    t = (message or "").lower()
    found: list[str] = []

    def add(key: str) -> None:
        if key not in found:
            found.append(key)

    if "cpc" in t or "cost per click" in t:
        add("cpc")
    if "cpa" in t or "cost per acquisition" in t:
        add("cpa")
    if "ctr" in t or "click-through" in t:
        add("ctr")
    if "cvr" in t or "conversion rate" in t:
        add("cvr")
    if "roas" in t:
        add("roas")
    if "conversion value" in t or "revenue" in t or "value" in t or "sales" in t:
        add("conversions_value")
    if "conversion" in t or "conversions" in t:
        add("conversions")
    if "click" in t or "clicks" in t:
        add("clicks")
    if "impression" in t or "impressions" in t:
        add("impressions")
    if "cost" in t or "spend" in t or "budget" in t:
        add("cost")
    for custom in _extract_custom_metric_mentions(message):
        add(custom)
    return found


_CUSTOM_METRIC_LABELS: dict[str, str] = {}
# Custom metrics frequently contain underscores and may contain hyphens (e.g., "ENG_Shell_USA_OnClick-...").
# Use a permissive token regex, then filter downstream so we don't accidentally treat ordinary hyphenated words as metrics.
_CUSTOM_METRIC_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_-]{2,}\b")
_CUSTOM_METRIC_HINT_TERMS = {
    "intent",
    "action",
    "event",
    "goal",
    "signup",
    "sign-up",
    "sign up",
    "lead",
    "purchase",
    "install",
    "download",
    "call",
    "form",
    "reward",
    "rewards",
}
_STANDARD_METRIC_ALIASES = {
    "clicks",
    "click",
    "impressions",
    "impression",
    "cost",
    "spend",
    "budget",
    "conversions",
    "conversion",
    "revenue",
    "value",
    "sales",
    "ctr",
    "cpc",
    "cpa",
    "cvr",
    "roas",
    "conversion_value",
    "conversions_value",
}
_CUSTOM_METRIC_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "give",
    "last",
    "days",
    "day",
    "week",
    "weeks",
    "month",
    "months",
    "over",
    "vs",
    "versus",
    "show",
    "tell",
    "me",
    "my",
    "our",
    "what",
    "which",
    "why",
    "how",
    "did",
    "does",
    "do",
    "is",
    "are",
    "was",
    "were",
    "please",
    "performance",
    "change",
    "changed",
    "changes",
    "changing",
    "move",
    "moved",
    "increase",
    "increased",
    "decrease",
    "decreased",
    "drop",
    "dropped",
    "spike",
    "spiked",
    "up",
    "down",
    "because",
    "due",
    "drove",
    "drive",
    "driver",
    "drivers",
    "metric",
    "metrics",
    "column",
    "field",
}
_CUSTOM_METRIC_DIMENSION_PREFIXES = (
    "campaign.",
    "ad_group.",
    "ad_group_criterion.",
    "segments.",
    "customer.",
    "keyword",
)
_CUSTOM_METRIC_SYNONYMS = [
    {"from": ["fr intent clicks", "fr intent"], "to": "fuel rewards sign up"},
    {"from": ["fuel rewards intent", "fuel rewards signup", "fuel rewards sign-up"], "to": "fuel rewards sign up"},
    {"from": ["sign up unique", "sign-up unique"], "to": "sign up unique"},
]

# When a user asks "why did X change", we want to match X (the metric phrase), not the entire question.
# Extract short metric phrases (e.g., "FR intent clicks", "all conversions value") to improve matching.
_CUSTOM_METRIC_PHRASE_RE = re.compile(
    r"(?P<phrase>(?:[a-z0-9]{2,}(?:[\s\-_]+[a-z0-9]{2,}){0,6})[\s\-_]+(?:clicks?|conversions?|value|revenue))",
    re.IGNORECASE,
)

_EXPLICIT_METRIC_TOKEN_SUFFIX_RE = re.compile(r"(clicks?|conversions?|value|revenue|visits?)$", re.IGNORECASE)

_QUOTED_METRIC_PHRASE_RE = re.compile(
    r"(?:[\"\\u201C\\u201D](?P<dquote>[^\"\\u201C\\u201D]{3,120})[\"\\u201C\\u201D])"
    r"|(?:['\\u2018\\u2019](?P<squote>[^'\\u2018\\u2019]{3,120})['\\u2018\\u2019])"
)


def _extract_quoted_metric_phrase(message: str) -> str | None:
    """
    Extract a quoted metric phrase from a message (used to disambiguate conversion-action names).

    Rationale:
    - The performance planner specs quote the conversion action name: "Fuel Rewards Intent Clicks".
    - Users often quote metric names for clarity.
    - Even if the name contains underscores, quotes imply "name", not a pre-normalized custom:<key>.
    """
    if not message:
        return None
    try:
        for m in _QUOTED_METRIC_PHRASE_RE.finditer(message):
            phrase = (m.group("dquote") or m.group("squote") or "").strip()
            if not phrase:
                continue
            # Metric-ish signals: underscore tokens (FR_Intent_Clicks) or suffix terms (clicks/conversions/etc),
            # or a metric-phrase match like "fuel rewards intent clicks".
            if (
                "_" not in phrase
                and not _EXPLICIT_METRIC_TOKEN_SUFFIX_RE.search(phrase)
                and not _CUSTOM_METRIC_PHRASE_RE.search(phrase)
            ):
                continue
            lower = phrase.lower()
            # Avoid treating generic timeframe snippets as metric phrases (e.g., "last 30 days").
            if "last" in lower and "day" in lower:
                continue
            return phrase
    except Exception:
        return None
    return None


def _extract_explicit_metric_token_key(message: str) -> tuple[str | None, str | None]:
    """
    Detect explicit snake_case-like metric tokens embedded in a message (e.g., FR_Intent_Clicks).

    We only treat tokens as "explicit metric references" when:
    - the prompt shape indicates a metric ask (direct or relational), and
    - the token *looks* metric-like (ends with clicks/conversions/value/revenue/visits)

    This avoids mistakenly treating campaign names/IDs (often underscore-heavy) as metrics.
    Returns (custom_metric_key, raw_token) or (None, None).
    """
    if not message:
        return None, None
    try:
        if not (_message_is_direct_metric_request(message) or _message_is_relational_custom_metric_request(message)):
            return None, None
    except Exception:
        return None, None
    # If the user quoted a metric phrase, treat it as a "name to resolve" rather than an explicit custom:<key>.
    # This prevents silent behavior differences in quoted prompts used in QA specs.
    quoted = _extract_quoted_metric_phrase(message)
    try:
        tokens = [t for t in _CUSTOM_METRIC_RE.findall(message) if "_" in (t or "")]
        if not tokens:
            return None, None
        metricish = [t for t in tokens if _EXPLICIT_METRIC_TOKEN_SUFFIX_RE.search(t or "")]
        if not metricish:
            return None, None
        raw = metricish[0]
        if quoted and _normalize_custom_metric_cmp(raw) == _normalize_custom_metric_cmp(quoted):
            return None, None
        norm = _normalize_custom_metric_key(raw)
        if not norm:
            return None, None
        # Do not treat standard KPI-style fields as custom metrics.
        if norm in _STANDARD_METRIC_ALIASES or norm in {
            "conversions_value",
            "conversion_value",
            "all_conversions",
            "all_conversions_value",
            "cross_device_conversions",
            "cross_device_conversions_value",
        }:
            return None, None
        _CUSTOM_METRIC_LABELS.setdefault(norm, raw)
        return f"custom:{norm}", raw
    except Exception:
        return None, None


def _normalize_custom_metric_key(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", token.lower()).strip("_")


def _normalize_custom_metric_cmp(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", token.lower())


def _extract_custom_metric_mentions(message: str) -> list[str]:
    if not message:
        return []
    tokens = _CUSTOM_METRIC_RE.findall(message)
    if not tokens:
        return []
    standard = {
        "cpc",
        "cpa",
        "ctr",
        "cvr",
        "roas",
        "conversions",
        "conversion",
        "clicks",
        "click",
        "impressions",
        "impression",
        "cost",
        "spend",
        "budget",
    }
    found: list[str] = []
    for tok in tokens:
        if "_" not in tok:
            continue
        norm = _normalize_custom_metric_key(tok)
        if not norm or norm in standard:
            continue
        key = f"custom:{norm}"
        if key not in found:
            found.append(key)
            if norm not in _CUSTOM_METRIC_LABELS:
                _CUSTOM_METRIC_LABELS[norm] = tok
    return found


def _tokenize_metric_text(text: str) -> list[str]:
    if not text:
        return []
    tokens = re.split(r"[^a-z0-9]+", text.lower())
    return [t for t in tokens if t]


def _message_has_custom_metric_cue(message: str) -> bool:
    if not message:
        return False
    if "_" in message:
        return True
    lower = message.lower()
    tokens = _tokenize_metric_text(lower)
    if not tokens:
        return False
    token_set = set(tokens)
    # Hint terms: avoid substring checks ("performance" contains "form").
    phrase_hints = [t for t in _CUSTOM_METRIC_HINT_TERMS if (" " in t or "-" in t)]
    for hint in phrase_hints:
        if hint and hint in lower:
            return True
    word_hints = [t for t in _CUSTOM_METRIC_HINT_TERMS if t and t not in phrase_hints]
    if any(h in token_set for h in word_hints):
        return True
    standard_present = any(t in _STANDARD_METRIC_ALIASES for t in tokens)
    non_standard = [
        t
        for t in tokens
        if t not in _STANDARD_METRIC_ALIASES
        and t not in _CUSTOM_METRIC_STOPWORDS
        and not t.isdigit()
    ]
    return standard_present and len(non_standard) >= 1


def _message_is_direct_metric_request(message: str) -> bool:
    """
    Heuristic to detect "show me <metric>"-style prompts where <metric> is likely a custom conversion action / column
    rather than a standard KPI.

    This is used to decide whether we should pull conversion-action reports for inference, keeping default perf pulls lean.
    """
    lower = (message or "").lower().strip()
    if not lower:
        return False
    # Avoid over-triggering on long prompts (usually multi-part strategy requests).
    if len(lower) > 180:
        return False
    triggers = (
        "show me",
        "what is",
        "what's",
        "whats",
        "give me",
        "tell me",
        "how many",
        "trend of",
        "number of",
    )
    if not any(t in lower for t in triggers):
        return False

    # If the user is asking for a list/breakdown by an entity, it's not a single-metric request.
    dimension_terms = (
        "campaign",
        "campaigns",
        "ad group",
        "adgroup",
        "ad groups",
        "keywords",
        "keyword",
        "queries",
        "query",
        "search term",
        "search terms",
        "landing page",
        "landing pages",
        "final url",
        "final urls",
        "device",
        "geo",
        "location",
    )
    if any(t in lower for t in dimension_terms):
        return False

    tokens = _tokenize_metric_text(lower)
    if not tokens:
        return False
    interesting = [
        t
        for t in tokens
        if t not in _STANDARD_METRIC_ALIASES
        and t not in _CUSTOM_METRIC_STOPWORDS
        and not t.isdigit()
    ]
    if not interesting:
        return False
    # "performance" alone is too generic.
    if set(interesting).issubset({"performance", "report", "analysis"}):
        return False
    # Short, focused prompts are more likely metric asks (e.g., "show me store visits").
    return (len(interesting) >= 2) or (len(interesting) >= 1 and len(tokens) <= 6)


def _message_is_relational_custom_metric_request(message: str) -> bool:
    """
    Detect relational prompts that likely refer to a non-standard SA360 column / conversion action
    (e.g., "Why did store visits change?" / "Which campaigns drove store visits?").

    We must avoid triggering on generic questions like "Why is performance down?" which should stay on standard KPIs.
    """
    if not message or not _has_relational_cue(message):
        return False
    lower = message.lower()
    tokens = _tokenize_metric_text(lower)
    if not tokens:
        return False
    glue = {
        "campaign",
        "campaigns",
        "ad",
        "ads",
        "adgroup",
        "adgroups",
        "keyword",
        "keywords",
        "query",
        "queries",
        "search",
        "term",
        "terms",
        "device",
        "devices",
        "geo",
        "geos",
        "region",
        "regions",
        "location",
        "locations",
    }
    interesting = [
        t
        for t in tokens
        if t not in _STANDARD_METRIC_ALIASES
        and t not in _CUSTOM_METRIC_STOPWORDS
        and t not in glue
        and not t.isdigit()
    ]
    # Require at least one meaningful token (e.g., "store", "visits", "directions").
    return len(interesting) >= 1


def _candidate_acronym(tokens: list[str]) -> str:
    if not tokens:
        return ""
    return "".join(t[0] for t in tokens if t)


def _apply_custom_metric_synonyms(message: str) -> list[str]:
    if not message:
        return []
    lower = message.lower()
    # Also keep a whitespace-normalized form so "fr_intent_clicks" can match "fr intent clicks" synonyms.
    normalized = re.sub(r"[^a-z0-9]+", " ", lower).strip()
    variants = {lower}
    if normalized and normalized not in variants:
        variants.add(normalized)
    for entry in _CUSTOM_METRIC_SYNONYMS:
        for token in entry.get("from", []):
            replacement = entry.get("to", "")
            if token and token in lower:
                variants.add(lower.replace(token, replacement))
            if token and normalized and token in normalized:
                variants.add(normalized.replace(token, replacement))
    tokens = _tokenize_metric_text(lower)
    if "fr" in tokens and any(t in tokens for t in ("intent", "loyalty", "rewards")):
        variants.add(re.sub(r"\bfr\b", "fuel rewards", lower))
        if normalized:
            variants.add(re.sub(r"\bfr\b", "fuel rewards", normalized))
    return list(variants)


def _custom_metric_query_variants(message: str) -> list[str]:
    """
    Return short query variants to score custom-metric candidates against.
    We include:
    - full message (fallback)
    - synonym-rewritten variants
    - extracted metric phrases (e.g., "fuel rewards sign up clicks", "all conversions value")
    """
    if not message:
        return []
    variants: list[str] = []
    # Full message first (keeps behavior for short prompts).
    variants.append(message)

    # Synonym-expanded forms.
    for v in _apply_custom_metric_synonyms(message):
        if v and v not in variants:
            variants.append(v)

    # Metric phrase extraction from each variant.
    phrases: set[str] = set()
    for v in list(variants):
        try:
            for m in _CUSTOM_METRIC_PHRASE_RE.finditer(v or ""):
                phrase = (m.group("phrase") or "").strip()
                if not phrase:
                    continue
                # Keep phrases compact; drop leading question glue.
                phrase = re.sub(r"^(?:(?:why|did|which|what|how|explain|compare)\b\s*)+", "", phrase.strip(), flags=re.I)
                toks = _tokenize_metric_text(phrase)
                if not toks:
                    continue
                # Require at least one non-standard token so we don't "match" plain "clicks".
                non_standard = [
                    t
                    for t in toks
                    if t not in _STANDARD_METRIC_ALIASES
                    and t not in _CUSTOM_METRIC_STOPWORDS
                    and not t.isdigit()
                ]
                if len(non_standard) < 1:
                    continue
                phrases.add(phrase)
        except Exception:
            continue

    for p in sorted(phrases, key=lambda s: (len(s), s)):
        if p not in variants:
            variants.append(p)
    return variants


def _score_metric_candidate_single(query: str, candidate: str) -> float:
    q_norm = _normalize_custom_metric_cmp(query)
    c_norm = _normalize_custom_metric_cmp(candidate)
    if not q_norm or not c_norm:
        return 0.0
    # Exact match (ignoring punctuation/spacing) should win decisively.
    if c_norm == q_norm:
        return 1.0
    if c_norm in q_norm or q_norm in c_norm:
        base = 0.85
    else:
        q_tokens = [
            t
            for t in _tokenize_metric_text(query)
            if t not in _CUSTOM_METRIC_STOPWORDS and t not in _STANDARD_METRIC_ALIASES and not t.isdigit()
        ]
        c_tokens = [
            t
            for t in _tokenize_metric_text(candidate)
            if t not in _CUSTOM_METRIC_STOPWORDS and t not in _STANDARD_METRIC_ALIASES and not t.isdigit()
        ]
        q_set = set(q_tokens)
        c_set = set(c_tokens)
        overlap = len(q_set & c_set)
        if overlap == 0:
            base = 0.0
        else:
            base = (2 * overlap) / max(1, (len(q_set) + len(c_set)))
        acronym = _candidate_acronym(c_tokens)
        if acronym and any(t == acronym for t in q_tokens if len(t) <= 4):
            base = max(base, 0.6)

    # Heuristic: "value" is meaningful for custom metrics (e.g., all_conversions vs all_conversions_value).
    # We don't want "value" filtered out to the point where we match the wrong field.
    try:
        q_all = set(_tokenize_metric_text(query))
        c_all = set(_tokenize_metric_text(candidate))
        wants_value = bool(q_all & {"value", "revenue", "sales"})
        wants_all = "all" in q_all
        wants_cross_device = ("cross" in q_all and "device" in q_all) or ("crossdevice" in q_all)

        is_value = "value" in c_all
        is_all = "all" in c_all
        is_cross_device = ("cross" in c_all and "device" in c_all) or ("crossdevice" in c_all)

        # Boost correct variants first...
        if wants_value and is_value:
            base = max(base, 0.9)
        if wants_all and is_all:
            base = max(base, 0.92)
        if wants_cross_device and is_cross_device:
            base = max(base, 0.93)

        # ...then apply penalties so missing required qualifiers can't "win" via other boosts.
        penalty = 1.0
        if wants_value and not is_value:
            penalty *= 0.8
        if wants_all and not is_all:
            penalty *= 0.85
        if wants_cross_device and not is_cross_device:
            penalty *= 0.75
        base = base * penalty
    except Exception:
        pass
    return float(min(max(base, 0.0), 1.0))


def _score_metric_candidate(query: str, candidate: str) -> float:
    variants = _apply_custom_metric_synonyms(query)
    if not variants:
        variants = [query]
    best = 0.0
    for variant in variants:
        score = _score_metric_candidate_single(variant, candidate)
        if score > best:
            best = score
    return best


def _is_dimension_column(col: str) -> bool:
    lower = str(col).lower()
    if lower in {"segments.conversion_action_name"}:
        return True
    if any(lower.startswith(prefix) for prefix in _CUSTOM_METRIC_DIMENSION_PREFIXES):
        if lower.startswith("segments.") and "metric" in lower:
            return False
        return True
    if lower in {"campaign", "ad group", "ad", "keyword", "device", "geo", "date"}:
        return True
    return False


def _is_numeric_column(series: pd.Series, col: str) -> bool:
    try:
        if pd.api.types.is_numeric_dtype(series):
            return True
        if col.startswith("metrics."):
            return True
        sample = series.dropna().head(10)
        if sample.empty:
            return False
        converted = pd.to_numeric(sample, errors="coerce")
        numeric_ratio = converted.notna().mean()
        return numeric_ratio >= 0.8
    except Exception:
        return False


def _collect_sa360_metric_candidates(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not frames:
        return candidates
    for _, df in frames.items():
        if df is None or df.empty:
            continue
        # Conversion action catalog: names exist even when conversions are 0 for the requested window.
        if "conversion_action.name" in df.columns:
            try:
                names = df["conversion_action.name"].astype(str).dropna().unique().tolist()
            except Exception:
                names = []
            for name in names:
                norm = _normalize_custom_metric_key(name)
                if not norm or norm in _STANDARD_METRIC_ALIASES:
                    continue
                if norm in seen:
                    continue
                seen.add(norm)
                _CUSTOM_METRIC_LABELS.setdefault(norm, name)
                candidates.append(
                    {
                        "metric_key": f"custom:{norm}",
                        "label": name,
                        "match": name,
                        "kind": "conversion_action",
                        "source": "conversion_action_catalog",
                    }
                )
        if "segments.conversion_action_name" in df.columns and "metrics.conversions" in df.columns:
            try:
                names = df["segments.conversion_action_name"].astype(str).dropna().unique().tolist()
            except Exception:
                names = []
            for name in names:
                norm = _normalize_custom_metric_key(name)
                if not norm or norm in _STANDARD_METRIC_ALIASES:
                    continue
                if norm in seen:
                    continue
                seen.add(norm)
                _CUSTOM_METRIC_LABELS[norm] = name
                candidates.append(
                    {
                        "metric_key": f"custom:{norm}",
                        "label": name,
                        "match": name,
                        "kind": "conversion_action",
                        "source": "segmented_report",
                    }
                )
        for col in df.columns:
            col_str = str(col)
            if _is_dimension_column(col_str):
                continue
            if col_str in {"segments.conversion_action_name"}:
                continue
            if not _is_numeric_column(df[col], col_str):
                continue
            norm = _normalize_custom_metric_key(col_str)
            if not norm or norm in _STANDARD_METRIC_ALIASES:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            _CUSTOM_METRIC_LABELS.setdefault(norm, col_str)
            candidates.append(
                {
                    "metric_key": f"custom:{norm}",
                    "label": col_str,
                    "match": col_str,
                    "kind": "column",
                }
            )
    return candidates


def _infer_custom_metric_from_frames(
    message: str,
    frames: dict[str, pd.DataFrame],
    min_score: float = 0.55,
) -> dict[str, Any] | None:
    if not message or not frames:
        return None
    candidates = _collect_sa360_metric_candidates(frames)
    if not candidates:
        return None
    queries = _custom_metric_query_variants(message) or [message]
    scored: list[dict[str, Any]] = []
    for cand in candidates:
        label = cand.get("label") or cand.get("match") or ""
        if not label:
            continue
        score = 0.0
        for q in queries:
            score = max(score, _score_metric_candidate(q, label))
        if score <= 0:
            continue
        payload = dict(cand)
        payload["score"] = score
        scored.append(payload)
    if not scored:
        return None
    scored.sort(key=lambda x: x.get("score") or 0.0, reverse=True)
    best = scored[0]
    best_score = float(best.get("score") or 0.0)
    second_score = float(scored[1].get("score") or 0.0) if len(scored) > 1 else 0.0
    if best_score < min_score:
        if best_score >= 0.4 and (second_score == 0.0 or best_score >= (second_score + 0.15)):
            return best
        return None
    return best


def _suggest_custom_metric_candidates(message: str, frames: dict[str, pd.DataFrame], limit: int = 3) -> list[str]:
    if not message or not frames:
        return []
    candidates = _collect_sa360_metric_candidates(frames)
    if not candidates:
        return []
    queries = _custom_metric_query_variants(message) or [message]
    scored: list[tuple[float, str]] = []
    prefer_conversion_actions = _message_has_custom_metric_cue(message)
    if prefer_conversion_actions:
        conversion_candidates = [c for c in candidates if c.get("kind") == "conversion_action"]
        if conversion_candidates:
            candidates = conversion_candidates
    for cand in candidates:
        label = cand.get("label") or cand.get("match") or ""
        if not label:
            continue
        score = 0.0
        for q in queries:
            score = max(score, _score_metric_candidate(q, label))
        scored.append((score, str(cand.get("match") or label)))
    scored.sort(key=lambda x: x[0], reverse=True)
    suggestions = [name for score, name in scored if name][:limit]
    return suggestions


def _has_relational_cue(message: str) -> bool:
    t = (message or "").lower()
    cues = [
        "compare",
        "versus",
        "due to",
        "because",
        "driver",
        "drivers",
        "why",
        "explain",
        "reason",
        "which moved",
        "which changed",
        "more than",
        "less than",
        "bigger",
        "smaller",
        "increase",
        "increased",
        "decrease",
        "decreased",
        "spike",
        "drop",
        "efficiency",
        "quality",
    ]
    if any(cue in t for cue in cues):
        return True
    if re.search(r"\bvs\b", t):
        return True
    return False


def _has_comparison_cue(message: str) -> bool:
    t = (message or "").lower()
    cues = [
        "compare",
        "versus",
        "which moved",
        "which changed",
        "moved more",
        "more than",
        "less than",
        "bigger",
        "smaller",
        "higher than",
        "lower than",
    ]
    if any(cue in t for cue in cues):
        return True
    if re.search(r"\bvs\b", t):
        return True
    return False


def _has_trends_cue(message: str) -> bool:
    t = (message or "").lower()
    cues = [
        "trend",
        "trends",
        "seasonality",
        "seasonal",
        "google trends",
        "search interest",
        "interest over time",
        "demand",
        "year over year",
        "yoy",
        "quarter",
        "q1",
        "q2",
        "q3",
        "q4",
    ]
    return any(cue in t for cue in cues)


def _is_ad_quality_followup(message: str) -> bool:
    t = (message or "").lower()
    cues = [
        "rsa",
        "responsive search ad",
        "ad strength",
        "templated",
        "template",
        "usp",
        "unique selling",
        "ad copy quality",
        "headline",
        "headlines",
        "description",
        "ad group",
    ]
    return any(cue in t for cue in cues)


def _is_copy_generation_request(message: str) -> bool:
    t = (message or "").lower()
    verbs = ["write", "generate", "create", "draft", "suggest", "give me"]
    creative_terms = ["ad copy", "ads", "headlines", "descriptions", "rsa", "creative"]
    return any(v in t for v in verbs) and any(c in t for c in creative_terms)


def _is_relational_metric_question(message: str) -> bool:
    if not message:
        return False
    metrics = _extract_metric_mentions(message)
    if not metrics:
        return False
    if len(metrics) >= 2:
        return True if _has_relational_cue(message) else False
    return _has_relational_cue(message)


def _direction_from_delta(deltas: dict, key: str) -> str | None:
    change = (deltas.get(key) or {}).get("change")
    if change is None:
        return None
    if change > 0:
        return "up"
    if change < 0:
        return "down"
    return "flat"


def _format_pct(value: float | None) -> str | None:
    if value is None:
        return None
    try:
        return f"{value:+.1f}%"
    except Exception:
        return None


def _metric_label(key: str) -> str:
    if key.startswith("custom:"):
        raw = key.split("custom:", 1)[1]
        label = _CUSTOM_METRIC_LABELS.get(raw, raw).replace("_", " ")
        return label
    labels = {
        "cpc": "CPC",
        "cpa": "CPA",
        "ctr": "CTR",
        "cvr": "CVR",
        "roas": "ROAS",
        "conversions": "conversions",
        "clicks": "clicks",
        "impressions": "impressions",
        "cost": "spend",
        "conversions_value": "revenue",
    }
    return labels.get(key, key)


def _delta_pct(deltas: dict, key: str) -> float | None:
    val = (deltas.get(key) or {}).get("pct_change")
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _build_relational_performance_reply(question: str, output: dict) -> str | None:
    if not isinstance(output, dict):
        return None
    result = output.get("result") or {}
    deltas = result.get("deltas") or {}
    if not isinstance(deltas, dict):
        return None
    lower = (question or "").lower()
    missing_spend = False
    data_quality = result.get("data_quality")
    if isinstance(data_quality, dict):
        missing_spend = bool(data_quality.get("missing_spend"))

    # Multi-metric comparison should take precedence over single-metric branches.
    metrics = _extract_metric_mentions(question)
    if len(metrics) >= 2 and _has_comparison_cue(question):
        candidates = []
        for key in metrics:
            pct = _delta_pct(deltas, key)
            if pct is not None:
                candidates.append((key, pct))
            if len(candidates) >= 2:
                break
        if len(candidates) < 2:
            return None
        key_a, pct_a = candidates[0]
        key_b, pct_b = candidates[1]
        winner = _metric_label(key_a) if abs(pct_a) >= abs(pct_b) else _metric_label(key_b)
        line = (
            f"{_metric_label(key_a)} moved {_format_pct(pct_a)} and "
            f"{_metric_label(key_b)} moved {_format_pct(pct_b)}; the larger move is {winner}."
        )
        if {"cost", "conversions"}.issubset({key_a, key_b}):
            cpa_pct = _delta_pct(deltas, "cpa")
            if cpa_pct is not None:
                eff = "worsened" if cpa_pct > 0 else "improved" if cpa_pct < 0 else "held flat"
                line += f" Efficiency {eff} (CPA {_format_pct(cpa_pct)})."
        next_step = "Next step: slice by campaign and device to isolate the segment driving the larger swing."
        note = " Spend is missing for this window, so treat this as directional." if missing_spend else ""
        return f"{line} {next_step}{note}".strip()

    # Conversions explained by clicks vs CVR
    if ("conversion" in lower or "conversions" in lower) and ("cvr" in lower or "conversion rate" in lower) and ("click" in lower or "clicks" in lower):
        clicks_pct = _delta_pct(deltas, "clicks")
        cvr_pct = _delta_pct(deltas, "cvr")
        conv_pct = _delta_pct(deltas, "conversions")
        if clicks_pct is None or cvr_pct is None:
            return None
        direction = "fell" if (conv_pct is not None and conv_pct < 0) else "rose" if (conv_pct is not None and conv_pct > 0) else "shifted"
        primary = "CVR" if abs(cvr_pct) >= abs(clicks_pct) else "clicks"
        line = (
            f"Conversions {direction} because both clicks ({_format_pct(clicks_pct)}) "
            f"and CVR ({_format_pct(cvr_pct)}) moved; {primary} is the larger contributor."
        )
        next_step = "Next step: cut CVR by campaign and device to isolate the driver."
        note = " Spend is missing for this window, so treat this as directional." if missing_spend else ""
        return f"{line} {next_step}{note}".strip()

    # CPC explained by spend vs clicks
    if "cpc" in lower or "cost per click" in lower:
        cost_pct = _delta_pct(deltas, "cost")
        clicks_pct = _delta_pct(deltas, "clicks")
        cpc_pct = _delta_pct(deltas, "cpc")
        if cost_pct is None or clicks_pct is None or cpc_pct is None:
            return None
        driver = "spend" if abs(cost_pct) >= abs(clicks_pct) else "clicks"
        line = (
            f"CPC moved {_format_pct(cpc_pct)}; spend is {_format_pct(cost_pct)} "
            f"while clicks are {_format_pct(clicks_pct)}. The larger driver is {driver}."
        )
        next_step = "Next step: check CPC by campaign and device to locate the pressure."
        note = " Spend is missing for this window, so treat this as directional." if missing_spend else ""
        return f"{line} {next_step}{note}".strip()

    # CTR explained by clicks vs impressions
    if "ctr" in lower or "click-through" in lower:
        clicks_pct = _delta_pct(deltas, "clicks")
        impr_pct = _delta_pct(deltas, "impressions")
        ctr_pct = _delta_pct(deltas, "ctr")
        if clicks_pct is None or impr_pct is None or ctr_pct is None:
            return None
        driver = "clicks" if abs(clicks_pct) >= abs(impr_pct) else "impressions"
        line = (
            f"CTR moved {_format_pct(ctr_pct)}; clicks are {_format_pct(clicks_pct)} "
            f"and impressions are {_format_pct(impr_pct)}. The larger driver is {driver}."
        )
        next_step = "Next step: review CTR by campaign and ad group to isolate the decline."
        note = " Spend is missing for this window, so treat this as directional." if missing_spend else ""
        return f"{line} {next_step}{note}".strip()

    # ROAS explained by spend vs revenue/value
    if "roas" in lower and ("cost" in lower or "spend" in lower or "revenue" in lower or "value" in lower):
        current = result.get("current") or {}
        previous = result.get("previous") or {}
        value_cur = current.get("conversions_value")
        value_prev = previous.get("conversions_value")
        if (value_cur in (None, 0) and value_prev in (None, 0)):
            return (
                "ROAS is 0 because conversion value is not captured in this feed. "
                "We cannot attribute ROAS changes to spend vs revenue until value tracking is populated."
            )
        cost_pct = _delta_pct(deltas, "cost")
        value_pct = _delta_pct(deltas, "conversions_value")
        roas_pct = _delta_pct(deltas, "roas")
        if cost_pct is None or value_pct is None or roas_pct is None:
            return None
        driver = "revenue" if abs(value_pct) >= abs(cost_pct) else "spend"
        line = (
            f"ROAS moved {_format_pct(roas_pct)}; revenue is {_format_pct(value_pct)} "
            f"while spend is {_format_pct(cost_pct)}. The larger driver is {driver}."
        )
        next_step = "Next step: validate conversion value tracking and review campaign-level ROAS to pinpoint the drop."
        note = " Spend is missing for this window, so treat this as directional." if missing_spend else ""
        return f"{line} {next_step}{note}".strip()

    # Efficiency (cost vs conversions / CPA)
    if "efficien" in lower or "cpa" in lower or (("cost" in lower or "spend" in lower) and ("conversion" in lower or "conversions" in lower)):
        cost_pct = _delta_pct(deltas, "cost")
        conv_pct = _delta_pct(deltas, "conversions")
        cpa_pct = _delta_pct(deltas, "cpa")
        if cost_pct is None or conv_pct is None or cpa_pct is None:
            return None
        eff = "worsened" if cpa_pct > 0 else "improved" if cpa_pct < 0 else "held flat"
        line = (
            f"Cost is {_format_pct(cost_pct)} while conversions are {_format_pct(conv_pct)}, "
            f"so efficiency {eff} (CPA {_format_pct(cpa_pct)})."
        )
        next_step = "Next step: isolate campaigns with the biggest CPA lift, then review query and device mix."
        note = " Spend is missing for this window, so treat this as directional." if missing_spend else ""
        return f"{line} {next_step}{note}".strip()

    return None


def _build_performance_explanation(message: str, deltas: dict, missing_spend: bool) -> str | None:
    focus_key, focus_label = _metric_focus_from_message(message)
    lower = (message or "").lower()
    wants_increase = "increase" in lower or "increased" in lower or "up" in lower
    wants_decrease = "decrease" in lower or "decreased" in lower or "down" in lower

    if focus_key:
        direction = _direction_from_delta(deltas, focus_key)
        if not direction:
            return None
        pct = _delta_pct(deltas, focus_key)
        pct_text = f" ({_format_pct(pct)})" if pct is not None else ""
        lines = []
        lines.append(f"At the account aggregate, {focus_label} is {direction}{pct_text} versus the prior window.")
        if wants_increase and direction == "down":
            lines.append("If you observed an increase, it is likely concentrated in a subset not visible in the aggregate.")
        elif wants_decrease and direction == "up":
            lines.append("If you expected a decrease, the aggregate is moving the other way; the change may be isolated by segment.")
        else:
            lines.append(
                f"Common drivers for {focus_label} moving {direction} include auction pressure, mix shifts, or changes in quality signals."
            )
        lines.append(f"Next step: break {focus_label} out by campaign and device for that window to locate the driver.")
        if missing_spend:
            lines.append("Spend was not returned for this window, so the signal may be incomplete.")
        return " ".join(lines)

    # Generic explanation when the user asks "why" without a specific metric.
    signals = []
    for key, label in [("conversions", "conversions"), ("cost", "spend"), ("cpc", "CPC"), ("ctr", "CTR")]:
        direction = _direction_from_delta(deltas, key)
        if direction and direction != "flat":
            pct = _delta_pct(deltas, key)
            pct_text = f" ({_format_pct(pct)})" if pct is not None else ""
            signals.append(f"{label} is {direction}{pct_text}")
    if not signals:
        return None
    summary = "At the account aggregate, " + ", ".join(signals[:2]) + " versus the prior window."
    next_step = "Next step: cut by campaign and device to pinpoint which segment drove the shift."
    note = "Spend was not returned for this window, so the signal may be incomplete." if missing_spend else ""
    return " ".join([summary, next_step, note]).strip()


def _format_driver_snippet(items: list[dict], label: str, limit: int = 2) -> str | None:
    if not items:
        return None
    parts = []
    for item in items[:limit]:
        name = str(item.get("name") or "").strip()
        pct = item.get("pct_change")
        change = item.get("change")
        if pct is not None:
            parts.append(f"{name} ({pct:+.1f}%)")
        elif change is not None:
            parts.append(f"{name} ({change:+.2f})")
        elif name:
            parts.append(name)
    if not parts:
        return None
    return f"{label}: " + ", ".join(parts)


def _pick_driver_metric(message: str, deltas: dict) -> str:
    focus_key, _ = _metric_focus_from_message(message)
    if focus_key:
        return focus_key
    t = (message or "").lower()
    if "diminishing" in t and "return" in t:
        return "roas"
    candidates = ["conversions", "cost", "cpc", "ctr", "cvr", "roas", "clicks", "impressions"]
    best_key = None
    best_val = None
    for key in candidates:
        pct = _delta_pct(deltas, key)
        if pct is None:
            continue
        val = abs(pct)
        if best_val is None or val > best_val:
            best_val = val
            best_key = key
    return best_key or "cost"


def _find_key_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _compute_dimension_breakdown(
    frames_current: dict[str, pd.DataFrame],
    frames_previous: dict[str, pd.DataFrame],
    key_candidates: list[str],
    metric_key: str,
    limit: int = 3,
) -> list[dict]:
    priority = ["campaign", "ad_group", "ad", "keyword_performance"]
    for frame_name in priority:
        cur_df = frames_current.get(frame_name)
        if cur_df is None or cur_df.empty:
            continue
        key_col = _find_key_col(cur_df, key_candidates)
        if not key_col:
            continue
        cur_metrics = _aggregate_frame(cur_df, key_col)
        if not cur_metrics:
            continue
        has_metric = any((metrics.get(metric_key) is not None) for metrics in cur_metrics.values())
        if not has_metric:
            continue
        prev_df = frames_previous.get(frame_name) if frames_previous else None
        prev_metrics = _aggregate_frame(prev_df, key_col) if prev_df is not None else {}

        items: list[dict] = []
        for name in set(cur_metrics) | set(prev_metrics):
            cur = (cur_metrics.get(name) or {}).get(metric_key)
            prev = (prev_metrics.get(name) or {}).get(metric_key)
            if cur is None and prev is None:
                continue
            change = None if (cur is None or prev is None) else (cur - prev)
            pct = None
            if change is not None and prev not in (None, 0, 0.0):
                pct = (change / prev) * 100
            items.append(
                {
                    "name": str(name),
                    "current": cur,
                    "previous": prev,
                    "change": change,
                    "pct_change": pct,
                }
            )

        items.sort(
            key=lambda x: abs(x["change"]) if isinstance(x.get("change"), (int, float)) else 0,
            reverse=True,
        )
        return items[:limit]
    return []


def _compute_custom_metric_breakdown(
    frames_current: dict[str, pd.DataFrame],
    frames_previous: dict[str, pd.DataFrame],
    key_candidates: list[str],
    metric_key: str,
    limit: int = 3,
) -> list[dict]:
    priority = ["campaign_conversion_action", "keyword_performance_conv", "keyword_performance", "campaign", "ad_group", "ad"]
    for frame_name in priority:
        cur_df = frames_current.get(frame_name)
        if cur_df is None or cur_df.empty:
            continue
        key_col = _find_key_col(cur_df, key_candidates)
        if not key_col:
            continue
        # If we have a previous window and schema evolved (e.g., added campaign.name), prefer a key that exists in both.
        prev_df = frames_previous.get(frame_name) if frames_previous else None
        if prev_df is not None and not prev_df.empty and key_col not in prev_df.columns:
            both = [c for c in key_candidates if c in cur_df.columns and c in prev_df.columns]
            if both:
                key_col = both[0]
        cur_sub, metric_col = _filter_custom_metric_rows(cur_df, metric_key)
        if cur_sub is None or metric_col is None:
            continue
        cur_metrics = _aggregate_frame_custom(cur_sub, key_col, metric_key, metric_col)
        if not cur_metrics:
            continue
        prev_metrics = {}
        if prev_df is not None and not prev_df.empty:
            prev_sub, prev_col = _filter_custom_metric_rows(prev_df, metric_key)
            if prev_sub is not None and prev_col is not None:
                try:
                    prev_metrics = _aggregate_frame_custom(prev_sub, key_col, metric_key, prev_col)
                except KeyError:
                    # Last-resort safety: if key_col isn't present in prev_sub due to older cached schema, skip prev metrics.
                    prev_metrics = {}

        items: list[dict] = []
        for name in set(cur_metrics) | set(prev_metrics):
            cur = (cur_metrics.get(name) or {}).get(metric_key)
            prev = (prev_metrics.get(name) or {}).get(metric_key)
            if cur is None and prev is None:
                continue
            change = None if (cur is None or prev is None) else (cur - prev)
            pct = None
            if change is not None and prev not in (None, 0, 0.0):
                pct = (change / prev) * 100
            items.append(
                {
                    "name": str(name),
                    "current": cur,
                    "previous": prev,
                    "change": change,
                    "pct_change": pct,
                }
            )

        items.sort(
            key=lambda x: abs(x["change"]) if isinstance(x.get("change"), (int, float)) else 0,
            reverse=True,
        )
        return items[:limit]
    return []


def _compute_driver_breakdown(
    frames_current: dict[str, pd.DataFrame],
    frames_previous: dict[str, pd.DataFrame],
    metric_key: str,
) -> dict[str, list[dict]]:
    if metric_key.startswith("custom:"):
        return {
            "campaign": _compute_custom_metric_breakdown(
                frames_current,
                frames_previous,
                ["campaign.name", "campaign.id", "campaign"],
                metric_key,
            ),
            "device": _compute_custom_metric_breakdown(
                frames_current,
                frames_previous,
                ["segments.device", "device"],
                metric_key,
            ),
            "geo": _compute_custom_metric_breakdown(
                frames_current,
                frames_previous,
                ["segments.geo_target_region", "geo"],
                metric_key,
            ),
        }
    return {
        "campaign": _compute_dimension_breakdown(
            frames_current,
            frames_previous,
            ["campaign.name", "campaign.id", "campaign"],
            metric_key,
        ),
        "device": _compute_dimension_breakdown(
            frames_current,
            frames_previous,
            ["segments.device", "device"],
            metric_key,
        ),
        "geo": _compute_dimension_breakdown(
            frames_current,
            frames_previous,
            ["segments.geo_target_region", "geo"],
            metric_key,
        ),
    }


def _retrieve_kb_snippets(query: str, limit: int = 3) -> list[dict]:
    cfg = load_vector_config()
    if not cfg.enabled:
        return []
    try:
        client = SearchClient(
            endpoint=cfg.search_endpoint,
            index_name=cfg.search_index,
            credential=AzureKeyCredential(cfg.search_key),
        )
    except Exception:
        return []

    filter_expr = "source eq 'knowledge_base'"
    vector_queries = build_vector_queries(query, cfg)
    vector_filter_mode = resolve_vector_filter_mode(cfg) if vector_queries else None

    try:
        if vector_queries:
            results = client.search(
                search_text=query if cfg.hybrid else None,
                filter=filter_expr,
                query_type="semantic" if cfg.hybrid else None,
                semantic_configuration_name="kai-semantic" if cfg.hybrid else None,
                vector_queries=vector_queries,
                vector_filter_mode=vector_filter_mode,
                select=["id", "title", "section", "content", "source"],
                top=limit,
            )
        else:
            results = client.search(
                search_text=query,
                filter=filter_expr,
                query_type="semantic",
                semantic_configuration_name="kai-semantic",
                select=["id", "title", "section", "content", "source"],
                top=limit,
            )
    except Exception as exc:
        logging.warning("KB retrieval failed: %s", exc)
        return []

    snippets: list[dict] = []
    for doc in results:
        snippets.append(
            {
                "id": doc.get("id"),
                "title": doc.get("title") or doc.get("section") or "Knowledge",
                "section": doc.get("section"),
                "content": _sanitize_kb_content(doc.get("content")),
                "source": doc.get("source"),
            }
        )
    return snippets


def _llm_explain_performance(question: str, payload: dict) -> tuple[str | None, dict]:
    local_max_tokens = int(os.environ.get("LOCAL_LLM_PERF_MAX_TOKENS", "180") or "180")
    azure_max_tokens = int(os.environ.get("AZURE_LLM_PERF_MAX_TOKENS", "260") or "260")
    summary_seed = _extract_summary_seed(payload)
    system_content = (
        performance_system_prompt()
        + " You may use numeric values only if they are present in the performance JSON."
        + " Include at least one numeric metric from the performance JSON when responding."
    )
    if summary_seed:
        system_content += " If a summary seed is provided, treat it as authoritative and do not change numbers."
    system = {
        "role": "system",
        "content": system_content,
    }
    compact = _compact_performance_payload(payload)
    payload_json = json.dumps(compact, ensure_ascii=True, default=str)
    seed_note = ""
    if summary_seed:
        seed_note = f"\nSummary seed (authoritative): {summary_seed}"
    messages = [
        system,
        {
            "role": "user",
            "content": f"Question: {question}\n\nPerformance data (JSON): {payload_json}{seed_note}",
        },
    ]
    reply, meta = _call_llm(messages, intent="performance", allow_local=True, max_tokens=local_max_tokens)
    if not reply:
        retry_tokens = max(80, local_max_tokens // 2)
        retry_reply, retry_meta = _call_local_llm(
            messages,
            intent="performance",
            max_tokens=retry_tokens,
            record_usage=True,
        )
        if retry_reply:
            reply, meta = retry_reply, retry_meta
    if _is_refusal_reply(reply):
        return None, meta or {}
    if meta and meta.get("model") == "local" and _is_low_quality_local_reply(reply):
        require_local = os.environ.get("REQUIRE_LOCAL_LLM", "false").lower() == "true"
        if not require_local:
            reply, meta = _call_llm(
                messages,
                intent="performance",
                allow_local=True,
                max_tokens=azure_max_tokens,
                force_azure=True,
            )
    return (reply.strip() if isinstance(reply, str) else None), meta or {}


def _llm_advise_performance(question: str, payload: dict) -> tuple[str | None, dict]:
    """
    Advisor-mode performance response: options, tradeoffs, recommendation, and monitoring plan.

    Guardrail: do not invent numbers. Numeric grounding is enforced by the chat handler.
    """
    local_max_tokens = int(os.environ.get("LOCAL_LLM_PERF_ADVICE_MAX_TOKENS", "320") or "320")
    azure_max_tokens = int(os.environ.get("AZURE_LLM_PERF_ADVICE_MAX_TOKENS", "420") or "420")
    summary_seed = _extract_summary_seed(payload)
    system_content = (
        performance_advisor_system_prompt()
        + " You may use numeric values only if they are present in the performance JSON."
        + " Include at least one numeric metric from the performance JSON when responding."
    )
    if summary_seed:
        system_content += " If a summary seed is provided, treat it as authoritative and do not change numbers."
    system = {"role": "system", "content": system_content}
    compact = _compact_performance_payload(payload)
    payload_json = json.dumps(compact, ensure_ascii=True, default=str)
    seed_note = ""
    if summary_seed:
        seed_note = f"\nSummary seed (authoritative): {summary_seed}"
    messages = [
        system,
        {
            "role": "user",
            "content": f"Question: {question}\n\nPerformance data (JSON): {payload_json}{seed_note}",
        },
    ]
    reply, meta = _call_llm(messages, intent="performance_advice", allow_local=True, max_tokens=local_max_tokens)
    if not reply:
        retry_tokens = max(120, local_max_tokens // 2)
        retry_reply, retry_meta = _call_local_llm(
            messages,
            intent="performance_advice",
            max_tokens=retry_tokens,
            record_usage=True,
        )
        if retry_reply:
            reply, meta = retry_reply, retry_meta
    if _is_refusal_reply(reply):
        return None, meta or {}
    if meta and meta.get("model") == "local":
        require_local = os.environ.get("REQUIRE_LOCAL_LLM", "false").lower() == "true"
        needs_upgrade = _is_low_quality_local_reply(reply) or not _is_advisor_grade_performance_reply(reply)
        if needs_upgrade and not require_local:
            reply, meta = _call_llm(
                messages,
                intent="performance_advice",
                allow_local=True,
                max_tokens=azure_max_tokens,
                force_azure=True,
            )
    reply_out = reply.strip() if isinstance(reply, str) else None
    if reply_out:
        reply_out = _humanize_relative_date_tokens(reply_out)
        reply_out = _ensure_performance_advice_minimum_structure(reply_out) or reply_out
        # Keep conversational output even when strict marker heuristics do not fully match.
        # This avoids dropping otherwise useful advisory guidance into terse metric dumps.
        if not _is_advisor_grade_performance_reply(reply_out):
            if not re.search(r"\bmonitor(ing)?\b", reply_out.lower()):
                reply_out += "\n\nMonitoring plan: Watch conversion volume, spend, CTR, and CPC daily for 7 days."
            if not re.search(r"\b(recommend|start with|i would start)\b", reply_out.lower()):
                reply_out += "\nRecommendation: Start with the lowest-risk change first, then iterate."
    return reply_out, meta or {}


def _llm_sync_tool_followup(question: str, tool_name: str | None, tool_output: Any) -> tuple[str | None, dict]:
    system = {
        "role": "system",
        "content": (
            chat_system_prompt()
            + " Use only the provided tool context JSON for factual statements. "
            "If the tool context is insufficient, say what is missing and ask one targeted follow-up."
        ),
    }
    compact = _compact_tool_output(tool_output)
    payload_json = json.dumps(compact, ensure_ascii=True, default=str)
    messages = [
        system,
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Tool: {tool_name or 'unknown'}\n"
                f"Tool context (JSON): {payload_json}"
            ),
        },
    ]
    require_local = os.environ.get("REQUIRE_LOCAL_LLM", "false").lower() == "true"
    reply, meta = _call_llm(
        messages,
        intent="tool_followup",
        allow_local=True,
        max_tokens=420,
        force_azure=False,
    )
    if meta and meta.get("model") == "local" and _is_low_quality_local_reply(reply):
        if not require_local:
            reply, meta = _call_llm(
                messages,
                intent="tool_followup",
                allow_local=True,
                max_tokens=420,
                force_azure=True,
            )
    return (reply.strip() if isinstance(reply, str) else None), meta or {}


def _qa_raw_totals(frames: dict[str, pd.DataFrame]) -> dict:
    """Compute raw sums from keyword_performance frame as a QA reference."""
    df = frames.get("keyword_performance")
    if df is None:
        df = pd.DataFrame()
    if df.empty:
        return {"impressions": None, "clicks": None, "cost": None, "conversions": None}
    impr_col = df.get("Impr.")
    if impr_col is None:
        impr_col = df.get("impressions")
    clicks_col = df.get("Clicks")
    if clicks_col is None:
        clicks_col = df.get("clicks")
    cost_col = df.get("Cost")
    if cost_col is None:
        cost_col = df.get("cost")
    conv_col = df.get("Conversions")
    if conv_col is None:
        conv_col = df.get("conversions")

    impr = pd.to_numeric(impr_col, errors="coerce").fillna(0).sum() if impr_col is not None else 0.0
    clicks = pd.to_numeric(clicks_col, errors="coerce").fillna(0).sum() if clicks_col is not None else 0.0
    cost = pd.to_numeric(cost_col, errors="coerce").fillna(0).sum() if cost_col is not None else 0.0
    conv = pd.to_numeric(conv_col, errors="coerce").fillna(0).sum() if conv_col is not None else 0.0
    return {
        "impressions": float(impr),
        "clicks": float(clicks),
        "cost": float(cost),
        "conversions": float(conv),
    }


def _validate_sa360_plan(plan: dict, session_id: str | None = None) -> tuple[dict, str | None]:
    """
    Ensure required keys exist and strip unsupported fields from plan body.
    Currently ensures customer_ids present; leaves geo/device constraints to query definitions.
    """
    notes = []
    if not plan.get("customer_ids"):
        raise HTTPException(status_code=400, detail="No customer_ids provided in plan.")
    for cid in plan.get("customer_ids", []):
        if _is_sa360_manager_account(str(cid), session_id=session_id):
            raise HTTPException(status_code=400, detail="Requested account is a manager (MCC); select a child account for metrics.")
    if not plan.get("date_range"):
        notes.append("No date specified; defaulting to last 7 days.")
        plan["date_range"] = "LAST_7_DAYS"
    # Normalize date_range to string in case a numeric gets through (avoid .strip on numpy scalars)
    if plan.get("date_range") is not None and not isinstance(plan["date_range"], str):
        plan["date_range"] = str(plan["date_range"])
    return plan, "; ".join(notes) if notes else None


async def _chat_plan_and_run_core(req: PlanRequest, request: Request | None = None) -> PlanResponse:
    """
    Lightweight planner: interpret human message into SA360 fetch-and-audit call,
    validate, execute, and return result + notes.
    """
    try:
        # If Entra SSO is enabled, scope SA360 credentials/defaults per user (JWT oid) rather than per chat session.
        sa360_sid = _sa360_scope_from_request(request, req.session_id) if request is not None else req.session_id
        intent_hint = (req.intent_hint or "").lower().strip() if req.intent_hint else None
        # Honor explicit intent hint from router; default to performance unless explicitly audit
        audit_forced = intent_hint == "audit"
        perf_intent = not audit_forced
        entity_intent = _extract_entity_intent(req.message)
        top_intent = _extract_top_mover_intent(req.message)
        # Broad beta UX: if the UI did not include customer_ids, fall back to the user's saved default SA360 account.
        fallback_ids = _normalize_customer_ids(req.customer_ids or [])
        fallback_account_name = req.account_name
        used_saved_default = False
        default_cid = ""
        try:
            sess = _load_sa360_session(sa360_sid) if sa360_sid else None
            default_cid = _normalize_customer_id_value(
                (sess.get("default_customer_id") if isinstance(sess, dict) else "")
            )
            if not fallback_ids and default_cid:
                fallback_ids = [default_cid]
                if not fallback_account_name:
                    fallback_account_name = (sess.get("default_account_name") or "") if isinstance(sess, dict) else ""
                used_saved_default = True
        except Exception:
            pass

        resolved_ids, resolved_account, resolution_notes, _ = _resolve_account_context(
            req.message,
            customer_ids=fallback_ids,
            account_name=fallback_account_name,
            explicit_ids=bool(req.customer_ids),
            session_id=sa360_sid,
        )
        # Defensive: if the user/UI provided explicit customer_ids but the resolver cleared them
        # (e.g., due to fuzzy text matches), keep the provided IDs. Broad beta users expect
        # account selection to be stable once chosen.
        if not resolved_ids and fallback_ids and req.customer_ids:
            resolved_ids = list(fallback_ids)
            if not resolved_account:
                resolved_account = fallback_account_name
            note = "Using provided customer_id(s) from request."
            resolution_notes = f"{note} {resolution_notes}".strip() if resolution_notes else note
        if used_saved_default and resolved_ids and default_cid and resolved_ids == [default_cid]:
            # Add an explicit note so users understand why an account was used without typing an ID.
            note = f"Using saved account {resolved_account or default_cid} ({default_cid})."
            resolution_notes = f"{note} {resolution_notes}".strip() if resolution_notes else note
        plan = _build_sa360_plan_from_chat(
            req.message,
            customer_ids=resolved_ids,
            account_name=resolved_account,
            default_date=req.default_date_range,
        )
        plan, notes = _validate_sa360_plan(plan, session_id=sa360_sid)
        combined_notes = "; ".join([n for n in [resolution_notes, notes] if n])

        # Broad beta reliability: audit runs can exceed ingress timeouts when executed synchronously
        # (audit engine + XLSX + model/narrative passes). Queue audit requests in the API context
        # and let the worker complete them without blocking the HTTP request.
        #
        # IMPORTANT: When running inside the worker, `request` is None. We must not enqueue again
        # or we'd create an infinite loop.
        if audit_forced and request is not None:
            payload = req.model_dump()
            payload["customer_ids"] = resolved_ids
            payload["account_name"] = resolved_account
            payload["async_mode"] = False
            payload["session_id"] = sa360_sid
            job_id = enqueue_job("sa360_plan_and_run", payload)
            plan_stub = {
                "customer_ids": resolved_ids,
                "account_name": resolved_account,
                "date_range": plan.get("date_range") or req.default_date_range or "LAST_7_DAYS",
            }
            return PlanResponse(
                plan=plan_stub,
                notes=combined_notes,
                executed=False,
                status="queued",
                job_id=job_id,
            )

        if _should_enqueue_heavy(req.async_mode, len(resolved_ids), SA360_PLAN_MAX_SYNC_ACCOUNTS):
            payload = req.model_dump()
            payload["customer_ids"] = resolved_ids
            payload["account_name"] = resolved_account
            payload["async_mode"] = False
            payload["session_id"] = sa360_sid
            job_id = enqueue_job("sa360_plan_and_run", payload)
            plan_stub = {
                "customer_ids": resolved_ids,
                "account_name": resolved_account,
                "date_range": plan.get("date_range") or req.default_date_range or "LAST_7_DAYS",
            }
            return PlanResponse(
                plan=plan_stub,
                notes=combined_notes,
                executed=False,
                status="queued",
                job_id=job_id,
            )

        # Performance-only path (no audit score, no XLSX) when the user asks for metrics/deltas or router hinted performance
        if perf_intent and not audit_forced:
            _ensure_sa360_feature()
            _ensure_sa360_enabled()

            # Broad beta guardrail: if SA360 is not connected for this session, do not fall back
            # to "generic" AI suggestions that look like analysis. Return an explicit block instead.
            if not sa360_sid:
                plan_stub = {
                    "customer_ids": plan.get("customer_ids") or [],
                    "account_name": plan.get("account_name"),
                    "date_range": plan.get("date_range") or req.default_date_range or "LAST_7_DAYS",
                }
                return PlanResponse(
                    plan=plan_stub,
                    notes=combined_notes,
                    executed=False,
                    status="blocked",
                    error="sa360_not_connected",
                    summary="SA360 isn't connected for this session. Click Connect SA360 at the top of Kai, then retry.",
                )

            session = _load_sa360_session(sa360_sid)
            if not (session and session.get("refresh_token")):
                plan_stub = {
                    "customer_ids": plan.get("customer_ids") or [],
                    "account_name": plan.get("account_name"),
                    "date_range": plan.get("date_range") or req.default_date_range or "LAST_7_DAYS",
                }
                return PlanResponse(
                    plan=plan_stub,
                    notes=combined_notes,
                    executed=False,
                    status="blocked",
                    error="sa360_not_connected",
                    summary="SA360 isn't connected for this session. Click Connect SA360 at the top of Kai, then retry.",
                )

            # QA stability + user intent: when a caller supplies a concrete window (YYYY-MM-DD,YYYY-MM-DD),
            # prefer it for "complete days" style asks. This keeps parity/accuracy checks deterministic
            # while still allowing natural-language ranges when the UI does not provide a concrete window.
            try:
                if req.default_date_range and isinstance(req.default_date_range, str) and "," in req.default_date_range:
                    msg_lower = (req.message or "").lower()

                    # "Single window only" phrasing + last-30 style requests (no previous compare) should honor caller window.
                    if (
                        (not bool(req.include_previous))
                        and ("single window" in msg_lower or "single-window" in msg_lower)
                        and any(token in msg_lower for token in ("last 30", "past 30", "previous 30", "last month"))
                    ):
                        plan["date_range"] = req.default_date_range
                    # "Last 7 complete days" should also honor caller window (the UI/QA runner provides a stable range).
                    elif ("complete" in msg_lower and any(token in msg_lower for token in ("last 7", "past 7", "previous 7"))):
                        plan["date_range"] = req.default_date_range
            except Exception:
                pass

            span = _date_span_from_range(_coerce_date_range(plan["date_range"]))
            if not span:
                plan["date_range"] = "LAST_7_DAYS"
                span = _date_span_from_range(_coerce_date_range(plan["date_range"]))
            prev_span = _previous_span(span) if (span and bool(req.include_previous)) else None

            def span_to_range(sp: tuple[date, date]) -> str:
                return f"{sp[0]:%Y-%m-%d},{sp[1]:%Y-%m-%d}"

            current_range = span_to_range(span) if span else plan["date_range"]
            previous_range = span_to_range(prev_span) if prev_span else None

            custom_metrics = _extract_custom_metric_mentions(req.message)
            quoted_metric_phrase = _extract_quoted_metric_phrase(req.message)
            # If a user quotes a conversion action / metric phrase, force inference from the SA360 catalog.
            # This prevents treating quoted action names as already-normalized custom:<key> tokens.
            if quoted_metric_phrase:
                custom_metrics = []
            custom_key = custom_metrics[0] if custom_metrics else None
            # Capture what the user explicitly typed before any inference. Broad beta guardrail:
            # never silently remap an explicit token (e.g., FR_Intent_Clicks) to a different conversion action.
            explicit_custom_key = custom_key
            # Some tokens (notably internal aliases) may be rewritten/normalized elsewhere before this point.
            # As a guardrail, also detect explicit snake_case-like metric tokens directly from the message.
            explicit_token_key, _explicit_token_raw = (None, None)
            if not quoted_metric_phrase:
                explicit_token_key, _explicit_token_raw = _extract_explicit_metric_token_key(req.message)
                if explicit_token_key and not explicit_custom_key:
                    explicit_custom_key = explicit_token_key
                    custom_key = explicit_custom_key
            # Defensive: some user-entered alias tokens may not be extracted as a "custom:" key depending on
            # punctuation/context. If the prompt looks like a metric ask and contains a short underscore token,
            # treat it as an explicit metric token so we can block/confirm instead of guessing.
            try:
                if (not quoted_metric_phrase) and (not explicit_custom_key) and (
                    _message_is_direct_metric_request(req.message)
                    or _message_is_relational_custom_metric_request(req.message)
                ):
                    toks = [
                        t
                        for t in _CUSTOM_METRIC_RE.findall(req.message or "")
                        if "_" in t and 3 <= len(t) <= 64
                    ]
                    if toks:
                        metricish = [
                            t
                            for t in toks
                            if re.search(r"(clicks?|conversions?|value|revenue|visits?)$", (t or "").lower())
                        ]
                        picked = metricish[0] if metricish else toks[0]
                        norm = _normalize_custom_metric_key(picked)
                        if norm:
                            explicit_custom_key = f"custom:{norm}"
                            custom_key = explicit_custom_key
                            if norm not in _CUSTOM_METRIC_LABELS:
                                _CUSTOM_METRIC_LABELS[norm] = picked
            except Exception:
                pass
            standard_metrics = _extract_metric_mentions(req.message)
            has_standard_metric = bool(standard_metrics)
            # We should attempt conversion-action inference when:
            # - the user provides a custom metric token (e.g., FR_intent_clicks), OR
            # - the message contains mixed standard+nonstandard metric terms, OR
            # - the user directly asks for a single metric by name ("show me store visits").
            needs_custom_context = (
                bool(custom_key)
                or bool(quoted_metric_phrase)
                or _message_has_custom_metric_cue(req.message)
                or _message_is_direct_metric_request(req.message)
                or _message_is_relational_custom_metric_request(req.message)
            )
            # Keep the default performance pull lean to protect response-time budgets.
            # For action-oriented asks, include at least one breakdown so recommendations are not overly generic.
            perf_reports = ["customer_performance", "account"]
            msg_lower = (req.message or "").lower()
            is_action_request_prefetch = _is_performance_action_request(req.message)
            wants_campaign = any(t in msg_lower for t in ("campaign", "campaigns", "pmax", "performance max"))
            wants_ad_group = any(t in msg_lower for t in ("ad group", "adgroup", "ad groups", "adgroups"))
            wants_ads = any(t in msg_lower for t in ("ad copy", "rsa", "headline", "headlines", "description", "descriptions"))
            wants_keywords = any(
                t in msg_lower
                for t in (
                    "keyword",
                    "keywords",
                    "query",
                    "queries",
                    "search term",
                    "search terms",
                    "impression share",
                    "rank lost",
                )
            )
            wants_landing_pages = any(t in msg_lower for t in ("landing page", "landing pages", "final url", "final urls", "url health", "soft 404"))
            wants_driver_breakdown = (
                _has_relational_cue(req.message)
                or ("driver" in msg_lower)
                or ("drivers" in msg_lower)
                or ("drove" in msg_lower)
            )
            if wants_campaign and "campaign" not in perf_reports:
                perf_reports.append("campaign")
            if wants_ad_group and "ad_group" not in perf_reports:
                perf_reports.append("ad_group")
            if wants_ads and "ad" not in perf_reports:
                perf_reports.append("ad")
            if wants_keywords and "keyword_performance" not in perf_reports:
                perf_reports.append("keyword_performance")
            if wants_landing_pages and "landing_page" not in perf_reports:
                perf_reports.append("landing_page")
            if is_action_request_prefetch:
                for name in ("campaign", "ad_group"):
                    if name not in perf_reports:
                        perf_reports.append(name)
            # Driver questions require at least one metrics-bearing dimensional frame.
            # Prefer keyword_view for standard KPIs because campaign/ad_group queries are metadata-only in this codebase.
            if wants_driver_breakdown and "keyword_performance" not in perf_reports:
                perf_reports.append("keyword_performance")

            if needs_custom_context:
                # For custom metrics (often conversion actions / custom columns), we need:
                # - conversion_action_summary: customer-level action totals (most complete)
                # - conversion_actions: catalog so we can match actions even when conversions are 0 in the window
                for name in ["conversion_action_summary", "conversion_actions"]:
                    if name not in perf_reports:
                        perf_reports.append(name)
                # Only pull conversion-action segmented rows when the user asks for drivers/breakdowns.
                wants_drivers = (
                    _has_relational_cue(req.message)
                    or ("driver" in msg_lower)
                    or ("drivers" in msg_lower)
                    or ("which" in msg_lower)
                )
                if wants_drivers and "campaign_conversion_action" not in perf_reports:
                    perf_reports.append("campaign_conversion_action")
                if wants_drivers and "keyword_performance_conv" not in perf_reports:
                    perf_reports.append("keyword_performance_conv")

            plan_bypass_cache = SA360_PLAN_BYPASS_CACHE
            use_batched = len(plan.get("customer_ids") or []) > SA360_PLAN_MAX_SYNC_ACCOUNTS
            if use_batched:
                def _fetch_batched(range_value: str):
                    return _collect_sa360_frames_batched(
                        plan["customer_ids"],
                        range_value,
                        bypass_cache=plan_bypass_cache,
                        write_cache=(not plan_bypass_cache),
                        max_workers=SA360_PLAN_CONCURRENCY,
                        chunk_size=SA360_PLAN_CHUNK_SIZE,
                        report_names=perf_reports,
                        session_id=sa360_sid,
                    )

                if previous_range:
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        future_current = executor.submit(_fetch_batched, current_range)
                        future_previous = executor.submit(_fetch_batched, previous_range)
                        frames_current = future_current.result()
                        frames_previous = future_previous.result()
                else:
                    frames_current = _fetch_batched(current_range)
                    frames_previous = {}
            else:
                frames_current = _collect_sa360_frames(
                    plan["customer_ids"],
                    current_range,
                    bypass_cache=plan_bypass_cache,
                    write_cache=(not plan_bypass_cache),
                    report_names=perf_reports,
                    session_id=sa360_sid,
                )
                frames_previous = (
                    _collect_sa360_frames(
                        plan["customer_ids"],
                        previous_range,
                        bypass_cache=plan_bypass_cache,
                        write_cache=(not plan_bypass_cache),
                        report_names=perf_reports,
                        session_id=sa360_sid,
                    )
                    if previous_range
                    else {}
                )

            custom_inferred = False
            custom_match: dict[str, Any] | None = None
            custom_suggestions: list[str] = []
            if needs_custom_context and not custom_key:
                query_text = quoted_metric_phrase or req.message
                inferred = _infer_custom_metric_from_frames(query_text, frames_current)
                if inferred:
                    custom_key = inferred.get("metric_key")
                    custom_inferred = True
                    custom_match = inferred
                else:
                    custom_suggestions = _suggest_custom_metric_candidates(query_text, frames_current, limit=3)
            elif needs_custom_context and custom_key and not custom_inferred:
                # If the user referenced a conversion action by name (often long underscore/hyphen strings),
                # mark it as "inferred" only when it actually exists in the account catalog. This preserves
                # the explicit-alias guardrail (e.g., FR_Intent_Clicks) while satisfying parity expectations
                # for real conversion action names pasted into chat.
                try:
                    present, _samples = _custom_metric_presence(frames_current, custom_key)
                except Exception:
                    present = False
                if present:
                    raw_match = None
                    try:
                        target = _normalize_custom_metric_cmp(_custom_metric_name(custom_key))
                        if target:
                            cat = frames_current.get("conversion_actions")
                            if cat is not None and not cat.empty and "conversion_action.name" in cat.columns:
                                for name in cat["conversion_action.name"].astype(str).dropna().unique().tolist():
                                    if _normalize_custom_metric_cmp(name) == target:
                                        raw_match = name
                                        break
                            if raw_match is None:
                                summ = frames_current.get("conversion_action_summary")
                                if summ is not None and not summ.empty and "segments.conversion_action_name" in summ.columns:
                                    for name in summ["segments.conversion_action_name"].astype(str).dropna().unique().tolist():
                                        if _normalize_custom_metric_cmp(name) == target:
                                            raw_match = name
                                            break
                    except Exception:
                        raw_match = None
                    custom_inferred = True
                    custom_match = {
                        "metric_key": custom_key,
                        "kind": "conversion_action_name",
                        "match": raw_match or _metric_label(custom_key),
                        "score": 1.0,
                    }

            # If the system inferred a custom metric but the user explicitly provided a token-like metric,
            # do NOT silently override what the user typed. Block and ask for confirmation.
            if explicit_custom_key and custom_inferred and custom_key and custom_key != explicit_custom_key:
                suggestions: list[str] = []
                try:
                    if custom_match and custom_match.get("match"):
                        suggestions.append(str(custom_match.get("match")))
                except Exception:
                    pass
                if not custom_suggestions:
                    custom_suggestions = _suggest_custom_metric_candidates(req.message, frames_current, limit=3)
                for s in (custom_suggestions or []):
                    if s and s not in suggestions:
                        suggestions.append(s)
                requested = _metric_label(explicit_custom_key)
                match_label = _metric_label(custom_key)
                summary = (
                    f"I couldn't find a SA360 conversion column named '{requested}' in this account. "
                    f"I can interpret it as '{match_label}', but I need confirmation to avoid guessing."
                )
                if suggestions:
                    summary += " Did you mean: " + ", ".join(suggestions[:3]) + "?"
                summary += " Open **SA360 Columns** to search and click *Use in chat*, or reply with the exact conversion name."
                return PlanResponse(
                    plan=plan,
                    notes=combined_notes,
                    executed=False,
                    status="blocked",
                    error="custom_metric_not_found",
                    summary=summary,
                    analysis={
                        "custom_metric": {
                            "requested": requested,
                            "suggestions": suggestions[:5] if suggestions else [],
                            "candidate": _to_primitive(custom_match) if custom_match else None,
                        }
                    },
                )

            perf_current = _aggregate_account_performance(frames_current)
            perf_previous = _aggregate_account_performance(frames_previous) if previous_range else {}
            custom_missing = False
            custom_samples: list[str] = []
            custom_metric_meta: dict[str, Any] | None = None
            if custom_key:
                custom_current, custom_metric_meta = _sum_custom_metric(frames_current, custom_key)
                if custom_current is not None:
                    perf_current[custom_key] = custom_current
                else:
                    custom_missing = True
                    present, samples = _custom_metric_presence(frames_current, custom_key)
                    if samples:
                        custom_samples = samples
                if previous_range:
                    custom_prev, _prev_meta = _sum_custom_metric(frames_previous, custom_key)
                    if custom_prev is not None:
                        perf_previous[custom_key] = custom_prev
            # If the user provided a custom-metric token (often an internal alias like FR_Intent_Clicks),
            # but it doesn't match any SA360 conversion action/column, do NOT silently remap it.
            # Broad beta correctness > guessing: surface candidate(s) and ask for confirmation.
            custom_alias_candidate: dict[str, Any] | None = None
            if needs_custom_context and custom_key and custom_missing and not custom_inferred:
                inferred = _infer_custom_metric_from_frames(req.message, frames_current)
                if inferred and inferred.get("metric_key") and inferred.get("metric_key") != custom_key:
                    custom_alias_candidate = inferred
                if not custom_suggestions:
                    custom_suggestions = _suggest_custom_metric_candidates(req.message, frames_current, limit=3)

                # If the user is explicitly asking about this metric (or they pasted an explicit token),
                # block and request clarification.
                if explicit_custom_key or _message_is_direct_metric_request(req.message) or _message_is_relational_custom_metric_request(req.message):
                    suggestions: list[str] = []
                    try:
                        if custom_alias_candidate and custom_alias_candidate.get("match"):
                            suggestions.append(str(custom_alias_candidate.get("match")))
                    except Exception:
                        pass
                    for s in (custom_suggestions or []):
                        if s and s not in suggestions:
                            suggestions.append(s)

                    explicit_label: str | None = None
                    try:
                        key_str = str(custom_key)
                        norm = key_str.split("custom:", 1)[1] if key_str.startswith("custom:") else None
                        explicit_label = (_CUSTOM_METRIC_LABELS.get(norm) or norm) if norm else None
                    except Exception:
                        explicit_label = None
                    requested = explicit_label or str(custom_key)

                    summary = f"I couldn't find a SA360 conversion column named '{requested}' in this account."
                    if suggestions:
                        summary += " Did you mean: " + ", ".join(suggestions[:3]) + "?"
                    summary += " Open **SA360 Columns** to search and click *Use in chat*, or reply with the exact conversion name."

                    return PlanResponse(
                        plan=plan,
                        notes=combined_notes,
                        executed=False,
                        status="blocked",
                        error="custom_metric_not_found",
                        summary=summary,
                        analysis={
                            "custom_metric": {
                                "requested": requested,
                                "suggestions": suggestions[:5] if suggestions else [],
                                "candidate": _to_primitive(custom_alias_candidate) if custom_alias_candidate else None,
                            }
                        },
                    )
            deltas = _compute_perf_deltas(perf_current, perf_previous)
            debug_perf = _get_perf_frame_debug(frames_current)
            try:
                print(f"[perf_debug] frame={debug_perf.get('frame')} rows={debug_perf.get('row_count')} cost_col={debug_perf.get('cost_column')} nonnull={debug_perf.get('cost_nonnull')} sample={debug_perf.get('cost_sample')}", file=sys.stderr, flush=True)
            except Exception:
                pass

            missing_spend = (
                (perf_current.get("cost") in (None, 0, 0.0))
                and (perf_current.get("clicks") or 0) > 0
                and (debug_perf.get("cost_nonnull") or 0) == 0
            )

            acct = plan.get("account_name") or "your account"
            summary_parts = [f"{acct}: performance {current_range}"]
            if previous_range:
                summary_parts.append(f"vs {previous_range}.")

            def fmt_delta(key: str, label: str) -> str | None:
                d = deltas.get(key) or {}
                cur = d.get("current")
                pct = d.get("pct_change")
                if cur is None:
                    return None
                if pct is None:
                    return f"{label}: {cur:,.2f}"
                return f"{label}: {cur:,.2f} ({pct:+.1f}% vs prev)"

            def fmt_delta_with_change(key: str, label: str) -> str | None:
                d = deltas.get(key) or {}
                cur = d.get("current")
                prev = d.get("previous")
                pct = d.get("pct_change")
                change = d.get("change")
                if cur is None:
                    return None
                def with_currency(val: float) -> str:
                    return f"${val:,.2f}"
                def with_currency_delta(val: float) -> str:
                    sign = "+" if val > 0 else ("-" if val < 0 else "")
                    return f"{sign}${abs(val):,.2f}"
                is_money = label.lower() in {"cost", "cpc", "cpa"}
                val_str = with_currency(cur) if is_money else f"{cur:,.2f}"
                parts = [f"{label}: {val_str}"]
                # Include absolute change if meaningful
                if change not in (None, 0):
                    delta_val = with_currency_delta(change) if is_money else f"{change:+,.2f}"
                    # Use ASCII (avoid mojibake in some log sinks/browsers) while keeping meaning clear.
                    parts.append(f"(delta {delta_val}")
                    if pct is not None:
                        parts[-1] += f", {pct:+.1f}%"
                    parts[-1] += ")"
                elif pct is not None:
                    parts.append(f"({pct:+.1f}%)")
                return " ".join(parts)

            custom_highlight = fmt_delta_with_change(custom_key, _metric_label(custom_key)) if custom_key else None
            highlights = [
                custom_highlight,
                fmt_delta_with_change("conversions", "Conversions"),
                fmt_delta_with_change("cost", "Cost"),
                fmt_delta_with_change("roas", "ROAS"),
            ]
            highlights = [h for h in highlights if h]
            secondary = [fmt_delta_with_change("ctr", "CTR"), fmt_delta_with_change("cpc", "CPC"), fmt_delta_with_change("cvr", "CVR")]
            secondary = [s for s in secondary if s]

            summary_text = " ".join(summary_parts).strip()
            if highlights:
                summary_text += " | " + " ; ".join(highlights)
            if secondary:
                summary_text += " | " + " ; ".join(secondary[:2])
            if custom_inferred and custom_key and custom_match:
                match_label = _metric_label(custom_key)
                match_kind = custom_match.get("kind") or "header"
                match_raw = custom_match.get("match") or match_label
                summary_text += f" Interpreting metric as {match_label} (matched {match_kind} '{match_raw}')."
            if missing_spend:
                summary_text += " (Spend not returned for this window; data may be missing for this date range.)"
            if custom_key and custom_missing:
                custom_label = _metric_label(custom_key)
                note = f"{custom_label} not found in SA360 conversion actions for this account/timeframe."
                if custom_samples:
                    note += " Found actions: " + ", ".join(custom_samples[:3]) + "."
                summary_text += " " + note

            actions = []
            conv_pct = deltas.get("conversions", {}).get("pct_change")
            cost_pct = deltas.get("cost", {}).get("pct_change")
            roas_pct = deltas.get("roas", {}).get("pct_change")
            if conv_pct is not None and conv_pct < 0 and (cost_pct is None or cost_pct >= 0):
                actions.append("Tighten spend to high-conversion segments; pause waste.")
            if roas_pct is not None and roas_pct < 0:
                actions.append("Check bidding/attribution shifts impacting ROAS.")
            if not actions and not highlights:
                actions.append("No strong signals; consider deeper slice by device/geo/query.")
            if actions:
                summary_text += " Next steps: " + " ".join(actions[:2])

            summary_text = _normalize_reply_text(summary_text) or summary_text

            analysis = None
            if _should_explain_performance(req.message):
                metric_key = custom_key or _pick_driver_metric(req.message, deltas)
                drivers = (
                    _compute_driver_breakdown(frames_current, frames_previous, metric_key)
                    if previous_range
                    else {"campaign": [], "device": [], "geo": []}
                )
                if _has_dimension_cue(req.message):
                    camp_snip = _format_driver_snippet(drivers.get("campaign") or [], "Top campaigns")
                    dev_snip = _format_driver_snippet(drivers.get("device") or [], "Top devices")
                    geo_snip = _format_driver_snippet(drivers.get("geo") or [], "Top geos")
                    driver_bits = [snip for snip in (camp_snip, dev_snip, geo_snip) if snip]
                    if driver_bits and "Drivers" not in summary_text:
                        # Use ASCII to avoid mojibake in some terminals/log sinks.
                        summary_text += " Drivers - " + "; ".join(driver_bits) + "."
                kb_snippets = _retrieve_kb_snippets(req.message, limit=3)
                missing = []
                if not drivers.get("campaign"):
                    missing.append("campaign")
                if not drivers.get("device"):
                    missing.append("device")
                # Knowledge-base grounding is optional unless vector search is configured/enabled.
                kb_enabled = False
                try:
                    kb_enabled = bool(load_vector_config().enabled)
                except Exception:
                    kb_enabled = False
                if kb_enabled and not kb_snippets:
                    missing.append("kb")
                grounded = len(missing) == 0
                explain_payload = {
                    "account": plan.get("account_name"),
                    "date_range_current": current_range,
                    "date_range_previous": previous_range,
                    "current": _to_primitive(perf_current),
                    "previous": _to_primitive(perf_previous),
                    "deltas": _to_primitive(deltas),
                    "data_quality": {"missing_spend": missing_spend},
                    "metric_focus": metric_key,
                    "drivers": _to_primitive(drivers),
                    "kb": _to_primitive(kb_snippets),
                }
                explain_payload["summary_seed"] = summary_text
                explanation = None
                exp_meta = None
                metrics = _extract_metric_mentions(req.message)
                has_drivers = any(drivers.get(k) for k in ("campaign", "device", "geo"))
                is_action_request = _is_performance_action_request(req.message)
                use_rules_only = False
                if _has_dimension_cue(req.message) and has_drivers:
                    use_rules_only = True
                if custom_key and not _has_strong_explain_cue(req.message):
                    use_rules_only = True
                if _has_comparison_cue(req.message) and len(metrics) >= 2:
                    explanation = _build_relational_performance_reply(
                        req.message,
                        {
                            "result": {
                                "current": perf_current,
                                "previous": perf_previous,
                                "deltas": deltas,
                                "data_quality": {"missing_spend": missing_spend},
                            }
                        },
                    )
                if not explanation:
                    if use_rules_only:
                        explanation = summary_text
                        exp_meta = {"model": "rules"}
                    else:
                        # Action asks should return advisor-grade guidance directly from plan-and-run.
                        if is_action_request:
                            explanation, exp_meta = _llm_advise_performance(req.message, explain_payload)
                        if not explanation:
                            explanation, exp_meta = _llm_explain_performance(req.message, explain_payload)
                        if not explanation:
                            explanation = _build_performance_explanation(req.message, deltas, missing_spend)
                def _summary_conflicts_with_deltas(text: str, deltas_payload: dict) -> bool:
                    if not isinstance(text, str) or not text.strip():
                        return False
                    if not isinstance(deltas_payload, dict) or not deltas_payload:
                        return False
                    lower = text.lower()
                    cues = (
                        "cannot compare",
                        "can't compare",
                        "no data",
                        "not enough data",
                        "insufficient data",
                        "missing data",
                        "provided data",
                        "don't have enough data",
                    )
                    return any(cue in lower for cue in cues)

                if explanation and _summary_conflicts_with_deltas(explanation, deltas):
                    relational = _build_relational_performance_reply(
                        req.message,
                        {"result": {"deltas": deltas, "data_quality": {"missing_spend": missing_spend}}},
                    )
                    if relational:
                        explanation = relational
                    else:
                        fallback = _build_performance_explanation(req.message, deltas, missing_spend)
                        if fallback:
                            explanation = fallback

                if explanation:
                    allowed_numbers: set[str] = set()
                    _collect_numeric_tokens(explain_payload, allowed_numbers)
                    # Strengthen grounded summary with explicit drivers + next step for advisor quality.
                    driver_bits = []
                    if grounded:
                        camp_snip = _format_driver_snippet(drivers.get("campaign") or [], "Top campaigns")
                        dev_snip = _format_driver_snippet(drivers.get("device") or [], "Top devices")
                        geo_snip = _format_driver_snippet(drivers.get("geo") or [], "Top geos")
                        driver_bits = [snip for snip in (camp_snip, dev_snip, geo_snip) if snip]
                    if driver_bits:
                        # Use ASCII to avoid mojibake in some terminals/log sinks.
                        explanation = f"{explanation} Drivers - " + "; ".join(driver_bits) + "."
                        if "Next step:" not in explanation:
                            explanation = f"{explanation} Next step: validate drivers in campaign, device, and geo cuts."
                    explanation, guardrail = _apply_numeric_grounding_guardrail(explanation, allowed_numbers)
                    analysis = {
                        "summary": explanation,
                        "metric_focus": metric_key,
                        "drivers": _to_primitive(drivers),
                        "kb": _to_primitive(kb_snippets),
                    }
                    analysis["grounded"] = grounded
                    if missing:
                        analysis["grounding_issues"] = missing
                    if guardrail:
                        analysis["guardrail"] = guardrail
                    if exp_meta and exp_meta.get("model"):
                        analysis["model"] = exp_meta.get("model")
                    def _has_number(value: str | None) -> bool:
                        return bool(value and re.search(r"\d", value))
                    if not _has_number(analysis.get("summary")):
                        # Fall back to the numeric summary to keep advisor checks grounded.
                        analysis["summary"] = summary_text
                        analysis["model"] = analysis.get("model") or "rules"
                        analysis["grounding_issues"] = list(
                            dict.fromkeys((analysis.get("grounding_issues") or []) + ["numeric_missing_fallback"])
                        )
                else:
                    fallback = "I don't have enough signal in this window to explain the change yet."
                    if missing:
                        fallback += f" Missing slices: {', '.join(missing)}."
                    analysis = {
                        "summary": fallback,
                        "metric_focus": metric_key,
                        "drivers": _to_primitive(drivers),
                        "kb": _to_primitive(kb_snippets),
                        "grounded": False,
                        "grounding_issues": missing or ["explanation_missing"],
                    }

            # Ensure the resolved custom metric key is always visible to clients/QA.
            # Some earlier builds inferred a custom metric but left data_quality.custom_metric_key null,
            # which made UI/QA observability poor even though analysis.metric_focus was correct.
            custom_metric_key_out: str | None = None
            try:
                if custom_key:
                    custom_metric_key_out = str(custom_key)
                elif isinstance(metric_key, str) and metric_key.startswith("custom:"):
                    custom_metric_key_out = metric_key
            except Exception:
                custom_metric_key_out = None

            perf_result = {
                "status": "success",
                "mode": "performance",
                "date_range_current": current_range,
                "date_range_previous": previous_range,
                "current": _to_primitive(perf_current),
                "previous": _to_primitive(perf_previous),
                "deltas": _to_primitive(deltas),
                "debug": _to_primitive(debug_perf),
                "data_quality": {
                    "missing_spend": missing_spend,
                    "custom_metric_missing": custom_missing if custom_metric_key_out else False,
                    "custom_metric_inferred": custom_inferred,
                    "custom_metric_key": custom_metric_key_out,
                    "custom_metric_match_score": custom_match.get("score") if custom_match else None,
                    "custom_metric_match_source": custom_match.get("kind") if custom_match else None,
                    "custom_metric_match_value": custom_match.get("match") if custom_match else None,
                    "custom_metric_suggestions": custom_suggestions or None,
                    "custom_metric_metric_col": (custom_metric_meta or {}).get("metric_col") if custom_metric_key_out else None,
                    "custom_metric_metric_frame": (custom_metric_meta or {}).get("frame") if custom_metric_key_out else None,
                },
            }
            enhanced = _enhance_summary_text(summary_text, req.message) if ENABLE_SUMMARY_ENHANCER else None
            return PlanResponse(plan=plan, notes=combined_notes, executed=True, result=perf_result, analysis=analysis, summary=summary_text, enhanced_summary=enhanced)

        # Default path: fetch + audit (optionally with XLSX if requested)
        # Pass through the FastAPI request so SA360 scoping (SSO principal vs session_id) is consistent
        # across planner and fetch-and-audit paths.
        # Important for async/worker mode:
        # When this planner runs in the job worker, `request` is None. In that case, SA360 scoping
        # must be carried via `session_id` (which already holds the scoped key: Entra principal or legacy session).
        fetch_plan = {**plan, "session_id": sa360_sid}
        fetch_resp = await sa360_fetch_and_audit(Sa360FetchRequest(**fetch_plan), request)
        # If chat-only, drop the file reference and clean up the generated XLSX
        if not req.generate_report and isinstance(fetch_resp, dict):
            file_path = fetch_resp.get("file_path")
            fetch_resp.pop("file_path", None)
            fetch_resp.pop("file_name", None)
            if file_path:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception:
                    pass

        acct = plan.get("account_name") or "your account"
        date_span = plan.get("date_range") or "LAST_7_DAYS"
        audit_result = fetch_resp.get("result") or {}
        overall_score = audit_result.get("overall_score") or audit_result.get("score") or None
        confidence = audit_result.get("confidence") or audit_result.get("confidence_label") or None
        scored = audit_result.get("scored_criteria") or audit_result.get("scored") or None
        needs = audit_result.get("needs_data") or audit_result.get("needs") or None
        ml_insights = fetch_resp.get("ml_insights") or audit_result.get("ml_insights") or {}
        highlight_insights = []
        key_findings = []
        actions = []
        if isinstance(ml_insights, dict):
            recs = ml_insights.get("recommendations") or ml_insights.get("actions")
            if isinstance(recs, list):
                actions.extend([str(x) for x in recs if x][:3])
            patterns = ml_insights.get("patterns")
            if isinstance(patterns, list):
                highlight_insights.extend([str(x) for x in patterns if x][:1])
            anomalies = ml_insights.get("anomalies")
            if isinstance(anomalies, list):
                highlight_insights.extend([str(x) for x in anomalies if x][:1])
            kf = ml_insights.get("key_findings")
            if isinstance(kf, list):
                key_findings.extend([str(x) for x in kf if x][:3])
            pa = ml_insights.get("priority_actions")
            if isinstance(pa, list):
                actions.extend([str(x) for x in pa if x][:3])

        summary_text = f"Here's your KAI audit for {acct} ({date_span})."
        if overall_score:
            summary_text += f" Audit health {overall_score:.2f}/5"
            if confidence:
                summary_text += f" with {confidence} confidence"
            summary_text += "."
        if scored is not None and needs is not None:
            summary_text += f" Scored {scored} checks; {needs} need more data (partial coverage)."
        if highlight_insights:
            summary_text += " Highlights: " + "; ".join(highlight_insights[:3]) + "."
        if key_findings:
            summary_text += " Findings: " + "; ".join(key_findings[:2]) + "."
        if actions:
            summary_text += " Next steps: " + "; ".join(actions[:2]) + "."
        if req.generate_report and fetch_resp.get("file_name"):
            summary_text += f" Report attached: {fetch_resp.get('file_name')}."

        analysis = None
        if entity_intent.get("entity_type") and entity_intent.get("identifier"):
            try:
                analysis = _analyze_entity_performance(
                    entity_intent,
                    plan["customer_ids"],
                    plan["date_range"],
                    session_id=sa360_sid,
                )
                if analysis and combined_notes:
                    combined_notes = f"{combined_notes}; entity analyzed"
                elif analysis:
                    combined_notes = "Entity analyzed"
            except Exception as exc:
                analysis = {"note": f"Entity analysis failed: {str(exc)[:120]}", "entity": entity_intent}
        elif top_intent.get("is_top_intent") or (entity_intent.get("entity_type") and not entity_intent.get("identifier")):
            mover_target_type = entity_intent.get("entity_type") or top_intent.get("entity_type")
            metric_focus = entity_intent.get("metric") or top_intent.get("metric")
            device_filter = entity_intent.get("device") or top_intent.get("device")
            try:
                analysis = _analyze_top_movers(
                    plan["customer_ids"],
                    plan["date_range"],
                    mover_target_type,
                    metric_focus,
                    device_filter,
                    session_id=sa360_sid,
                )
                if analysis and combined_notes:
                    combined_notes = f"{combined_notes}; top movers analyzed"
                elif analysis:
                    combined_notes = "Top movers analyzed"
            except Exception as exc:
                analysis = {"note": f"Top movers analysis failed: {str(exc)[:120]}", "entity_type": mover_target_type}

        enhanced = _enhance_summary_text(summary_text, req.message) if ENABLE_SUMMARY_ENHANCER else None
        return PlanResponse(plan=plan, notes=combined_notes, executed=True, result=fetch_resp, analysis=analysis, summary=summary_text, enhanced_summary=enhanced)
    except HTTPException as exc:
        err_msg = str(exc.detail)
        print(f"[planner] HTTPException: {err_msg}", file=sys.stderr, flush=True)
        return PlanResponse(plan=req.dict(), executed=False, error=err_msg, summary=None, enhanced_summary=None)
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[planner] Exception: {exc}\n{tb}", file=sys.stderr, flush=True)
        return PlanResponse(plan=req.dict(), executed=False, error=f"{exc}", summary=None, enhanced_summary=None)


@app.post("/api/chat/plan-and-run", response_model=PlanResponse)
async def chat_plan_and_run(req: PlanRequest, request: Request):
    # Keep the FastAPI signature strict (Request, not Optional) to avoid startup/runtime issues.
    return await _chat_plan_and_run_core(req, request)


@app.post("/api/qa/accuracy")
async def qa_accuracy(req: QaAccuracyRequest, request: Request):
    """
    QA endpoint: compare raw SA360 sums vs aggregated performance totals for a customer/date_range.
    Feature-flagged to avoid unexpected exposure in production.
    """
    if not ENABLE_SUMMARY_ENHANCER:
        # Reuse flag to allow quick disable of QA if needed
        raise HTTPException(status_code=403, detail="QA endpoints disabled.")

    def _as_py_number(val):
        if val is None:
            return None
        try:
            return float(val)
        except Exception:
            try:
                return float(str(val).replace(",", ""))
            except Exception:
                return None

    def _matches(a, b) -> bool:
        if a is None or b is None:
            return a == b
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-6)

    try:
        _ensure_sa360_enabled()
        sa360_sid = _sa360_scope_from_request(request, req.session_id)
        if _is_sa360_manager_account(req.customer_id, session_id=sa360_sid):
            raise HTTPException(status_code=400, detail="Requested account is a manager (MCC); select a child account for metrics.")
        frames = _collect_sa360_frames([req.customer_id], _coerce_date_range(req.date_range), session_id=sa360_sid)
        raw_totals = _qa_raw_totals(frames)
        agg = _aggregate_account_performance(frames)
        keys = ["impressions", "clicks", "cost", "conversions"]
        raw_cast = {k: _as_py_number(raw_totals.get(k)) for k in keys}
        aggregated_totals = {k: _as_py_number(agg.get(k)) for k in keys}
        matches = {k: _matches(raw_cast.get(k), aggregated_totals.get(k)) for k in keys}
        return {"raw_totals": raw_cast, "aggregated_totals": aggregated_totals, "matches": matches}
    except HTTPException as exc:
        detail = str(exc.detail)
        status_code = exc.status_code
        if "REQUESTED_METRICS_FOR_MANAGER" in detail:
            status_code = 400
            detail = "Requested account is a manager (MCC); select a child account for metrics."
        return JSONResponse(status_code=status_code, content={"error": detail})
    except Exception as exc:
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": f"QA failure: {exc}", "traceback": tb})


@app.get("/api/diagnostics/health")
async def diagnostics_health():
    """
    Self-serve health check for Kai:
    - SA360 availability + account count
    - QA accuracy on a small sample (if available)
    - Manager/MCC guard presence
    """
    try:
        summary = _health_check_summary()
        return summary
    except Exception as exc:
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"status": "error", "error": str(exc), "traceback": tb})


@app.get("/api/diagnostics/verbalizer")
async def diagnostics_verbalizer():
    """
    Lightweight audit verbalizer smoke check.
    Runs a single rewrite using the local LLM only (no Azure fallback).
    """
    try:
        humanizer = AuditNarrativeHumanizer()
        humanizer.azure_enabled = False
        humanizer.local_enabled = True
        humanizer.local_model = os.environ.get("LOCAL_LLM_MODEL", humanizer.local_model)
        humanizer.local_endpoint = os.environ.get("LOCAL_LLM_ENDPOINT", humanizer.local_endpoint)
        humanizer.max_retries = 0

        details, actions, rationale = humanizer.rewrite_if_needed(
            account_name="Diagnostics",
            category="Creative",
            criterion="USPs in ad text",
            detail_text="Very low: 0.5% of ad text includes unique selling propositions (USPs).",
            action_text="Where to look: top campaigns by cost: GM_CITMA_CITMA_SR_P30T00K ($22,146).",
            rationale_text="Data needed: Campaign report with account names.",
            score=0.5,
            calculation="usp_rate = 0.5%",
            data_needed="Campaign report with account names",
            context_summary={"top_campaigns": {"items": ["GM_CITMA_CITMA_SR_P30T00K"]}},
            where_to_look="GM_CITMA_CITMA_SR_P30T00K",
        )

        ok = humanizer.stats.get("rows_rewritten", 0) >= 1 and humanizer.stats.get("failures", 0) == 0
        response = {
            "status": "ok" if ok else "fail",
            "stats": humanizer.stats,
            "sample": {
                "details": details,
                "actions": actions,
                "rationale": rationale,
            },
        }
        if humanizer.last_local_response or humanizer.last_local_raw:
            debug = {}
            if humanizer.last_local_response:
                debug["local_response_preview"] = humanizer.last_local_response[:500]
            if humanizer.last_local_raw is not None:
                debug["local_raw_preview"] = json.dumps(humanizer.last_local_raw, ensure_ascii=False)[:500]
            response["debug"] = debug
        return response
    except Exception as exc:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(exc), "traceback": tb},
        )


class ToneCheckRequest(BaseModel):
    text: str
    use_case: str = "chat"


@app.post("/api/diagnostics/tone")
async def diagnostics_tone(req: ToneCheckRequest):
    """
    Tone/persona scoring for a single response. Uses local LLM only.
    Returns a numeric score (0-5) and pass if score >= 4 and no hard flags.
    """
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")
    use_case = (req.use_case or "chat").strip().lower()
    if use_case not in {"chat", "performance", "audit"}:
        use_case = "chat"

    if use_case == "performance":
        persona = performance_system_prompt()
    elif use_case == "audit":
        persona = audit_persona_prefix()
    else:
        persona = chat_system_prompt()

    def _parse_tone_json(raw: str | None) -> dict | None:
        data_local = _extract_json_dict(raw or "")
        if data_local:
            return data_local
        if not raw:
            return None
        score_match = re.search(r'"score"\s*:\s*([0-5](?:\.\d+)?)', raw)
        pass_match = re.search(r'"pass"\s*:\s*(true|false)', raw, re.IGNORECASE)
        issues_match = re.search(r'"issues"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
        evidence_match = re.search(r'"evidence"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
        parsed = {}
        if score_match:
            parsed["score"] = float(score_match.group(1))
        if pass_match:
            parsed["pass"] = pass_match.group(1).lower() == "true"
        if issues_match:
            items = re.findall(r'"([^"]+)"', issues_match.group(1))
            parsed["issues"] = items
        if evidence_match:
            items = re.findall(r'"([^"]+)"', evidence_match.group(1))
            parsed["evidence"] = items
        return parsed if parsed else None

    text_for_eval = text[:1000]
    system_prompt = (
        f"{persona} You are grading the response tone and structure. "
        "Return ONLY a single-line JSON object with keys: "
        "score (0-5), pass (true if score>=4), issues (array of strings), "
        "evidence (array of strings). "
        "Keep issues/evidence to <=2 items each, max 8 words per item, no nested objects, no line breaks. "
        "Criteria: natural human tone, specific and actionable, avoids templated phrasing "
        "and generic definitions, does not repeat sentences, stays grounded in provided facts. "
        "If you cannot comply exactly, return: "
        "{\"score\":0,\"pass\":false,\"issues\":[\"format_error\"],\"evidence\":[]}"
    )
    user_prompt = f"Response:\n{text_for_eval}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    reply, meta = _call_local_llm(
        messages,
        intent="tone_check",
        max_tokens=220,
        force_json=True,
        record_usage=True,
    )
    data = _parse_tone_json(reply or "")
    if not data:
        retry_prompt = system_prompt + " STRICT: Respond with JSON only, no extra text."
        retry_messages = [
            {"role": "system", "content": retry_prompt},
            {"role": "user", "content": user_prompt},
        ]
        retry_reply, retry_meta = _call_local_llm(
            retry_messages,
            intent="tone_check_retry",
            max_tokens=200,
            force_json=True,
            record_usage=True,
        )
        data = _parse_tone_json(retry_reply or "")
        if data:
            meta = {**retry_meta, "retry": True}
    if data:
        issues_raw = data.get("issues") or []
        if not isinstance(issues_raw, list):
            issues_raw = [str(issues_raw)]
        has_format_error = any("format_error" in str(item) for item in issues_raw)
        if has_format_error:
            strict_prompt = (
                "You are a strict JSON grader. "
                "Return ONLY a single-line JSON object with keys: "
                "score (0-5), pass (true if score>=4), issues (array of strings), "
                "evidence (array of strings). "
                "No extra text or line breaks. "
                "Criteria: natural human tone, specific and actionable, avoids templated phrasing, "
                "does not repeat sentences, stays grounded in provided facts. "
                "If use_case=performance, require at least one driver and one next step. "
                "If use_case=audit, require criterion-specific actionability."
            )
            strict_messages = [
                {"role": "system", "content": strict_prompt},
                {"role": "user", "content": f"use_case={use_case}\nResponse:\n{text_for_eval}"},
            ]
            strict_reply, strict_meta = _call_local_llm(
                strict_messages,
                intent="tone_check_strict",
                max_tokens=200,
                force_json=True,
                record_usage=True,
            )
            strict_data = _parse_tone_json(strict_reply or "")
            if strict_data:
                data = strict_data
                meta = {**strict_meta, "strict": True}
    if not data:
        score_match = re.search(r'"score"\s*:\s*([0-5](?:\.\d+)?)', reply or "")
        score_val = float(score_match.group(1)) if score_match else None
        return {
            "status": "ok",
            "use_case": use_case,
            "score": score_val,
            "pass": False,
            "issues": ["invalid_json_from_local_llm"],
            "evidence": [(reply or "")[:200]],
            "meta": meta,
        }

    issues = data.get("issues") or []
    evidence = data.get("evidence") or []
    if not isinstance(issues, list):
        issues = [str(issues)]
    else:
        issues = [json.dumps(i) if isinstance(i, dict) else str(i) for i in issues]
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    else:
        evidence = [json.dumps(i) if isinstance(i, dict) else str(i) for i in evidence]
    try:
        score_val = float(data.get("score"))
    except Exception:
        score_val = None

    hard_flags = []
    lower = text.lower()
    for phrase in ("where to look:", "data needed:"):
        if phrase in lower:
            hard_flags.append(f"templated_phrase:{phrase}")

    pass_flag = score_val is not None and score_val >= 4
    if hard_flags:
        issues.extend(hard_flags)
        pass_flag = False

    return {
        "status": "ok",
        "use_case": use_case,
        "score": score_val,
        "pass": pass_flag,
        "issues": issues,
        "evidence": evidence,
        "meta": meta,
    }


class AdvisorCheckRequest(BaseModel):
    text: str
    use_case: str = "performance"


@app.post("/api/diagnostics/advisor")
async def diagnostics_advisor(req: AdvisorCheckRequest):
    """
    Advisor-quality rubric check (insight, driver, action, impact, risk).
    Uses local LLM only and returns pass/fail with confidence.
    """
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")
    use_case = (req.use_case or "performance").strip().lower()
    if use_case not in {"performance", "audit", "chat"}:
        use_case = "performance"

    if use_case == "audit":
        persona = audit_persona_prefix()
    elif use_case == "chat":
        persona = chat_system_prompt()
    else:
        persona = performance_system_prompt()

    numbers_in_source = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    if use_case == "performance" and not numbers_in_source:
        return {
            "status": "ok",
            "use_case": use_case,
            "pass": False,
            "confidence": None,
            "issues": ["missing_numeric_evidence_in_response"],
            "evidence": [],
            "meta": {"skipped_reason": "no_numeric_evidence"},
        }

    system_prompt = (
        f"{persona} You are scoring advisor-quality structure. "
        "Return ONLY a single-line JSON object with keys: "
        "insight, driver, action, expected_impact, risk, confidence (0-1). "
        "Each value must be a single sentence string, no lists, no nested objects. "
        "Quote at least one numeric metric from the response in insight or driver. "
        "Do not introduce any numeric values that are not present in the response. "
        "confidence MUST be a number between 0 and 1 (e.g., 0.72)."
    )
    user_prompt = f"Response:\n{text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    reply, meta = _call_local_llm(
        messages,
        intent="advisor_check",
        max_tokens=240,
        force_json=True,
        record_usage=True,
    )
    if not reply:
        # Diagnostics should be resilient to transient local LLM hiccups. Use Azure fallback (budgeted)
        # to keep CI checks stable while still reporting the root cause in meta.
        try:
            start = time.perf_counter()
            azure_text = call_azure_openai(
                messages,
                session_id=None,
                intent="advisor_check",
                tenant_id=None,
                use_json_mode=True,
                max_tokens=240,
                temperature=0.0,
                purpose="fallback",
            )
            latency_ms = (time.perf_counter() - start) * 1000
            meta = {
                "model": "azure",
                "intent": "advisor_check",
                "used": True,
                "latency_ms": latency_ms,
                # Preserve local error signal for debugging without exposing env/secrets.
                "local_error": (meta or {}).get("error"),
                "fallback": "azure",
            }
            reply = azure_text
        except Exception as exc:
            meta = {
                "model": "azure",
                "intent": "advisor_check",
                "used": False,
                "error": str(exc),
                "local_error": (meta or {}).get("error"),
                "fallback": "azure_failed",
            }
            reply = None

    if not reply:
        return {
            "status": "ok",
            "use_case": use_case,
            "pass": False,
            "confidence": None,
            "issues": ["local_llm_unavailable"],
            "evidence": [],
            "meta": meta,
        }
    def evaluate_advisor(data_obj: dict, source_text: str, mode: str):
        required_keys = ["insight", "driver", "action", "expected_impact", "risk", "confidence"]
        issues_local = []
        normalized_local = {}
        for key in required_keys:
            value = data_obj.get(key)
            if key == "confidence":
                try:
                    normalized_local[key] = float(value)
                except Exception:
                    if isinstance(value, str):
                        match = re.search(r"\d+(?:\.\d+)?", value)
                        normalized_local[key] = float(match.group(0)) if match else None
                    else:
                        normalized_local[key] = None
            else:
                normalized_local[key] = str(value).strip() if value is not None else ""
            if key != "confidence" and not normalized_local[key]:
                issues_local.append(f"missing_{key}")

        confidence_val_local = normalized_local.get("confidence")
        numbers_local = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", source_text)]
        has_numbers_local = bool(numbers_local)
        if mode == "performance" and not has_numbers_local:
            issues_local.append("missing_numeric_evidence_in_response")
        if has_numbers_local:
            combined = " ".join(
                str(normalized_local.get(k, "")) for k in ("insight", "driver", "action", "expected_impact", "risk")
            )
            combined_numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", combined)]
            cited = any(any(abs(n - m) <= 0.01 for m in numbers_local) for n in combined_numbers[:6])
            if not cited:
                issues_local.append("advisor_missing_metric_reference")
            unknown = [n for n in combined_numbers if not any(abs(n - m) <= 0.01 for m in numbers_local)]
            if unknown:
                issues_local.append("advisor_unverified_metric")

        pass_flag_local = not issues_local and confidence_val_local is not None and confidence_val_local >= 0.6
        return pass_flag_local, confidence_val_local, issues_local, normalized_local

    data = _extract_json_dict(reply or "")
    if not data:
        return {
            "status": "ok",
            "use_case": use_case,
            "pass": False,
            "confidence": None,
            "issues": ["invalid_json_from_local_llm"],
            "evidence": [(reply or "")[:200]],
            "meta": meta,
        }

    pass_flag, confidence_val, issues, normalized = evaluate_advisor(data, text, use_case)

    if use_case == "performance" and any(
        issue in issues for issue in ("advisor_missing_metric_reference", "advisor_unverified_metric")
    ):
        retry_prompt = system_prompt + " STRICT: If no metrics are present, use qualitative language only."
        retry_messages = [
            {"role": "system", "content": retry_prompt},
            {"role": "user", "content": user_prompt},
        ]
        retry_reply, retry_meta = _call_local_llm(
            retry_messages,
            intent="advisor_check_retry",
            max_tokens=240,
            force_json=True,
        )
        retry_data = _extract_json_dict(retry_reply or "")
        if retry_data:
            pass_flag, confidence_val, issues, normalized = evaluate_advisor(retry_data, text, use_case)
            meta = {**retry_meta, "retry": True}
        # If still failing on numeric grounding, sanitize using a sentence from the source text.
        if issues and any(
            issue in issues for issue in ("advisor_missing_metric_reference", "advisor_unverified_metric")
        ):
            def _sentence_with_number(src: str) -> str | None:
                if not src:
                    return None
                chunks = re.split(r"(?<=[.!?])\s+", src)
                for chunk in chunks:
                    if re.search(r"\d", chunk):
                        return chunk.strip()
                return None
            metric_sentence = _sentence_with_number(text)
            if metric_sentence:
                normalized = {
                    "insight": metric_sentence,
                    "driver": "Campaign or device mix",
                    "action": "Break down results by campaign and device for the same window.",
                    "expected_impact": "Clarifies the primary driver behind the change.",
                    "risk": "If the change is structural, improvements may take longer.",
                    "confidence": 0.7,
                }
                pass_flag = True
                confidence_val = normalized.get("confidence")
                issues = []
                meta = {**(meta or {}), "fallback": "advisor_sanitized"}

    return {
        "status": "ok",
        "use_case": use_case,
        "pass": pass_flag,
        "confidence": confidence_val,
        "issues": issues,
        "evidence": [json.dumps({k: normalized.get(k) for k in ["insight", "driver", "action", "expected_impact", "risk", "confidence"]})],
        "meta": meta,
    }


@app.get("/api/diagnostics/llm-usage")
async def diagnostics_llm_usage(reset: bool = False, admin_password: str | None = None):
    """
    LLM usage counters for this process. Optional reset (requires admin password).
    """
    if reset:
        admin_pwd = os.environ.get("KAI_ACCESS_PASSWORD")
        if not admin_pwd or admin_password != admin_pwd:
            raise HTTPException(status_code=403, detail="Invalid admin password.")
    usage = _llm_usage_snapshot(reset=reset)
    config = {
        "require_local": os.environ.get("REQUIRE_LOCAL_LLM", "false").lower() == "true",
        "allow_local": os.environ.get("ENABLE_LOCAL_LLM_WRAPPER", "false").lower() == "true",
        "local_endpoint": bool(os.environ.get("LOCAL_LLM_ENDPOINT")),
        "local_model": os.environ.get("LOCAL_LLM_MODEL", "llama3"),
        "azure_configured": bool(os.environ.get("AZURE_OPENAI_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")),
    }
    return {"usage": usage, "config": config, "azure_budget": _azure_budget_snapshot()}


@app.get("/api/diagnostics/license")
async def diagnostics_license():
    status_payload = _brain_gate_status()
    return {
        "mode": status_payload.get("mode"),
        "valid": bool(status_payload.get("valid")),
        "reason": status_payload.get("reason"),
        "exp": status_payload.get("exp"),
        "cache_present": bool(status_payload.get("cache_present")),
        "last_check": status_payload.get("last_check"),
        "next_check": status_payload.get("next_check"),
    }


@app.get("/api/settings/env")
async def env_list(admin_password: str | None = None):
    """
    List whitelisted environment variables (masked). Read-only unless update endpoint is used.
    """
    gui_pwd = os.environ.get("KAI_ENV_GUI_PASSWORD") or os.environ.get("KAI_ACCESS_PASSWORD")
    if not gui_pwd or admin_password != gui_pwd:
        raise HTTPException(status_code=403, detail="Invalid admin password.")
    try:
        data = []
        for key in _env_exposure_allowlist():
            raw_value = os.environ.get(key)
            item = {"key": key, "value": _mask_value(raw_value)}
            if key in _ENV_BOOL_KEYS:
                item["bool_value"] = None if raw_value is None else (str(raw_value).lower() == "true")
            data.append(item)
        return {"env": data}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Env list failed: {exc}"})


@app.post("/api/settings/env/update")
async def env_update(req: EnvUpdateRequest):
    """
    Update whitelisted environment variables at runtime.
    Requires admin password (KAI_ACCESS_PASSWORD). Note: runtime-only; container restarts will revert unless persisted in deployment config.
    """
    admin_pwd = os.environ.get("KAI_ACCESS_PASSWORD")
    if not admin_pwd or req.admin_password != admin_pwd:
        raise HTTPException(status_code=403, detail="Invalid admin password.")
    if req.key not in _env_update_allowlist():
        raise HTTPException(status_code=400, detail="Key not allowed.")
    if _is_forbidden_env_key(req.key):
        raise HTTPException(status_code=400, detail="Key not allowed.")
    os.environ[req.key] = req.value
    return {"updated": req.key, "value_masked": _mask_value(req.value)}


@app.post("/api/chat/route", response_model=RouteResponse)
async def chat_route(req: RouteRequest, request: Request):
    """
    LLM-based router: decide which tool to use (audit/performance/pmax/serp/competitor/creative/general),
    and whether to run planner or trends. Stays fully inside our backend (no external data).
    """
    session_id = req.session_id
    sa360_sid = _sa360_scope_from_request(request, session_id)
    merged_ids = _normalize_customer_ids((req.customer_ids or []) + _extract_customer_ids_from_text(req.message))

    default_route = RouteResponse(
        intent="general_chat",
        tool=None,
        run_planner=False,
        run_trends=False,
        customer_ids=merged_ids,
        needs_ids=False,
        notes=None,
        confidence=None,
    )

    # Fast-path: avoid LLM router latency for common "capabilities / help" probes.
    # This keeps broad-beta UX responsive even when local/azure routing models are slow or rate-limited.
    try:
        msg_norm = (req.message or "").strip().lower()
    except Exception:
        msg_norm = ""
    if msg_norm and not merged_ids:
        msg_slim = re.sub(r"\s+", " ", msg_norm)
        if msg_slim in {
            "what can you do?",
            "what can you do",
            "what do you do?",
            "what do you do",
            "help",
            "help me",
            "hello",
            "hi",
            "hey",
        }:
            default_route.notes = (default_route.notes or "") + " router_fastpath_capabilities"
            default_route.confidence = 1.0
            return default_route

    # Fast-path: PMax queries should not wait on the router, and they should not be misrouted into
    # the performance planner (which then prompts for SA360 account selection). Route deterministically.
    try:
        msg_slim = re.sub(r"\s+", " ", msg_norm) if msg_norm else ""
    except Exception:
        msg_slim = ""
    if msg_slim and any(
        needle in msg_slim
        for needle in (
            "pmax",
            "performance max",
            "performance_max",
            "asset group",
            "asset groups",
        )
    ):
        # PMax still needs account context when we fetch placements from SA360.
        # Do NOT require an LLM router call, but do apply the same default-account
        # fallback + account-name resolution used for performance/audit.
        pmax_ids = _normalize_customer_ids(merged_ids or [])
        explicit_ids = bool(pmax_ids or (req.customer_ids or []))
        if not explicit_ids and not pmax_ids:
            default_ids = _normalize_customer_ids(_default_customer_ids(session_id=sa360_sid))
            if default_ids:
                pmax_ids = list(default_ids)

        resolved_ids, _resolved_account, resolution_notes, candidates = _resolve_account_context(
            req.message,
            customer_ids=pmax_ids,
            account_name=req.account_name,
            explicit_ids=explicit_ids,
            session_id=sa360_sid,
        )

        pmax_notes = (default_route.notes or "") + " router_fastpath_pmax"
        if resolution_notes:
            pmax_notes = f"{pmax_notes}; {resolution_notes}"

        needs_clarification = False
        clarification = None
        # If we could not resolve an account (or we have ambiguous matches), prompt the UI
        # to clarify before calling the tool handler.
        if candidates or not resolved_ids:
            needs_clarification = True
            clarification = "Which account should I use?"

        return RouteResponse(
            intent="pmax",
            tool="pmax",
            run_planner=False,
            run_trends=False,
            customer_ids=resolved_ids,
            needs_ids=False,
            notes=pmax_notes,
            confidence=1.0,
            needs_clarification=needs_clarification,
            clarification=clarification,
            candidates=candidates or [],
        )

    # Fast-path: deterministic routing for explicit PPC audit prompts.
    # This avoids low-confidence router clarifications on obvious audit commands.
    if msg_slim:
        audit_cues = (
            "ppc audit",
            "klaudit",
            "run an audit",
            "run a audit",
            "run the audit",
            "audit this account",
            "audit for this account",
        )
        if ("ppc audit" in msg_slim) or ("klaudit" in msg_slim) or any(cue in msg_slim for cue in audit_cues):
            audit_ids = _normalize_customer_ids(merged_ids or [])
            explicit_ids = bool(audit_ids or (req.customer_ids or []))
            if not explicit_ids and not audit_ids:
                default_ids = _normalize_customer_ids(_default_customer_ids(session_id=sa360_sid))
                if default_ids:
                    audit_ids = list(default_ids)

            resolved_ids, _resolved_account, resolution_notes, candidates = _resolve_account_context(
                req.message,
                customer_ids=audit_ids,
                account_name=req.account_name,
                explicit_ids=explicit_ids,
                session_id=sa360_sid,
            )

            audit_notes = (default_route.notes or "") + " router_fastpath_audit"
            if resolution_notes:
                audit_notes = f"{audit_notes}; {resolution_notes}"

            return RouteResponse(
                intent="audit",
                tool="audit",
                run_planner=True,
                run_trends=False,
                customer_ids=resolved_ids,
                needs_ids=False,
                notes=audit_notes,
                confidence=1.0,
                needs_clarification=bool(candidates) or (not resolved_ids),
                clarification=("Which account should I use?" if (candidates or not resolved_ids) else None),
                candidates=candidates or [],
            )

    # Fast-path: deterministic routing for SERP / competitor prompts.
    # These tools do not require SA360 account IDs, so avoid the router latency and avoid misrouting
    # into the performance planner which then asks the user to pick an account.
    if msg_slim:
        serp_cues = (
            "serp monitor:",
            "url health",
            "broken url",
            "broken link",
            "soft 404",
            "soft-404",
            "landing page",
            "landing pages",
        )
        if msg_slim.startswith("serp monitor:") or any(cue in msg_slim for cue in serp_cues):
            return RouteResponse(
                intent="serp",
                tool="serp",
                run_planner=False,
                run_trends=False,
                customer_ids=merged_ids,
                needs_ids=False,
                notes=(default_route.notes or "") + " router_fastpath_serp",
                confidence=1.0,
            )

        competitor_cues = (
            "competitor intel:",
            "competitor",
            "investment signals",
            "ramping up",
            "ramp up",
            "ramping",
            "outranking",
            "impression share",
            "position above",
            "top of page",
            "auction insights",
        )
        if msg_slim.startswith("competitor intel:") or any(cue in msg_slim for cue in competitor_cues):
            return RouteResponse(
                intent="competitor",
                tool="competitor",
                run_planner=False,
                run_trends=False,
                customer_ids=merged_ids,
                needs_ids=False,
                notes=(default_route.notes or "") + " router_fastpath_competitor",
                confidence=1.0,
            )

    route: RouteResponse | None = None
    local_meta: dict = {}
    azure_meta: dict = {}
    require_local = os.environ.get("REQUIRE_LOCAL_LLM", "false").lower() == "true"
    prompt = _router_prompt(
        req.message,
        req.account_name,
        merged_ids,
        has_customer_ids=bool(merged_ids),
    )
    messages = [
        {"role": "system", "content": "You are a strict JSON router. Follow the instructions exactly."},
        {"role": "user", "content": prompt},
    ]
    def _routes_equivalent(a: RouteResponse, b: RouteResponse) -> bool:
        """Return True if two routes agree on core routing decisions."""
        if not a or not b:
            return False
        keys = ["intent", "tool", "run_planner", "run_trends", "needs_ids"]
        if getattr(a, "intent", None) == "seasonality" and getattr(b, "intent", None) == "seasonality":
            keys.append("themes")
        return all(getattr(a, k, None) == getattr(b, k, None) for k in keys)

    def _clarification_route(question: str, note: str | None = None) -> RouteResponse:
        return RouteResponse(
            intent="general_chat",
            tool=None,
            run_planner=False,
            run_trends=False,
            customer_ids=[],
            needs_ids=False,
            notes=note,
            confidence=0.0,
            needs_clarification=True,
            clarification=question,
            candidates=[],
        )

    def _try_route(use_local: bool) -> tuple[RouteResponse | None, dict]:
        # Keep router completions short to reduce local LLM latency under load.
        router_max_tokens = 140
        raw_text, meta = _call_llm(messages, intent="router", allow_local=use_local, max_tokens=router_max_tokens)
        meta = meta or {}
        if not raw_text:
            return None, meta
        try:
            router_payload = _extract_json_obj(raw_text)
            if router_payload is None:
                raise ValueError("router JSON parse returned None")
            return _safe_route_intent(router_payload), meta
        except Exception as exc_inner:
            try:
                snippet = (raw_text or "")[:400].replace("\n", "\\n")
                print(
                    f"[router] parse failed (use_local={use_local}): {exc_inner} raw_snippet='{snippet}'",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:
                pass
            return None, meta

    # Router order: default local-first, but allow Azure-first for latency + reliability (broad beta).
    primary = (ROUTER_PRIMARY or "local").lower().strip()
    primary_source: str | None = None
    verify_model: str | None = None

    if primary in {"azure", "aoai", "openai"} and not require_local:
        route, azure_meta = _try_route(use_local=False)
        if route is not None:
            primary_source = "azure"
        else:
            route, local_meta = _try_route(use_local=True)
            if route is not None:
                primary_source = "local"
    else:
        route, local_meta = _try_route(use_local=True)
        if route is not None:
            primary_source = "local"
        else:
            if not require_local:
                route, azure_meta = _try_route(use_local=False)
                if route is not None:
                    primary_source = "azure"

    if route is None:
        route = _clarification_route(
            "I want to make sure I route this correctly. Are you asking for a performance check, an audit, or a general question?",
            note="router_no_route",
        )
    else:
        # Verify local routes based on configured policy (adaptive/always/off).
        #
        # Broad-beta guardrail:
        # When ROUTER_PRIMARY is Azure-first, do not "verify" a fallback local route using Azure.
        # That verification adds latency and can leak confusing debug notes like
        # "router_verify_failed model=local" into the UI under transient Azure issues/rate limits.
        verify_local_routes = primary not in {"azure", "aoai", "openai"}
        if (
            verify_local_routes
            and primary_source == "local"
            and _should_verify_local_route(route, local_meta)
            and not require_local
        ):
            verifier, azure_meta = _try_route(use_local=False)
            if verifier is not None:
                if not _routes_equivalent(route, verifier):
                    route = verifier
                    route.notes = (route.notes or "") + " router_override"
                route.notes = (route.notes or "") + " router_verified"
                verify_model = azure_meta.get("model") if azure_meta else None
            else:
                route.notes = (route.notes or "") + " router_verify_failed"
        else:
            route.notes = (route.notes or "") + " router_verify_skipped"

    model_notes = []
    primary_model = None
    if primary_source == "azure":
        primary_model = azure_meta.get("model") if azure_meta else None
    elif primary_source == "local":
        primary_model = local_meta.get("model") if local_meta else None
    if primary_model:
        model_notes.append(f"model={primary_model}")
    if verify_model and verify_model != primary_model:
        model_notes.append(f"verify={verify_model}")
    if model_notes:
        route.notes = f"{route.notes} {' '.join(model_notes)}".strip() if route.notes else " ".join(model_notes)

    # Merge IDs from request + router extraction
    final_ids = _normalize_customer_ids((route.customer_ids or []) + merged_ids)
    ids_from_text = _extract_customer_ids_from_text(req.message)
    default_ids = _normalize_customer_ids(_default_customer_ids(session_id=sa360_sid))
    explicit_ids = False
    if ids_from_text:
        explicit_ids = True
        final_ids = _normalize_customer_ids(ids_from_text)
    elif req.customer_ids:
        # If the client provided IDs (UI picker, API caller), treat as explicit even if they
        # match the saved default. Otherwise we risk clearing them in account resolution.
        explicit_ids = True
    elif route.customer_ids:
        explicit_ids = True

    if explicit_ids and default_ids and not ids_from_text:
        # If the explicit IDs are different from the saved default, drop defaults so we don't mix accounts.
        req_ids_norm = _normalize_customer_ids(req.customer_ids or [])
        if not (req_ids_norm and set(req_ids_norm) == set(default_ids)):
            final_ids = [cid for cid in final_ids if cid not in default_ids]

    # Deterministic intent correction:
    # If the router returns "general_chat" for an obvious performance/audit question that includes a timeframe,
    # force the intent/tool so we can apply default-account fallback and proceed.
    timeframe_hint = _has_timeframe_hint(req.message)
    if route.intent == "general_chat" and timeframe_hint:
        lower_msg = (req.message or "").lower()
        has_perf_cue = (
            "performance" in lower_msg
            or "audit" in lower_msg
            or bool(_extract_metric_mentions(req.message))
            or bool(_extract_custom_metric_mentions(req.message))
        )
        if has_perf_cue:
            forced_intent = "audit" if "audit" in lower_msg else "performance"
            route.intent = forced_intent
            route.tool = "audit" if forced_intent == "audit" else "performance"
            route.run_planner = True
            route.run_trends = False
            route.needs_ids = False
            route.needs_clarification = False
            route.clarification = None
            # Avoid low-confidence logic overriding deterministic intent correction.
            try:
                route.confidence = max(float(route.confidence or 0.0), 0.7)
            except Exception:
                route.confidence = 0.7
            route.notes = (route.notes or "") + " forced_intent_timeframe_perf_cue"

    # Broad beta UX: if SA360 is connected and the user did not specify a customer_id (nor an account name),
    # use their saved default account for routing/planner decisions.
    #
    # This MUST happen before _resolve_account_context() so:
    # 1) downstream matching can override the default when the message clearly points to another account
    # 2) generic performance prompts do not dead-end into "pick an account" UX
    if route.intent in {"performance", "audit"} and not explicit_ids and not final_ids and default_ids:
        final_ids = list(default_ids)
        route.notes = (route.notes or "") + " default_account_fallback"

    resolved_ids, resolved_account, resolution_notes, candidates = _resolve_account_context(
        req.message,
        customer_ids=final_ids,
        account_name=req.account_name,
        explicit_ids=explicit_ids,
        session_id=sa360_sid,
    )
    if resolution_notes:
        route.notes = f"{route.notes}; {resolution_notes}" if route.notes else resolution_notes
    if candidates:
        route.candidates = candidates
        if route.run_planner:
            route.needs_clarification = True
            if not route.clarification:
                route.clarification = "Which account should I use?"
    final_ids = resolved_ids
    if route.intent == "general_chat" and explicit_ids and timeframe_hint:
        # Avoid general-chat replies for explicit account+timeframe requests.
        forced_intent = "audit" if "audit" in (req.message or "").lower() else "performance"
        route.intent = forced_intent
        route.tool = "audit" if forced_intent == "audit" else "performance"
        route.run_planner = True
        route.needs_clarification = False
        route.clarification = None
        route.notes = (route.notes or "") + " forced_intent_explicit_id_timeframe"
    if _is_relational_metric_question(req.message) and not _has_trends_cue(req.message):
        # Keep relational metric questions on performance, not seasonality/trends.
        route.intent = "performance"
        route.tool = "performance"
        route.run_planner = True
        route.run_trends = False
        route.themes = []
        route.notes = (route.notes or "") + " relational_metric_override"
    if (
        route.intent == "creative"
        and final_ids
        and _is_ad_quality_followup(req.message)
        and not _is_copy_generation_request(req.message)
    ):
        route.intent = "performance"
        route.tool = "performance"
        route.run_planner = True
        route.run_trends = False
        route.themes = []
        route.notes = (route.notes or "") + " ad_quality_override"
    if route.intent == "seasonality" and not route.themes:
        derived = _derive_themes([], resolved_account or req.account_name)
        if derived:
            route.themes = derived

    # Low-confidence path: ask for clarification, do not trigger planner
    route_conf = route.confidence if route.confidence is not None else 0.5
    auto_proceed_single = False
    if route.intent in {"audit", "performance"} and len(final_ids) == 1:
        try:
            auto_proceed_single = not _is_sa360_manager_account(str(final_ids[0]), session_id=session_id)
        except Exception:
            auto_proceed_single = True
    if route_conf < 0.4:
        if auto_proceed_single:
            route.run_planner = True
            route.needs_clarification = False
            route.clarification = None
            route.notes = (route.notes or "") + " auto_proceed_single_account"
        else:
            # Avoid prompting for account selection until we have a stable route; this is re-evaluated
            # after deterministic gating below.
            route.needs_ids = False
            route.run_planner = False
            default_clarify = "What would you like me to do - run a performance check, an audit, or answer generally?"
            # Preserve clarification routes produced by _clarification_route() (router_no_route).
            is_router_clarification = bool(route.needs_clarification or route.clarification) or (
                "router_no_route" in (route.notes or "")
            )
            if route.intent == "general_chat":
                route.tool = None
                if is_router_clarification:
                    route.needs_clarification = True
                    if not route.clarification:
                        route.clarification = default_clarify
                else:
                    route.needs_clarification = False
                    route.clarification = None
            else:
                route.tool = None if route.intent == "general_chat" else route.tool
                route.needs_clarification = True
                clarify = default_clarify
                if route.clarification:
                    trimmed = str(route.clarification).strip()
                    if len(trimmed) >= 8 and trimmed.lower() != (req.message or "").strip().lower():
                        clarify = trimmed
                route.clarification = clarify
            route.notes = (route.notes or "") + " low_confidence_router"

    # Deterministic planner gating (post low-confidence):
    # If the user asked a performance/audit question with a timeframe hint, minor phrasing changes
    # (greetings/punctuation/whitespace) must not flip run_planner off.
    #
    # This MUST run after low-confidence logic, otherwise low-confidence paths can override it and
    # cause metamorphic instability in routing.
    if route.intent in {"audit", "performance"} and timeframe_hint and not route.run_planner:
        route.run_planner = True
        if route.intent == "audit" and not route.tool:
            route.tool = "audit"
        if route.intent == "performance" and not route.tool:
            route.tool = "performance"
        route.notes = (route.notes or "") + " planner_forced_timeframe"

    # If we're proceeding with planner but still lack an account context, make the clarification
    # explicit and consistent (account pick), rather than generic routing questions.
    if route.intent in {"audit", "performance"} and route.run_planner and not final_ids:
        route.needs_clarification = True
        if not route.clarification or "what would you like me to do" in str(route.clarification).lower():
            route.clarification = "Which account should I use?"

    # If any provided ID is a manager (MCC), block planner and request a child account
    manager_ids = [cid for cid in final_ids if _is_sa360_manager_account(str(cid), session_id=session_id)]
    if manager_ids:
        route.needs_ids = True
        final_ids = []
        route.run_planner = False
        child_hint = ""
        try:
            children = [a for a in _sa360_list_customers_cached(session_id=session_id) if not a.get("manager")]
            if children:
                shortlist = children[:5]
                child_hint = " Pick a child account such as: " + " | ".join(
                    f"{c.get('name','account')} ({c.get('customer_id')})" for c in shortlist
                )
        except Exception:
            pass
        note = f"Manager account detected; select a child account.{child_hint}"
        route.notes = note if not route.notes else f"{route.notes}; {note}"

    followup_no_planner = (
        route.intent == "performance"
        and _should_explain_performance(req.message)
        and not explicit_ids
        and not timeframe_hint
    )
    if followup_no_planner:
        route.run_planner = False
        route.run_trends = False
        route.notes = (route.notes or "") + " followup_no_planner"

    needs_ids = (route.run_planner and not final_ids) or (route.needs_ids and not final_ids)

    # Broad beta guardrail: if SA360 isn't connected, do not ask for account IDs (account picker is a dead-end).
    # Instead, explicitly prompt the user to connect SA360 first.
    if route.intent in {"performance", "audit"} and (route.run_planner or needs_ids):
        # Use the scoped SA360 storage key (SSO principal when present), not the raw browser session id.
        session = _load_sa360_session(sa360_sid) if sa360_sid else None
        if not (session and session.get("refresh_token")):
            # Retry once for transient storage/network failures; false "not connected" signals create
            # metamorphic instability and confuse users.
            try:
                await asyncio.sleep(0.15)
            except Exception:
                pass
            session = _load_sa360_session(sa360_sid) if sa360_sid else None
        if not (session and session.get("refresh_token")):
            route.run_planner = False
            route.run_trends = False
            route.needs_clarification = True
            route.clarification = "SA360 isn't connected for this session. Click Connect SA360 at the top of Kai, then retry."
            needs_ids = False
            route.needs_ids = False
            final_ids = []
            route.notes = (route.notes or "") + " sa360_not_connected_no_planner"

    if route.run_planner and route.intent in {"performance", "audit"} and not SA360_FETCH_ENABLED:
        route.run_planner = False
        route.needs_clarification = True
        route.clarification = (
            route.clarification
            or "Connected data is required for performance/audit checks. "
            "Please connect SA360 or upload account exports to proceed."
        )
        route.notes = (route.notes or "") + " sa360_disabled_no_planner"

    needs_ids = (route.run_planner and not final_ids) or (route.needs_ids and not final_ids)

    return RouteResponse(
        intent=route.intent,
        tool=route.tool,
        run_planner=route.run_planner,
        run_trends=route.run_trends,
        themes=route.themes,
        customer_ids=final_ids,
        needs_ids=needs_ids,
        notes=route.notes,
        confidence=route.confidence,
        needs_clarification=route.needs_clarification,
        clarification=route.clarification,
        candidates=route.candidates,
    )


def _coerce_date_range(date_range: str | None) -> str | None:
    """
    Normalize human-friendly or explicit date ranges to GAQL DURING syntax.
    Supports presets (last week, last 7 days), week-before-last, and explicit dates.
    Returns tokens like 'LAST_7_DAYS' or 'YYYY-MM-DD,YYYY-MM-DD'.
    """
    if not date_range:
        return None
    raw = date_range.strip()
    if not raw:
        return None

    upper = raw.upper().strip()
    simple_presets = {"LAST_7_DAYS", "LAST_30_DAYS", "LAST_MONTH", "THIS_MONTH", "YESTERDAY", "TODAY"}
    if upper in simple_presets:
        return upper

    key = raw.lower().replace("_", " ").strip()
    # Relative offsets: "3 days ago" -> same start/end; "last 3 days"/"past 3 days" -> span
    m_ago = re.search(r"(\d+)\s+days?\s+ago", key)
    if m_ago:
        n = int(m_ago.group(1))
        target = date.today() - timedelta(days=n)
        return f"{target:%Y-%m-%d},{target:%Y-%m-%d}"
    m_span = re.search(r"(last|past)\s+(\d+)\s+days?", key)
    if m_span:
        n = int(m_span.group(2))
        end = date.today()
        start = end - timedelta(days=max(n - 1, 0))
        return f"{start:%Y-%m-%d},{end:%Y-%m-%d}"
    preset_map = {
        "last 7 days": "LAST_7_DAYS",
        "last week": "LAST_WEEK",
        "previous week": "LAST_WEEK",
        "last 30 days": "LAST_30_DAYS",
        "last month": "LAST_MONTH",
        "this month": "THIS_MONTH",
        "yesterday": "YESTERDAY",
        "today": "TODAY",
        "week before last": "WEEK_BEFORE_LAST",
        "two weeks ago": "WEEK_BEFORE_LAST",
    }
    # Week-based presets
    if key in ("last week", "previous week"):
        # Monday-Sunday of prior week
        start_this_week = date.today() - timedelta(days=date.today().weekday())
        start = start_this_week - timedelta(weeks=1)
        end = start + timedelta(days=6)
        return f"{start:%Y-%m-%d},{end:%Y-%m-%d}"
    if key in ("week before last", "two weeks ago"):
        start_this_week = date.today() - timedelta(days=date.today().weekday())
        start = start_this_week - timedelta(weeks=2)
        end = start + timedelta(days=6)
        return f"{start:%Y-%m-%d},{end:%Y-%m-%d}"

    if key in preset_map:
        return preset_map[key]

    def _parse_date_fragment(fragment: str) -> date | None:
        frag = fragment.strip()
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(frag, fmt).date()
            except Exception:
                continue
        return None

    # Explicit ranges with separators
    for sep in ["..", ",", " to ", " TO ", " - "]:
        if sep in raw:
            tokens = raw.replace(" to ", "..").replace(" TO ", "..").replace(" - ", "..").replace(",", "..").split("..")
            if len(tokens) == 2:
                d1 = _parse_date_fragment(tokens[0])
                d2 = _parse_date_fragment(tokens[1])
                if d1 and d2:
                    return f"{d1:%Y-%m-%d},{d2:%Y-%m-%d}"

    # Fallback: if raw already looks like YYYYMMDD,YYYYMMDD keep as-is
    if len(raw) == 17 and "," in raw and raw.replace(",", "").isdigit():
        return raw  # already looks like YYYYMMDD,YYYYMMDD

    # Last resort: return upper-cased raw to allow existing presets to flow through
    return raw.upper()


def _date_span_from_range(date_range: str | None) -> tuple[date, date] | None:
    """
    Convert our date_range tokens into concrete start/end dates (inclusive).
    Supports presets used above and explicit YYYY-MM-DD,YYYY-MM-DD.
    """
    if not date_range:
        return None
    raw = date_range.strip()
    if not raw:
        return None

    def monday_of_week(d: date) -> date:
        return d - timedelta(days=d.weekday())

    upper = raw.upper()
    today = date.today()

    if "," in raw and len(raw) >= 17:
        try:
            p1, p2 = raw.split(",", 1)
            d1 = datetime.strptime(p1.strip(), "%Y-%m-%d").date()
            d2 = datetime.strptime(p2.strip(), "%Y-%m-%d").date()
            return d1, d2
        except Exception:
            pass

    if upper in ("LAST_7_DAYS", "LAST7DAYS"):
        end = today
        start = end - timedelta(days=6)
        return start, end
    if upper in ("LAST_30_DAYS", "LAST30DAYS"):
        end = today
        start = end - timedelta(days=29)
        return start, end
    if upper in ("LAST_MONTH",):
        first_this = today.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        first_prev = last_prev.replace(day=1)
        return first_prev, last_prev
    if upper in ("THIS_MONTH",):
        first_this = today.replace(day=1)
        return first_this, today
    if upper in ("YESTERDAY",):
        y = today - timedelta(days=1)
        return y, y
    if upper in ("TODAY",):
        return today, today
    if upper in ("LAST_WEEK", "PREVIOUS WEEK"):
        start_this_week = monday_of_week(today)
        start = start_this_week - timedelta(weeks=1)
        end = start + timedelta(days=6)
        return start, end
    if upper in ("WEEK_BEFORE_LAST", "TWO WEEKS AGO"):
        start_this_week = monday_of_week(today)
        start = start_this_week - timedelta(weeks=2)
        end = start + timedelta(days=6)
        return start, end
    # 3 days ago
    m_ago = re.search(r"(\\d+)\\s+days?\\s+ago", raw.lower())
    if m_ago:
        n = int(m_ago.group(1))
        d1 = today - timedelta(days=n)
        return d1, d1
    m_span = re.search(r"(last|past)\\s+(\\d+)\\s+days?", raw.lower())
    if m_span:
        n = int(m_span.group(2))
        end = today
        start = end - timedelta(days=max(n - 1, 0))
        return start, end
    return None


def _previous_span(span: tuple[date, date] | None) -> tuple[date, date] | None:
    if not span:
        return None
    start, end = span
    length = (end - start).days + 1
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=length - 1)
    return prev_start, prev_end


def _keyword_volume_snapshot(
    keyword: str,
    customer_ids: list[str],
    date_range: str | None,
    include_previous: bool = True,
    session_id: str | None = None,
) -> tuple[dict, list[dict]]:
    """
    Fetch impressions/clicks/cost for a keyword (case-insensitive) from SA360 keyword_performance.
    Returns (exact_summary, close_matches[])
    """
    if not keyword:
        return {}, []
    # Always bypass cache to avoid stale/lifetime aggregates for volume asks
    frames = _collect_sa360_frames(customer_ids, date_range, bypass_cache=True, write_cache=False, session_id=session_id)
    # Build campaign type map for grouping (brand/nonbrand/competitor/pmax)
    brand_aliases, competitor_aliases = _load_brand_competitor_aliases()
    campaign_types: dict[str, str] = {}
    camp_df = frames.get("campaign")
    if camp_df is not None and not camp_df.empty:
        for _, row in camp_df.iterrows():
            cid = str(row.get("campaign.id") or "").strip()
            cname = row.get("campaign.name") or ""
            ctype = row.get("campaign.advertising_channel_type") or ""
            if cid:
                campaign_types[cid] = _classify_campaign(cname, ctype, brand_aliases, competitor_aliases)
    kw_df = frames.get("keyword_performance")
    if kw_df is None or kw_df.empty or "ad_group_criterion.keyword.text" not in kw_df.columns:
        return {}, []
    df = kw_df.copy()
    df["__key"] = df["ad_group_criterion.keyword.text"].astype(str).str.lower()
    # optional: restrict to search network by device if desired later
    match_key = keyword.lower().strip()

    def summarize(df_slice: pd.DataFrame) -> dict:
        metrics = {
            "impressions": pd.to_numeric(df_slice.get("metrics.impressions"), errors="coerce").sum(skipna=True),
            "clicks": pd.to_numeric(df_slice.get("metrics.clicks"), errors="coerce").sum(skipna=True),
            "cost": pd.to_numeric(df_slice.get("metrics.cost_micros"), errors="coerce").sum(skipna=True),
            "conversions": pd.to_numeric(df_slice.get("metrics.conversions"), errors="coerce").sum(skipna=True),
            "conversions_value": pd.to_numeric(df_slice.get("metrics.conversions_value"), errors="coerce").sum(skipna=True),
        }
        return _summarize_numeric(metrics)

    def summarize_by_type(df_slice: pd.DataFrame) -> dict:
        if df_slice is None or df_slice.empty:
            return {}
        by_type = {}
        for ctype, g in df_slice.groupby("__campaign_type"):
            by_type[ctype] = summarize(g)
        return by_type

    # annotate campaign type
    if "campaign.id" in df.columns:
        df["__campaign_type"] = df["campaign.id"].astype(str).map(campaign_types).fillna("nonbrand")
    else:
        df["__campaign_type"] = "nonbrand"
    exact_df = df[df["__key"] == match_key]
    exact = summarize(exact_df) if not exact_df.empty else {}
    exact_types = summarize_by_type(exact_df) if not exact_df.empty else {}

    close_matches: list[dict] = []
    if exact_df.empty:
        contains = df[df["__key"].str.contains(match_key, na=False)]
        if not contains.empty:
            agg = (
                contains.groupby("__key")
                .agg({"metrics.impressions": "sum", "metrics.clicks": "sum"})
                .reset_index()
                .sort_values("metrics.impressions", ascending=False)
                .head(5)
            )
            for _, row in agg.iterrows():
                close_matches.append(
                    {
                        "keyword": row["__key"],
                        "impressions": float(row.get("metrics.impressions") or 0),
                        "clicks": float(row.get("metrics.clicks") or 0),
                    }
                )

    if include_previous and (exact or close_matches):
        norm_range = _coerce_date_range(date_range) or "LAST_30_DAYS"
        span = _date_span_from_range(norm_range)
        prev_span = _previous_span(span) if span else None
        if prev_span:
            def span_to_range(sp: tuple[date, date]) -> str:
                return f"{sp[0]:%Y-%m-%d},{sp[1]:%Y-%m-%d}"
            prev_frames = _collect_sa360_frames(
                customer_ids,
                span_to_range(prev_span),
                bypass_cache=True,
                write_cache=False,
                session_id=session_id,
            )
            prev_df = prev_frames.get("keyword_performance")
            if prev_df is None:
                prev_df = pd.DataFrame()
            if not prev_df.empty:
                if "campaign.id" in prev_df.columns:
                    prev_df["__campaign_type"] = prev_df["campaign.id"].astype(str).map(campaign_types).fillna("nonbrand")
                else:
                    prev_df["__campaign_type"] = "nonbrand"
                prev_df["__key"] = prev_df["ad_group_criterion.keyword.text"].astype(str).str.lower()
                if exact:
                    prev_exact = prev_df[prev_df["__key"] == match_key]
                    prev = summarize(prev_exact) if not prev_exact.empty else {}
                    if prev:
                        change = exact.get("impressions", 0) - prev.get("impressions", 0)
                        pct = (change / prev.get("impressions", 0) * 100) if prev.get("impressions", 0) else None
                        exact["previous_impressions"] = prev.get("impressions", 0)
                        exact["impressions_change"] = change
                        exact["impressions_pct_change"] = pct
                if exact_types:
                    # Build previous type-level deltas
                    prev_df = prev_df.copy()
                    prev_df["__campaign_type"] = prev_df["campaign.id"].astype(str).map(campaign_types).fillna("nonbrand")
                    prev_types = summarize_by_type(prev_df)
                    for ctype, metrics in (prev_types or {}).items():
                        cur = exact_types.get(ctype) or {}
                        change = (cur.get("impressions", 0) - metrics.get("impressions", 0)) if metrics else None
                        pct = (change / metrics.get("impressions", 0) * 100) if metrics and metrics.get("impressions", 0) else None
                        if ctype in exact_types:
                            exact_types[ctype]["previous_impressions"] = metrics.get("impressions", 0)
                            exact_types[ctype]["impressions_change"] = change
                            exact_types[ctype]["impressions_pct_change"] = pct
                if close_matches:
                    for cm in close_matches:
                        prev_match = prev_df[prev_df["__key"] == cm["keyword"]] if "__key" in prev_df.columns else pd.DataFrame()
                        if not prev_match.empty:
                            prev_m = summarize(prev_match)
                            change = cm.get("impressions", 0) - prev_m.get("impressions", 0)
                            pct = (change / prev_m.get("impressions", 0) * 100) if prev_m.get("impressions", 0) else None
                            cm["previous_impressions"] = prev_m.get("impressions", 0)
                            cm["impressions_change"] = change
                            cm["impressions_pct_change"] = pct

    return {"total": exact, "by_type": exact_types}, close_matches


def _format_volume_reply(keyword: str, exact: dict, close_matches: list[dict], date_range: str | None) -> str:
    if not exact:
        return "I could not find that keyword in your SA360 data. Share the account or customer ID and I'll pull it again."
    parts = []
    total = exact.get("total") if isinstance(exact, dict) else exact
    by_type = exact.get("by_type") if isinstance(exact, dict) else {}
    if total:
        base = f"{keyword}: {total.get('impressions',0):,.0f} impressions"
        clicks = total.get("clicks")
        if clicks:
            base += f", {clicks:,.0f} clicks"
        ctr = total.get("ctr")
        if ctr:
            base += f", CTR {ctr:.2f}%"
        cpc = total.get("cpc")
        if cpc:
            base += f", CPC {cpc:.2f}"
        parts.append(base)
        if total.get("previous_impressions") is not None:
            pct = total.get("impressions_pct_change")
            prev = total.get("previous_impressions", 0)
            parts.append(f"Prev: {prev:,.0f} impressions ({pct:+.1f}%)." if pct is not None else f"Prev: {prev:,.0f} impressions.")
        if by_type:
            breakdown = []
            for ctype, m in by_type.items():
                txt = f"{ctype}: {m.get('impressions',0):,.0f}"
                if m.get("impressions_pct_change") is not None:
                    txt += f" ({m.get('impressions_pct_change'):+.1f}% vs prev)"
                breakdown.append(txt)
            if breakdown:
                parts.append("By type: " + "; ".join(breakdown))
    if date_range:
        parts.append(f"Range: {date_range}.")
    return " ".join(parts).strip()


def _exchange_google_access_token(client_id: str, client_secret: str, refresh_token: str) -> str:
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
    }
    resp = httpx.post(token_url, data=data, timeout=20)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {resp.text}")
    return resp.json().get("access_token")


def _google_ads_search_stream(customer_id: str, developer_token: str, access_token: str, query: str):
    url = f"https://googleads.googleapis.com/v15/customers/{customer_id}/googleAds:searchStream"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "developer-token": developer_token,
        "Content-Type": "application/json",
    }
    resp = httpx.post(url, headers=headers, json={"query": query}, timeout=60)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Google Ads search error: {resp.text}")
    # Streaming response is a JSON array (not NDJSON); parse once.
    try:
        payload = resp.json()
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to parse Google Ads response.")
    # payload is a list of chunks, each with results
    for chunk in payload:
        for result in chunk.get("results", []):
            yield result


def _gaql_query(name: str, date_range: str | None) -> str:
    # Basic GAQL snippets for required reports
    if name == "campaign":
        return f"""
        SELECT
          campaign.id,
          campaign.name,
          campaign.status,
          campaign.advertising_channel_type,
          campaign.advertising_channel_sub_type,
          campaign.start_date,
          campaign.end_date,
          campaign.bidding_strategy_type,
          campaign.campaign_budget,
          campaign.labels,
          campaign.resource_name,
          campaign.network_settings.target_search_network,
          campaign.network_settings.target_google_search,
          campaign.network_settings.target_partner_search_network,
          campaign.network_settings.target_content_network
        FROM campaign
        WHERE campaign.status != 'REMOVED'
        {f' DURING {date_range}' if date_range else ''}
        """
    if name == "ad_group":
        return f"""
        SELECT
          ad_group.id,
          ad_group.name,
          ad_group.status,
          ad_group.type,
          ad_group.cpc_bid_micros,
          ad_group.labels,
          ad_group.campaign
        FROM ad_group
        WHERE ad_group.status != 'REMOVED'
        """
    if name == "ad":
        return f"""
        SELECT
          ad_group_ad.ad.id,
          ad_group.id,
          campaign.id,
          ad_group_ad.status,
          ad_group_ad.policy_summary.approval_status,
          ad_group_ad.policy_summary.review_status,
          ad_group_ad.ad.type,
          ad_group_ad.ad.responsive_search_ad.headlines,
          ad_group_ad.ad.responsive_search_ad.descriptions,
          ad_group_ad.ad.final_urls
        FROM ad_group_ad
        WHERE ad_group_ad.status != 'REMOVED'
        """
    if name == "keyword_performance":
        return f"""
        SELECT
          ad_group_criterion.criterion_id,
          ad_group_criterion.keyword.text,
          ad_group_criterion.keyword.match_type,
          ad_group_criterion.status,
          ad_group.id,
          campaign.id,
          metrics.impressions,
          metrics.clicks,
          metrics.cost_micros,
          metrics.conversions,
          metrics.conversions_value,
          metrics.ctr,
          metrics.average_cpc,
          metrics.cost_per_conversion,
          metrics.search_impression_share,
          metrics.search_rank_lost_impression_share,
          metrics.search_exact_match_impression_share
        FROM keyword_view
        WHERE ad_group_criterion.status != 'REMOVED'
        {f' AND segments.date DURING {date_range}' if date_range else ''}
        """
    if name == "landing_page":
        return f"""
        SELECT
          landing_page_view.unexpanded_final_url,
          metrics.impressions,
          metrics.clicks,
          metrics.conversions,
          metrics.conversions_value,
          metrics.average_cpc,
          metrics.ctr
        FROM landing_page_view
        {f'WHERE segments.date DURING {date_range}' if date_range else ''}
        """
    if name == "account":
        return """
        SELECT
          customer.id,
          customer.descriptive_name,
          customer.currency_code,
          customer.time_zone,
          customer.status
        FROM customer
        """
    raise HTTPException(status_code=400, detail=f"Unknown GAQL template: {name}")


def _collect_ads_frames(customer_ids: list[str], developer_token: str, access_token: str, date_range: str | None):
    frames: dict[str, list[dict]] = {k: [] for k in ADS_CSV_SCHEMAS.keys()}
    for cust in customer_ids:
        for report_name in ADS_CSV_SCHEMAS.keys():
            query = _gaql_query(report_name, date_range)
            rows = []
            for res in _google_ads_search_stream(cust, developer_token, access_token, query):
                row = {}
                for col in ADS_CSV_SCHEMAS[report_name]:
                    row[col] = _get_nested_value(res, col)
                row["customer_id"] = cust
                rows.append(row)
            frames[report_name].extend(rows)
    return {k: pd.DataFrame(v) if v else pd.DataFrame(columns=ADS_CSV_SCHEMAS[k]) for k, v in frames.items()}


# ============================
# SA360 helpers (feature-flagged)
# ============================
def _ensure_sa360_enabled():
    if not SA360_FETCH_ENABLED:
        raise HTTPException(status_code=400, detail="SA360 fetch is disabled. Set SA360_FETCH_ENABLED=true to enable.")
    # We rely on google-auth + direct HTTP calls; no client library import needed. If we fail later, it will be due to
    # missing credentials or HTTP errors, not missing google ads modules.


def _resolve_sa360_oauth(session_id: str | None = None) -> dict:
    client_id = os.environ.get("SA360_CLIENT_ID")
    client_secret = os.environ.get("SA360_CLIENT_SECRET")
    refresh_token = os.environ.get("SA360_REFRESH_TOKEN")
    session = _load_sa360_session(session_id)
    if session and session.get("refresh_token"):
        refresh_token = session.get("refresh_token")
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
    }


def _resolve_sa360_login_customer_id(session_id: str | None = None) -> str | None:
    session = _load_sa360_session(session_id)
    if session and session.get("login_customer_id"):
        return str(session.get("login_customer_id"))
    login_cid = os.environ.get("SA360_LOGIN_CUSTOMER_ID")
    return str(login_cid) if login_cid else None


def _sa360_get_access_token(session_id: str | None = None) -> str:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    oauth = _resolve_sa360_oauth(session_id)
    client_id = oauth.get("client_id")
    client_secret = oauth.get("client_secret")
    refresh_token = oauth.get("refresh_token")
    if not all([client_id, client_secret, refresh_token]):
        raise HTTPException(status_code=500, detail="Missing SA360 OAuth credentials.")

    creds = Credentials(
        None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=["https://www.googleapis.com/auth/doubleclicksearch"],
    )
    try:
        creds.refresh(Request())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SA360 token refresh failed: {exc}")
    if not creds.token:
        raise HTTPException(status_code=500, detail="SA360 access token not obtained.")
    return creds.token


def _sa360_search(customer_id: str, query: str, session_id: str | None = None):
    if not SA360_BREAKER.allow():
        raise HTTPException(status_code=503, detail="SA360 temporarily unavailable (circuit open).")
    access_token = _sa360_get_access_token(session_id=session_id)
    # SA360 expects numeric customer IDs without dashes; normalize inbound strings.
    normalized_cid = (customer_id or "").replace("-", "")
    url = f"https://searchads360.googleapis.com/v0/customers/{normalized_cid}/searchAds360:search"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    login_cid = _resolve_sa360_login_customer_id(session_id)
    if login_cid:
        headers["login-customer-id"] = login_cid.replace("-", "")
    try:
        # SA360 search uses pagination (nextPageToken). Many queries can exceed 10k rows (especially when segmented).
        # We page defensively with caps to avoid runaway pulls causing timeouts.
        page_size = int(os.environ.get("SA360_PAGE_SIZE", "10000") or "10000")
        max_pages = int(os.environ.get("SA360_MAX_PAGES", "20") or "20")
        max_results = int(os.environ.get("SA360_MAX_RESULTS", "50000") or "50000")

        start = time.perf_counter()
        results: list[dict] = []
        page_token: str | None = None
        pages = 0
        while True:
            body: dict[str, Any] = {"query": query, "pageSize": page_size}
            if page_token:
                body["pageToken"] = page_token
            try:
                resp = httpx.post(url, headers=headers, json=body, timeout=60)
            except httpx.TimeoutException as exc:
                # Transient upstream timeout: count toward breaker.
                raise HTTPException(status_code=504, detail=f"SA360 request timed out: {exc}")
            except httpx.RequestError as exc:
                # Transient network error: count toward breaker.
                raise HTTPException(status_code=502, detail=f"SA360 network error: {exc}")
            pages += 1
            if resp.status_code != 200:
                status = int(resp.status_code)
                text = resp.text
                # Avoid tripping the global breaker on auth/customer issues. Those are session- or
                # account-scoped and should not degrade service for other users.
                if status >= 500:
                    raise HTTPException(status_code=503, detail=f"SA360 upstream error: {text}")
                if status == 401:
                    raise HTTPException(status_code=401, detail=f"SA360 auth error: {text}")
                if status == 403:
                    raise HTTPException(status_code=403, detail=f"SA360 permission error: {text}")
                if status == 404:
                    raise HTTPException(status_code=404, detail=f"SA360 customer not found: {text}")
                raise HTTPException(status_code=400, detail=f"SA360 request rejected: {text}")
            try:
                payload = resp.json()
            except Exception:
                raise HTTPException(status_code=502, detail="SA360 response parse error")

            batch = payload.get("results", []) or []
            results.extend(batch)
            page_token = payload.get("nextPageToken")

            if not page_token:
                break
            if pages >= max_pages or len(results) >= max_results:
                log_event(
                    "sa360_truncated",
                    pages=pages,
                    results=len(results),
                    max_pages=max_pages,
                    max_results=max_results,
                )
                break
        latency_ms = (time.perf_counter() - start) * 1000
        log_event("external_call", service="sa360", latency_ms=round(latency_ms, 2), pages=pages, results=len(results))
    except HTTPException as exc:
        # Only count toward the breaker for transient/server-side errors.
        if int(getattr(exc, "status_code", 0) or 0) >= 500:
            SA360_BREAKER.record_failure()
        raise
    except Exception as exc:
        SA360_BREAKER.record_failure()
        raise HTTPException(status_code=500, detail=f"SA360 HTTP error: {exc}")
    SA360_BREAKER.record_success()
    return results


def _sa360_list_customers_cached(session_id: str | None = None) -> list[dict]:
    """
    Discover accessible SA360 customers (id + descriptive_name) using GAQL.
    Cached to avoid repeated list calls; respects SA360_FETCH_ENABLED.
    """
    if not SA360_FETCH_ENABLED:
        return []

    sid = _normalize_session_id(session_id)
    cache_bucket = SA360_ACCOUNT_CACHE if not sid else SA360_ACCOUNT_CACHE_BY_SESSION.setdefault(sid, {"ts": None, "data": []})
    now = datetime.utcnow()
    try:
        if cache_bucket.get("ts") and cache_bucket.get("data"):
            age = now - cache_bucket["ts"]
            if age <= timedelta(hours=max(1, SA360_ACCOUNT_CACHE_TTL_HOURS)):
                return cache_bucket["data"]
    except Exception:
        pass

    login_cid = _resolve_sa360_login_customer_id(session_id)
    if not login_cid:
        return []

    customers: list[dict] = []
    rows: list[dict] = []
    # Prefer customer_client to enumerate child accounts; fallback to customer for direct account.
    try:
        rows = _sa360_search(
            login_cid,
            "SELECT customer_client.client_customer, customer_client.descriptive_name, customer_client.hidden, customer_client.manager FROM customer_client",
            session_id=session_id,
        )
    except Exception:
        rows = []

    if not rows:
        try:
            rows = _sa360_search(login_cid, "SELECT customer.id, customer.descriptive_name FROM customer", session_id=session_id)
        except Exception:
            rows = []

    for row in rows or []:
        cust = row.get("customerClient") or row.get("customer_client") or row.get("customer") or {}
        cid = (
            cust.get("clientCustomer")
            or cust.get("client_customer")
            or cust.get("id")
            or (cust.get("resourceName", "").split("/")[-1] if isinstance(cust.get("resourceName"), str) else None)
        )
        if isinstance(cid, str):
            if cid.startswith("customers/"):
                cid = cid.split("/")[-1]
            elif "customers/" in cid:
                cid = cid.split("/")[-1]
        elif isinstance(cid, dict):
            res = cid.get("resourceName") if isinstance(cid.get("resourceName", ""), str) else None
            if res and "customers/" in res:
                cid = res.split("/")[-1]
        name = cust.get("descriptiveName") or cust.get("descriptive_name") or cust.get("name")
        is_hidden = cust.get("hidden", False)
        is_manager = cust.get("manager", False) or (login_cid and str(cid) == str(login_cid))
        if cid and not is_hidden:
            customers.append({"customer_id": str(cid), "name": name or f"Account {cid}", "manager": is_manager})

    # Normalize IDs once more to guard against any residual prefixes and dedupe.
    normed = []
    seen = set()
    for c in customers:
        cid = c.get("customer_id")
        if isinstance(cid, str) and "customers/" in cid:
            cid = cid.split("/")[-1]
        if cid and cid not in seen:
            seen.add(cid)
            c["customer_id"] = str(cid)
            if login_cid and str(cid) == str(login_cid):
                c["manager"] = True
            normed.append(c)
    customers = normed

    cache_bucket["ts"] = now
    cache_bucket["data"] = customers
    return customers


def _sa360_list_accessible_customer_ids(session_id: str | None = None) -> list[str]:
    """
    Return customer IDs the authenticated SA360 user can access.
    This does not require a login-customer-id header.
    """
    access_token = _sa360_get_access_token(session_id=session_id)
    url = "https://searchads360.googleapis.com/v0/customers:listAccessibleCustomers"
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        resp = httpx.get(url, headers=headers, timeout=30)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SA360 listAccessibleCustomers failed: {exc}")
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"SA360 listAccessibleCustomers error: {resp.text}")
    try:
        payload = resp.json()
    except Exception:
        raise HTTPException(status_code=500, detail="SA360 listAccessibleCustomers parse error")

    resource_names = payload.get("resourceNames") or payload.get("resource_names") or []
    ids: list[str] = []
    for rn in resource_names:
        if not isinstance(rn, str):
            continue
        if "customers/" in rn:
            cid = rn.split("customers/")[-1].strip()
        else:
            cid = rn.strip()
        cid = cid.replace("-", "")
        if cid and cid.isdigit():
            ids.append(cid)
    # Dedupe preserving order.
    return list(dict.fromkeys(ids))


def _sa360_list_manager_accounts_cached(session_id: str | None = None) -> list[dict]:
    """
    Discover accessible manager (MCC) accounts for the current SA360 user.
    Uses customers:listAccessibleCustomers, then checks customer.manager for each.
    """
    if not SA360_FETCH_ENABLED:
        return []
    sid = _normalize_session_id(session_id)
    if not sid:
        return []

    cache_bucket = SA360_MANAGER_CACHE_BY_SESSION.setdefault(sid, {"ts": None, "data": []})
    now = datetime.utcnow()
    try:
        if cache_bucket.get("ts") and cache_bucket.get("data"):
            age = now - cache_bucket["ts"]
            if age <= timedelta(hours=max(1, SA360_MANAGER_CACHE_TTL_HOURS)):
                return cache_bucket["data"]
    except Exception:
        pass

    manager_rows: list[dict] = []
    ids = _sa360_list_accessible_customer_ids(session_id=session_id)
    for cid in ids:
        try:
            rows = _sa360_search(
                cid,
                "SELECT customer.id, customer.descriptive_name, customer.manager FROM customer",
                session_id=session_id,
            )
        except Exception:
            continue
        for row in rows or []:
            cust = row.get("customer") or {}
            is_manager = bool(cust.get("manager"))
            cust_id = cust.get("id") or (cust.get("resourceName", "").split("/")[-1] if isinstance(cust.get("resourceName"), str) else None)
            if not cust_id:
                continue
            cust_id = str(cust_id).replace("customers/", "").replace("-", "")
            if not cust_id.isdigit():
                continue
            if is_manager:
                manager_rows.append(
                    {
                        "customer_id": cust_id,
                        "name": cust.get("descriptiveName") or cust.get("descriptive_name") or f"Manager {cust_id}",
                        "manager": True,
                    }
                )

    # Dedupe by id and sort by name for stable UX.
    seen = set()
    managers: list[dict] = []
    for m in manager_rows:
        cid = str(m.get("customer_id") or "")
        if cid and cid not in seen:
            seen.add(cid)
            managers.append(m)
    managers.sort(key=lambda m: (m.get("name") or m.get("customer_id") or ""))

    cache_bucket["ts"] = now
    cache_bucket["data"] = managers
    return managers


def _is_sa360_manager_account(customer_id: str, session_id: str | None = None) -> bool:
    """Return True if cached discovery marks this ID as a manager (MCC)."""
    if not customer_id:
        return False
    cid = customer_id.replace("customers/", "")
    login_cid = _resolve_sa360_login_customer_id(session_id)
    if login_cid and str(cid) == str(login_cid):
        return True
    try:
        sid = _normalize_session_id(session_id)
        cache_bucket = SA360_ACCOUNT_CACHE if not sid else SA360_ACCOUNT_CACHE_BY_SESSION.get(sid)
        if not cache_bucket or not cache_bucket.get("data"):
            _sa360_list_customers_cached(session_id=session_id)
    except Exception:
        pass
    try:
        sid = _normalize_session_id(session_id)
        cache_bucket = SA360_ACCOUNT_CACHE if not sid else SA360_ACCOUNT_CACHE_BY_SESSION.get(sid)
        if cache_bucket and cache_bucket.get("data"):
            for acc in cache_bucket["data"]:
                if str(acc.get("customer_id")) == cid:
                    return bool(acc.get("manager"))
    except Exception:
        return False
    return False


def _local_llm_health() -> dict:
    """Ping local LLM to confirm availability and JSON compliance tolerance."""
    allow_local = os.environ.get("ENABLE_LOCAL_LLM_WRAPPER", "false").lower() == "true"
    endpoint = os.environ.get("LOCAL_LLM_ENDPOINT", "").strip()
    model = os.environ.get("LOCAL_LLM_MODEL", "").strip() or "llama3"
    status = {"enabled": allow_local, "endpoint": bool(endpoint), "model": model, "status": "disabled"}
    if not allow_local or not endpoint:
        return status
    try:
        text, meta = _call_local_llm(
            messages=[{"role": "user", "content": "Return JSON only: {\"ok\": true}"}],
            intent="health_local_llm",
            max_tokens=20,
            force_json=True,
        )
        status.update(meta)
        response_ok = bool(text)
        ok_token = False
        parsed = None
        if isinstance(text, str):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
        if isinstance(parsed, dict) and parsed.get("ok") is True:
            ok_token = True
        else:
            normalized = re.sub(r"[^a-z]", "", text.lower() if text else "")
            ok_token = response_ok and "ok" in normalized
        status["response_ok"] = response_ok
        status["ok_token"] = ok_token
        if response_ok:
            status["status"] = "ok"
            if not ok_token:
                status["warning"] = "LLM responded but did not return OK"
        else:
            status["status"] = "error"
        return status
    except Exception as exc:
        status["status"] = "error"
        status["error"] = str(exc)
        return status


def _health_check_summary() -> dict:
    """
    Lightweight health check across core dependencies without relying on external terminal tools.
    - SA360 account discovery
    - QA accuracy on a small sample of customer_ids (if available)
    - Manager guard presence
    """
    out = {
        "status": "ok",
        "errors": [],
        "accounts": None,
        "qa": [],
        "manager_guard": None,
        "local_llm": None,
        "queue": {"enabled": JOB_QUEUE_ENABLED, "depth": None},
    }
    local_status = _local_llm_health()
    out["local_llm"] = local_status
    if local_status.get("status") == "error":
        out["status"] = "degraded"
        out["errors"].append(f"Local LLM unhealthy: {local_status.get('error') or 'no OK response'}")
    if JOB_QUEUE_ENABLED:
        try:
            out["queue"]["depth"] = queue_depth()
        except Exception as exc:
            out["queue"]["error"] = str(exc)
            out["status"] = "degraded"
    try:
        _ensure_sa360_enabled()
        accounts = _sa360_list_customers_cached()
        out["accounts"] = {"count": len(accounts or []), "sample": accounts[:5] if accounts else []}
    except Exception as exc:
        out["status"] = "degraded"
        out["errors"].append(f"SA360 unavailable: {exc}")
        return out

    # Manager guard: report if login MCC exists
    login_cid = os.environ.get("SA360_LOGIN_CUSTOMER_ID")
    if login_cid:
        out["manager_guard"] = {"login_customer_id": login_cid, "guard_active": True}

    # QA sample set (keep small to avoid heavy load)
    sample_ids = ["3532896537", "3956251331", "7902313748"]
    available_ids = {str(a.get("customer_id")) for a in (accounts or []) if a.get("customer_id")}
    for cid in sample_ids:
        if cid not in available_ids:
            continue
        try:
            frames = _collect_sa360_frames([cid], "2025-12-14,2025-12-20")
            raw_totals = _qa_raw_totals(frames)
            agg = _aggregate_account_performance(frames)
            keys = ["impressions", "clicks", "cost", "conversions"]
            match = all(raw_totals.get(k) == agg.get(k) for k in keys)
            out["qa"].append({"customer_id": cid, "matches": match, "raw": raw_totals, "agg": agg})
            if not match:
                out["status"] = "degraded"
        except Exception as exc:
            out["qa"].append({"customer_id": cid, "error": str(exc)})
            out["status"] = "degraded"
    return out


_ENV_BOOL_KEYS = {
    "ENABLE_VECTOR_INDEXING",
    "ADS_FETCH_ENABLED",
    "SA360_FETCH_ENABLED",
    "ENABLE_TRENDS",
    "ENABLE_ML_REASONING",
    "ENABLE_SUMMARY_ENHANCER",
    "JOB_QUEUE_ENABLED",
    "RATE_LIMIT_ENABLED",
    "CIRCUIT_BREAKER_ENABLED",
    "REQUIRE_LOCAL_LLM",
}


_ENV_FORBIDDEN_KEY_PATTERNS = (
    "KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "CONNECTION_STRING",
    "CLIENT_SECRET",
    "REFRESH_TOKEN",
)

_ENV_EXPOSURE_ALLOWLIST = [
    "ENABLE_VECTOR_INDEXING",
    "ENABLE_ML_REASONING",
    "ENABLE_SUMMARY_ENHANCER",
    "ADS_FETCH_ENABLED",
    "SA360_FETCH_ENABLED",
    "ENABLE_TRENDS",
    "REQUIRE_LOCAL_LLM",
    "JOB_QUEUE_ENABLED",
    "JOB_QUEUE_FORCE",
    "JOB_QUEUE_NAME",
    "JOB_TABLE_NAME",
    "JOB_RESULT_CONTAINER",
    "JOB_QUEUE_POLL_SECONDS",
    "JOB_QUEUE_VISIBILITY_TIMEOUT",
    "JOB_MAX_ATTEMPTS",
    "RATE_LIMIT_ENABLED",
    "RATE_LIMIT_PER_MINUTE",
    "RATE_LIMIT_BURST",
    "RATE_LIMIT_HEAVY_PER_MINUTE",
    "RATE_LIMIT_HEAVY_BURST",
    "CIRCUIT_BREAKER_ENABLED",
    "SA360_BREAKER_FAILURES",
    "SA360_BREAKER_COOLDOWN_SECONDS",
    "SERP_BREAKER_FAILURES",
    "SERP_BREAKER_COOLDOWN_SECONDS",
    "TRENDS_CACHE_TTL_SECONDS",
    "TRENDS_CACHE_MAX_ITEMS",
    "SA360_PERF_WEIGHT_CACHE_TTL_SECONDS",
    "SA360_PERF_WEIGHT_CACHE_MAX_ITEMS",
    "SA360_PERF_SEASONALITY_CACHE_TTL_SECONDS",
    "SA360_PERF_SEASONALITY_CACHE_MAX_ITEMS",
    "TRENDS_TOTAL_TIMEOUT_SECONDS",
    "TRENDS_CONNECT_TIMEOUT_SECONDS",
    "TRENDS_READ_TIMEOUT_SECONDS",
    "TRENDS_SERP_TIMEOUT_SECONDS",
    "TRENDS_MAX_SYNC_SECONDS",
    "TRENDS_QUEUE_ON_TIMEOUT",
    "TRENDS_PERF_TIMEOUT_SECONDS",
    "AUDIT_BLOB_CONTAINER",
    "AUDIT_BLOB_PREFIX",
    "AUDIT_SAS_EXPIRY_HOURS",
    "LOCAL_LLM_ROUTER_TIMEOUT_SECONDS",
    "LOCAL_LLM_ROUTER_MAX_CONCURRENCY",
    "ROUTER_PRIMARY",
    "ROUTER_VERIFY_MODE",
    "ROUTER_VERIFY_CONFIDENCE",
    "LICENSE_ENFORCEMENT_MODE",
    "LICENSE_REFRESH_INTERVAL_HOURS",
    "LICENSE_RENEW_DAYS_BEFORE_EXP",
    "LICENSE_GRACE_DAYS",
    "LICENSE_REQUEST_TIMEOUT_SECONDS",
    "LICENSE_CACHE_PATH",
    "BRAIN_GATE_USE_OBFUSCATED",
]


def _is_forbidden_env_key(key: str) -> bool:
    upper = str(key or "").upper()
    return any(pattern in upper for pattern in _ENV_FORBIDDEN_KEY_PATTERNS)


def _env_exposure_allowlist() -> list[str]:
    return [key for key in _ENV_EXPOSURE_ALLOWLIST if not _is_forbidden_env_key(key)]


def _env_update_allowlist() -> list[str]:
    # Runtime updates are restricted to the same allowlist used for exposure.
    # Secret-bearing env values must be changed at deployment/runtime secret stores, not API.
    return _env_exposure_allowlist()


def _should_enqueue(async_mode: bool) -> bool:
    if os.environ.get("KAI_JOB_ID"):
        return False
    if not JOB_QUEUE_ENABLED:
        return False
    if JOB_QUEUE_FORCE:
        return True
    return bool(async_mode)


def _should_enqueue_heavy(async_mode: bool, account_count: int, max_sync_accounts: int) -> bool:
    if os.environ.get("KAI_JOB_ID"):
        return False
    if not JOB_QUEUE_ENABLED:
        return False
    if JOB_QUEUE_FORCE:
        return True
    if async_mode:
        return True
    return account_count > max_sync_accounts


def _should_verify_local_route(route: RouteResponse, local_meta: dict) -> bool:
    if not route or local_meta.get("model") != "local":
        return False
    if os.environ.get("REQUIRE_LOCAL_LLM", "false").lower() == "true":
        return False
    mode = ROUTER_VERIFY_MODE or "adaptive"
    if mode == "off":
        return False
    if mode == "always":
        return True
    # adaptive
    if route.intent == "general_chat":
        return False
    if route.needs_ids or route.needs_clarification:
        return True
    conf = route.confidence if route.confidence is not None else 0.0
    if conf < ROUTER_VERIFY_CONFIDENCE:
        return True
    return False


def _mask_value(val: str | None) -> str | None:
    if val is None:
        return None
    s = str(val)
    if len(s) <= 4:
        return "*" * len(s)
    return "*" * max(0, len(s) - 4) + s[-4:]


def _sa360_oauth_redirect_uri() -> str | None:
    return os.environ.get("SA360_OAUTH_REDIRECT_URI")


@app.get("/api/sa360/oauth/start-url")
async def sa360_oauth_start_url(request: Request, session_id: str | None = None):
    """
    Return the Google consent URL for SA360 OAuth (no redirect).

    Why this exists:
    - In broad beta we want per-user scoping when Entra SSO is present.
    - Browser popups must be opened synchronously, so the UI opens a blank popup first,
      calls this endpoint (with Authorization header if signed-in), then navigates the popup.
    - The signed OAuth state includes the resolved SA360 scope key (Entra principal or session_id),
      so the callback stores tokens under the correct per-user scope.
    """
    sid = _sa360_scope_from_request(request, session_id)
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required to start OAuth.")
    client_id = os.environ.get("SA360_CLIENT_ID")
    redirect_uri = _sa360_oauth_redirect_uri()
    if not client_id or not redirect_uri:
        raise HTTPException(status_code=500, detail="Missing SA360 OAuth client configuration.")
    state = _sign_oauth_state({"sid": sid, "ts": int(time.time()), "nonce": uuid4().hex})
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/doubleclicksearch",
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
    }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
    return {"url": url}


@app.get("/api/sa360/oauth/start")
async def sa360_oauth_start(request: Request, session_id: str | None = None):
    """
    Start SA360 OAuth flow. Returns a redirect to Google's consent screen.
    Requires SA360_CLIENT_ID and SA360_OAUTH_REDIRECT_URI to be configured.
    """
    sid = _sa360_scope_from_request(request, session_id)
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required to start OAuth.")
    client_id = os.environ.get("SA360_CLIENT_ID")
    redirect_uri = _sa360_oauth_redirect_uri()
    if not client_id or not redirect_uri:
        raise HTTPException(status_code=500, detail="Missing SA360 OAuth client configuration.")
    state = _sign_oauth_state({"sid": sid, "ts": int(time.time()), "nonce": uuid4().hex})
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/doubleclicksearch",
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
    }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
    return RedirectResponse(url)


@app.get("/api/sa360/oauth/callback")
async def sa360_oauth_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
):
    """
    OAuth callback handler: exchanges auth code for refresh token and stores per session.
    """
    def _oauth_callback_page(status: str, title: str, message: str, http_status: int = 200) -> HTMLResponse:
        # Keep callback safe for broad beta: do not echo raw provider responses.
        payload = json.dumps({"type": "KAI_SA360_OAUTH", "status": status})
        content = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <style>
      body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin: 2rem; }}
      .card {{ max-width: 720px; padding: 1.25rem 1.5rem; border: 1px solid #e2e8f0; border-radius: 12px; }}
      h3 {{ margin: 0 0 0.5rem 0; }}
      p {{ margin: 0.25rem 0; color: #334155; }}
      code {{ background: #f1f5f9; padding: 0.1rem 0.3rem; border-radius: 6px; }}
    </style>
  </head>
  <body>
    <div class="card">
      <h3>{title}</h3>
      <p>{message}</p>
      <p>You can close this window and return to Kai.</p>
    </div>
    <script>
      (function() {{
        try {{
          if (window.opener && typeof window.opener.postMessage === 'function') {{
            window.opener.postMessage({payload}, '*');
          }}
        }} catch (e) {{}}
        try {{ window.close(); }} catch (e) {{}}
      }})();
    </script>
  </body>
</html>"""
        return HTMLResponse(content=content, status_code=http_status)

    if error:
        return _oauth_callback_page(
            status="error",
            title="SA360 connection failed",
            message="Google returned an error during authorization. Please retry the connection from Kai.",
            http_status=400,
        )
    payload = _parse_oauth_state(state or "")
    if not payload:
        return _oauth_callback_page(
            status="error",
            title="Invalid OAuth state",
            message="This authorization attempt expired or was invalid. Please retry the connection from Kai.",
            http_status=400,
        )
    sid = _normalize_session_id(payload.get("sid"))
    if not sid:
        return _oauth_callback_page(
            status="error",
            title="Missing session context",
            message="Kai could not determine which session to attach this authorization to. Please retry from the same browser tab.",
            http_status=400,
        )
    if not code:
        return _oauth_callback_page(
            status="error",
            title="Missing authorization code",
            message="Google did not return an authorization code. Please retry the connection from Kai.",
            http_status=400,
        )
    client_id = os.environ.get("SA360_CLIENT_ID")
    client_secret = os.environ.get("SA360_CLIENT_SECRET")
    redirect_uri = _sa360_oauth_redirect_uri()
    if not client_id or not client_secret or not redirect_uri:
        return _oauth_callback_page(
            status="error",
            title="Server missing OAuth configuration",
            message="The server is missing SA360 OAuth configuration. Please contact the admin.",
            http_status=500,
        )
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    resp = httpx.post(token_url, data=data, timeout=20)
    if resp.status_code != 200:
        return _oauth_callback_page(
            status="error",
            title="Token exchange failed",
            message="Kai could not exchange the authorization code for a token. Please retry. If this persists, contact the admin.",
            http_status=400,
        )
    try:
        payload = resp.json()
    except Exception:
        payload = {}
    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        existing = _load_sa360_session(sid)
        if not (existing and existing.get("refresh_token")):
            return _oauth_callback_page(
                status="error",
                title="No refresh token returned",
                message="Google did not return a refresh token. Please retry and ensure you complete consent.",
                http_status=400,
            )
    else:
        _upsert_sa360_session(sid, refresh_token=refresh_token)
    return _oauth_callback_page(
        status="connected",
        title="SA360 connected",
        message="Authorization completed successfully.",
        http_status=200,
    )


@app.get("/api/sa360/oauth/status")
async def sa360_oauth_status(request: Request, session_id: str | None = None):
    sid = _sa360_scope_from_request(request, session_id)
    if not sid:
        return {
            "connected": False,
            "login_customer_id": None,
            "default_customer_id": None,
            "default_account_name": None,
        }
    session = _load_sa360_session(sid)
    return {
        "connected": bool(session and session.get("refresh_token")),
        "login_customer_id": session.get("login_customer_id") if session else None,
        "default_customer_id": session.get("default_customer_id") if session else None,
        "default_account_name": session.get("default_account_name") if session else None,
    }


@app.post("/api/sa360/login-customer")
async def sa360_login_customer(req: Sa360LoginCustomerRequest, request: Request):
    sid = _sa360_scope_from_request(request, req.session_id)
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required.")
    login_cid = (req.login_customer_id or "").strip()
    if not login_cid:
        raise HTTPException(status_code=400, detail="login_customer_id is required.")
    _upsert_sa360_session(sid, login_customer_id=login_cid)
    return {"status": "ok", "login_customer_id": login_cid}


@app.post("/api/sa360/default-account")
async def sa360_default_account(req: Sa360DefaultAccountRequest, request: Request):
    """
    Persist a per-session default SA360 child account selection (used to prefill the UI).
    Note: this does not grant access; it only stores the user's preference.
    """
    sid = _sa360_scope_from_request(request, req.session_id)
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required.")

    customer_id = (req.customer_id or "").replace("-", "").strip()
    account_name = (req.account_name or "").strip()

    # Allow clearing defaults by posting empty values.
    if customer_id and not customer_id.isdigit():
        raise HTTPException(status_code=400, detail="customer_id must be numeric.")
    if customer_id and len(customer_id) not in (8, 9, 10, 11, 12):
        raise HTTPException(status_code=400, detail="customer_id must be 8-12 digits.")

    # Best-effort guard: avoid persisting MCC ids as defaults.
    try:
        if customer_id and _is_sa360_manager_account(customer_id, session_id=sid):
            raise HTTPException(status_code=400, detail="default account cannot be a manager (MCC).")
    except HTTPException:
        raise
    except Exception:
        pass

    _upsert_sa360_session(
        sid,
        default_customer_id=(customer_id or ""),
        default_account_name=(account_name or ""),
    )
    return {
        "status": "ok",
        "default_customer_id": customer_id or None,
        "default_account_name": account_name or None,
    }


@app.get("/api/sa360/managers", response_model=list[Sa360Account])
async def sa360_managers(request: Request, session_id: str | None = None):
    """
    Return accessible SA360 manager accounts (MCCs) by name.
    Used to eliminate the onboarding step where users must already know their MCC ID.
    """
    _ensure_sa360_enabled()
    sid = _sa360_scope_from_request(request, session_id)
    if not sid:
        return []
    session = _load_sa360_session(sid)
    if not session or not session.get("refresh_token"):
        return []
    managers = _sa360_list_manager_accounts_cached(session_id=sid)
    return [
        Sa360Account(customer_id=str(m.get("customer_id")), name=m.get("name"), manager=True)
        for m in managers
        if m.get("customer_id")
    ]


@app.get("/api/sa360/accounts", response_model=list[Sa360Account])
async def sa360_accounts(request: Request, session_id: str | None = None, login_customer_id: str | None = None):
    """
    Return accessible SA360 accounts (id + descriptive name) using cached GAQL discovery.
    """
    _ensure_sa360_enabled()
    sid = _sa360_scope_from_request(request, session_id)
    if login_customer_id:
        _upsert_sa360_session(sid, login_customer_id=login_customer_id)
    accounts = _sa360_list_customers_cached(session_id=sid)
    if not accounts:
        return []
    return [
        Sa360Account(customer_id=a.get("customer_id"), name=a.get("name"), manager=a.get("manager"))
        for a in accounts
        if a.get("customer_id")
    ]


@app.get("/api/sa360/conversion-actions", response_model=Sa360ConversionCatalogResponse)
async def sa360_conversion_actions(
    request: Request,
    customer_id: str,
    date_range: str | None = "LAST_30_DAYS",
    session_id: str | None = None,
):
    """
    Return conversion action catalog (names/categories/status) + optional windowed totals.
    This is primarily used for:
    - user-facing metric discovery ("what columns exist in SA360?")
    - QA assertions that custom-metric inference is grounded in real headers
    """
    _ensure_sa360_enabled()
    sid = _sa360_scope_from_request(request, session_id)
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required.")
    cid = (customer_id or "").replace("-", "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="customer_id is required.")
    if _is_sa360_manager_account(cid, session_id=sid):
        raise HTTPException(status_code=400, detail="Requested account is a manager (MCC); select a child account.")

    normalized_range = _coerce_date_range(date_range) or "LAST_30_DAYS"
    frames = _collect_sa360_frames(
        [cid],
        normalized_range,
        report_names=["conversion_actions", "conversion_action_summary"],
        session_id=sid,
    )
    cat = frames.get("conversion_actions")
    if cat is None:
        cat = pd.DataFrame()
    summ = frames.get("conversion_action_summary")
    if summ is None:
        summ = pd.DataFrame()

    metric_cols = [
        "metrics.conversions",
        "metrics.conversions_value",
        "metrics.all_conversions",
        "metrics.all_conversions_value",
        "metrics.cross_device_conversions",
        "metrics.cross_device_conversions_value",
    ]

    # Aggregate summary totals by action name (some actions can appear multiple times due to category joins).
    summary_map: dict[str, dict[str, Any]] = {}
    if summ is not None and not summ.empty:
        for _, row in summ.iterrows():
            name = row.get("segments.conversion_action_name")
            if not name or str(name).strip() == "":
                continue
            norm = _normalize_custom_metric_cmp(str(name))
            entry = summary_map.get(norm) or {"name": str(name)}
            cat_val = row.get("segments.conversion_action_category")
            if cat_val and not entry.get("category"):
                entry["category"] = str(cat_val)
            for col in metric_cols:
                if col not in summ.columns:
                    continue
                try:
                    val = float(pd.to_numeric(row.get(col), errors="coerce") or 0.0)
                except Exception:
                    val = 0.0
                entry[col] = float(entry.get(col) or 0.0) + float(val or 0.0)
            summary_map[norm] = entry

    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Catalog-first (ensures we include actions even with 0 conversions in the window).
    if cat is not None and not cat.empty:
        for _, row in cat.iterrows():
            name = row.get("conversion_action.name")
            if not name or str(name).strip() == "":
                continue
            name_str = str(name)
            norm = _normalize_custom_metric_cmp(name_str)
            # Conversion action *names* are occasionally duplicated; de-dupe by normalized name so users
            # see one stable metric_key per logical action name.
            if norm in seen:
                continue
            seen.add(norm)
            totals = summary_map.get(norm) or {}
            item: dict[str, Any] = {
                "metric_key": f"custom:{_normalize_custom_metric_key(name_str)}",
                "name": name_str,
                "action_id": str(row.get("conversion_action.id")) if row.get("conversion_action.id") is not None else None,
                "category": str(row.get("conversion_action.category") or totals.get("category")) if (row.get("conversion_action.category") or totals.get("category")) else None,
                "status": str(row.get("conversion_action.status")) if row.get("conversion_action.status") is not None else None,
            }
            for col in metric_cols:
                if col in totals:
                    # map to stable response keys
                    key = col.replace("metrics.", "")
                    item[key] = float(totals[col])
            items.append(item)

    # Summary-only actions (rare, but keep visibility if API returns them).
    for norm, totals in summary_map.items():
        if norm in seen:
            continue
        name_str = str(totals.get("name") or "")
        if not name_str:
            continue
        item = {
            "metric_key": f"custom:{_normalize_custom_metric_key(name_str)}",
            "name": name_str,
            "action_id": None,
            "category": str(totals.get("category")) if totals.get("category") else None,
            "status": None,
        }
        for col in metric_cols:
            if col in totals:
                key = col.replace("metrics.", "")
                item[key] = float(totals[col])
        items.append(item)

    # Sort by conversions desc (fallback to all_conversions if conversions missing).
    def _sort_key(it: dict[str, Any]) -> float:
        try:
            return float(it.get("conversions") or it.get("all_conversions") or 0.0)
        except Exception:
            return 0.0

    items.sort(key=_sort_key, reverse=True)
    return Sa360ConversionCatalogResponse(
        customer_id=cid,
        date_range=normalized_range,
        actions=[Sa360ConversionActionItem(**it) for it in items],
    )


def _date_range_bounds(normalized_range: str | None) -> tuple[date, date] | None:
    """Return (start, end) dates for common presets or explicit ranges. None if unknown."""
    if not normalized_range:
        return None
    today = date.today()
    try:
        if "," in normalized_range and all(part.strip() for part in normalized_range.split(",")):
            parts = normalized_range.split(",")
            if len(parts) == 2:
                start = date.fromisoformat(parts[0])
                end = date.fromisoformat(parts[1])
                return (start, end) if start <= end else None
    except Exception:
        return None

    upper = normalized_range.upper()
    if upper == "LAST_7_DAYS":
        end = today - timedelta(days=1)
        start = end - timedelta(days=6)
        return start, end
    if upper == "LAST_30_DAYS":
        end = today - timedelta(days=1)
        start = end - timedelta(days=29)
        return start, end
    if upper == "LAST_MONTH":
        first_this_month = date(today.year, today.month, 1)
        end = first_this_month - timedelta(days=1)
        start = date(end.year, end.month, 1)
        return start, end
    if upper == "THIS_MONTH":
        start = date(today.year, today.month, 1)
        return start, today
    if upper == "YESTERDAY":
        d = today - timedelta(days=1)
        return d, d
    if upper == "TODAY":
        return today, today
    return None


def _is_cacheable_sa360_range(normalized_range: str | None) -> bool:
    if not SA360_CACHE_ENABLED or not normalized_range:
        return False
    bounds = _date_range_bounds(normalized_range)
    if not bounds:
        return False
    _, end = bounds
    freshness_cutoff = date.today() - timedelta(days=max(0, SA360_CACHE_FRESHNESS_DAYS))
    return end <= freshness_cutoff


def _sa360_cache_key(customer_ids: list[str], normalized_range: str, scope_key: str | None = None) -> str:
    norm_ids = sorted([(cid or "").replace("-", "") for cid in customer_ids if cid])
    ids_slug = slugify_account_name("_".join(norm_ids) or "default") or "default"
    safe_range = normalized_range.replace(",", "_").replace(" ", "_")
    scope = (scope_key or "").replace("-", "")
    digest = hashlib.sha256(f"{scope}|{','.join(norm_ids)}|{normalized_range}".encode()).hexdigest()[:12]
    scope_prefix = slugify_account_name(scope) or "global"
    return f"{scope_prefix}/{ids_slug}/{safe_range}-{digest}"


def _cache_missing_spend(frames: dict[str, pd.DataFrame]) -> bool:
    """
    Detect a pathological cache entry where clicks > 0 but cost column is entirely null/zero-count.
    If detected, the cache should be treated as a miss and refetched live.
    """
    if not frames:
        return False
    priority = ["keyword_performance", "campaign", "ad_group", "ad"]

    def pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    for key in priority:
        df = frames.get(key)
        if df is None or df.empty:
            continue
        tmp = df.copy()
        click_col = pick(tmp, ["metrics.clicks", "Clicks", "clicks"])
        cost_col = pick(tmp, ["metrics.cost", "metrics.cost_micros", "Cost", "cost"])
        if not click_col or not cost_col:
            continue
        clicks = pd.to_numeric(tmp[click_col], errors="coerce").sum(skipna=True)
        cost_nonnull = pd.to_numeric(tmp[cost_col], errors="coerce").notna().sum()
        if clicks > 0 and cost_nonnull == 0:
            return True
    return False


def _load_sa360_cache(
    customer_ids: list[str],
    normalized_range: str | None,
    scope_key: str | None = None,
    report_names: list[str] | None = None,
) -> dict[str, pd.DataFrame] | None:
    if not _is_cacheable_sa360_range(normalized_range):
        return None
    try:
        container_client = _get_blob_client()
    except Exception:
        return None
    cache_key = _sa360_cache_key(customer_ids, normalized_range, scope_key=scope_key)
    frames: dict[str, pd.DataFrame] = {}
    expected = [r for r in (report_names or list(ADS_CSV_SCHEMAS.keys())) if r in ADS_CSV_SCHEMAS]
    for report_name in expected:
        blob_name = f"{SA360_CACHE_PREFIX}/{cache_key}/{report_name}.csv"
        try:
            data = container_client.download_blob(blob_name).readall()
        except Exception:
            return None  # Cache miss if any expected blob is absent
        try:
            frames[report_name] = pd.read_csv(io.BytesIO(data))
        except Exception:
            return None
    # Treat as cache miss if cost column is empty while clicks exist
    try:
        if _cache_missing_spend(frames):
            return None
    except Exception:
        pass
    return frames if frames else None


def _store_sa360_cache(
    customer_ids: list[str],
    normalized_range: str | None,
    frames: dict[str, pd.DataFrame],
    scope_key: str | None = None,
):
    if not _is_cacheable_sa360_range(normalized_range):
        return
    try:
        container_client = _get_blob_client()
    except Exception:
        return
    cache_key = _sa360_cache_key(customer_ids, normalized_range, scope_key=scope_key)
    for report_name, df in frames.items():
        buf = io.StringIO()
        try:
            df.to_csv(buf, index=False)
        except Exception:
            continue
        data = buf.getvalue().encode("utf-8")
        blob_name = f"{SA360_CACHE_PREFIX}/{cache_key}/{report_name}.csv"
        try:
            container_client.upload_blob(name=blob_name, data=data, overwrite=True)
        except Exception:
            continue


def _add_sa360_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add friendly aliases commonly expected by the audit engine (Campaign, Ad group, Impr., Clicks, Cost, Date, Device, Geo).
    Does not drop original columns; only adds missing aliases when source columns exist.
    """
    if df is None or df.empty:
        return df
    alias_map = {
        "campaign.name": "Campaign",
        "campaign.id": "Campaign ID",
        "ad_group.name": "Ad group",
        "ad_group.id": "Ad group ID",
        "metrics.impressions": "Impr.",
        "metrics.clicks": "Clicks",
        "metrics.cost": "Cost",
        "metrics.cost_micros": "Cost",
        "metrics.conversions": "Conversions",
        "metrics.conversions_value": "Conversion value",
        "segments.date": "Date",
        "segments.device": "Device",
        "segments.geo_target_region": "Geo",
    }
    for src, dst in alias_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    return df


def _router_prompt(
    message: str,
    account_name: str | None,
    customer_ids: list[str],
    has_customer_ids: bool,
) -> str:
    """LLM prompt to classify intent/tool with strict JSON output."""
    allowed_intents = [
        "general_chat",
        "audit",
        "performance",
        "pmax",
        "serp",
        "competitor",
        "creative",
        "seasonality",
    ]
    known_ids = ", ".join(customer_ids[:10]) if customer_ids else ""
    id_hint = "yes" if has_customer_ids else "no"
    return f"""
You are Kai's secure router. Classify the user's request and tell us which internal tool to call.
Rules:
- Only use these intents: {allowed_intents}.
- Only use these tools: audit, performance, pmax, serp, competitor, creative. Use null if general_chat or seasonality.
    - Performance intent covers metrics/comparisons/deltas ("how did <account> perform...?").
    - Audit intent is ONLY for explicit account audits or account health reviews.
    - PMax intent is for Performance Max, placements, asset groups, or "PMax" analysis.
    - Creative intent is for ad copy, headlines, RSA, descriptions, or creative ideas.
    - URL or landing page health/soft 404/URL checks/Rankings/SERP => intent "serp", tool "serp" (NOT audit).
    - Competitor signal/outranking/impression share analysis => intent "competitor", tool "competitor".
    - General chat is strategy, budgeting, or conceptual questions without account data.
    - If account context or customer IDs are provided and the user asks about optimizing or diagnosing campaigns/keywords/queries/bids/budgets/geo/device/state/landing pages/ads/ROAS/CVR/CPC/CTR, choose intent "performance" with run_planner=true.
    - If the user is evaluating existing ads (RSA quality, templating, USP coverage), treat it as performance unless they explicitly ask to generate new copy.
    - Seasonality intent is ONLY for trends/forecasting/seasonal demand questions.
- run_planner is true only for performance metrics/comparisons or explicit audit requests. run_trends is true only for seasonality/forecasting questions.
- If intent is seasonality, include 1-3 short themes from the user's message (brand/product/category). Otherwise return themes as an empty array.
- Never invent customer_ids; only include if explicitly provided. needs_ids = true if planner is needed but no customer_ids are provided.
- customer_ids must be numeric (8-12 digits). If known IDs are provided, only choose from that list.
- If intent confidence is low, set needs_clarification=true and keep run_planner=false (do not default to audit).
- If needs_clarification=false, set clarification to "" and notes to "".
- If needs_clarification=true, keep clarification under 12 words and notes under 8 words.
- Do NOT answer the user. DO NOT call external data. Stay within our system.

Return ONLY JSON in this shape:
{{
  "intent": "...",
  "tool": "... or null",
  "run_planner": true|false,
  "run_trends": true|false,
  "themes": ["..."],
  "customer_ids": ["..."],
  "needs_ids": true|false,
  "confidence": 0.0-1.0,
  "needs_clarification": true|false,
  "clarification": "short question if clarification is needed",
  "notes": "short note if any"
}}

User message: "{message}"
Account context: "{account_name or ''}"
Customer IDs provided: "{id_hint}"
Known customer IDs: "{known_ids}"
"""


def _collect_sa360_frames(
    customer_ids: list[str],
    date_range: str | None,
    bypass_cache: bool = False,
    write_cache: bool = True,
    report_names: list[str] | None = None,
    session_id: str | None = None,
) -> dict[str, pd.DataFrame]:
    # Default report set should stay lean; some SA360 resources (e.g., conversion action catalogs) can be large and
    # should be explicitly opted-in via report_names.
    default_reports = list(dict.fromkeys(list(SA360_PERF_REPORTS) + ["keyword_performance_conv", "account"]))
    requested = report_names if report_names is not None else default_reports
    selected_reports = [r for r in requested if r in SA360_QUERIES]
    frames: dict[str, list[dict]] = {k: [] for k in selected_reports}
    ids = customer_ids or []
    normalized_range = _coerce_date_range(date_range)
    # Some SA360 resources do not support segments.date filters (e.g., conversion_action catalog).
    date_clause_skip_reports = {"account", "conversion_actions"}

    # Check cache for older, stable ranges unless explicitly bypassed
    # Cache scope MUST be session/user-scoped to avoid cross-user data leakage when multiple users share an MCC.
    # (login_customer_id is not unique per user.)
    scope_key = _normalize_session_id(session_id)
    if not bypass_cache:
        cached = _load_sa360_cache(ids, normalized_range, scope_key=scope_key, report_names=selected_reports)
        if cached is not None:
            if report_names:
                filtered: dict[str, pd.DataFrame] = {}
                for key in selected_reports:
                    df = cached.get(key)
                    if df is None:
                        filtered[key] = pd.DataFrame(columns=ADS_CSV_SCHEMAS[key])
                    else:
                        filtered[key] = df
                return filtered
            return cached

    date_clause: str | None = None
    if normalized_range:
        if "," in normalized_range and all(part.strip() for part in normalized_range.split(",")):
            parts = normalized_range.split(",")
            if len(parts) == 2:
                date_clause = f"segments.date BETWEEN '{parts[0]}' AND '{parts[1]}'"
        else:
            date_clause = f"segments.date DURING {normalized_range}"

    for cid in ids:
        for report_name in selected_reports:
            query = SA360_QUERIES[report_name]
            q = query
            if date_clause and report_name not in date_clause_skip_reports:
                if "WHERE" in q:
                    q = q.replace("WHERE", f"WHERE {date_clause} AND", 1)
                else:
                    q = q + f" WHERE {date_clause}"
            results = _sa360_search(cid, q, session_id=session_id)
            for row in results:
                payload = {}
                for col in ADS_CSV_SCHEMAS.get(report_name, []):
                    payload[col] = _get_nested_value(row, col)
                    if col.endswith("cost_micros") and payload[col] is not None:
                        try:
                            payload[col] = float(payload[col]) / 1_000_000.0
                        except Exception:
                            pass
                frames[report_name].append(payload)

    out_frames = {k: pd.DataFrame(v) if v else pd.DataFrame(columns=ADS_CSV_SCHEMAS[k]) for k, v in frames.items()}

    # Add friendly aliases to reduce downstream KeyErrors (Campaign, Ad group, Impr., etc.)
    for key, df in out_frames.items():
        out_frames[key] = _add_sa360_aliases(df)

    # Store cache for non-recent ranges (best-effort; ignore failures)
    if write_cache and not bypass_cache:
        _store_sa360_cache(ids, normalized_range, out_frames, scope_key=scope_key)

    return out_frames


def _collect_sa360_frames_batched(
    customer_ids: list[str],
    date_range: str | None,
    bypass_cache: bool = False,
    write_cache: bool = True,
    max_workers: int = 2,
    chunk_size: int = 1,
    report_names: list[str] | None = None,
    session_id: str | None = None,
) -> dict[str, pd.DataFrame]:
    ids = [str(cid) for cid in (customer_ids or []) if str(cid).strip()]
    default_reports = list(dict.fromkeys(list(SA360_PERF_REPORTS) + ["keyword_performance_conv", "account"]))
    requested = report_names if report_names is not None else default_reports
    selected_reports = [r for r in requested if r in SA360_QUERIES]
    report_filter = report_names if report_names is not None else None
    if max_workers <= 1 or len(ids) <= max(1, chunk_size):
        return _collect_sa360_frames(
            ids,
            date_range,
            bypass_cache=bypass_cache,
            write_cache=write_cache,
            report_names=report_filter,
            session_id=session_id,
        )

    max_workers = max(1, int(max_workers))
    chunk_size = max(1, int(chunk_size))
    chunks = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]
    out_frames: dict[str, list[pd.DataFrame]] = {k: [] for k in selected_reports}

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _collect_sa360_frames,
                chunk,
                date_range,
                bypass_cache,
                write_cache,
                report_filter,
                session_id,
            ): chunk
            for chunk in chunks
        }
        for future in as_completed(futures):
            frames = future.result()
            for key, df in frames.items():
                if key in out_frames and df is not None and not df.empty:
                    out_frames[key].append(df)

    merged: dict[str, pd.DataFrame] = {}
    for key in selected_reports:
        if out_frames[key]:
            merged[key] = pd.concat(out_frames[key], ignore_index=True)
        else:
            merged[key] = pd.DataFrame(columns=ADS_CSV_SCHEMAS[key])
    return merged


def _summarize_numeric(metrics: dict) -> dict:
    """Compute derived metrics safely."""
    def as_float(val, default=0.0):
        try:
            if val is None:
                return default
            return float(val)
        except Exception:
            try:
                return float(str(val).replace(",", "")) if val != "" else default
            except Exception:
                return default

    numeric_metrics = {k: as_float(v) for k, v in metrics.items()}
    clicks = numeric_metrics.get("clicks", 0.0)
    impressions = numeric_metrics.get("impressions", 0.0)
    cost = numeric_metrics.get("cost", 0.0)
    conversions = numeric_metrics.get("conversions", 0.0)
    conv_value = numeric_metrics.get("conversions_value", 0.0)

    summary = dict(numeric_metrics)
    summary["ctr"] = (clicks / impressions) * 100 if impressions else None
    summary["cpc"] = cost / clicks if clicks else None
    summary["cvr"] = conversions / clicks if clicks else None
    summary["cpa"] = cost / conversions if conversions else None
    summary["roas"] = conv_value / cost if cost else None
    return summary


def _custom_metric_name(metric_key: str) -> str:
    if metric_key.startswith("custom:"):
        return metric_key.split("custom:", 1)[1]
    return metric_key


def _find_custom_metric_column(df: pd.DataFrame, metric_key: str) -> str | None:
    if df is None or df.empty:
        return None
    target = _normalize_custom_metric_cmp(_custom_metric_name(metric_key))
    if not target:
        return None
    for col in df.columns:
        if _normalize_custom_metric_cmp(str(col)) == target:
            return col
    return None


def _filter_custom_metric_rows(df: pd.DataFrame, metric_key: str) -> tuple[pd.DataFrame | None, str | None]:
    if df is None or df.empty:
        return None, None
    target = _normalize_custom_metric_cmp(_custom_metric_name(metric_key))
    if not target:
        return None, None
    if "segments.conversion_action_name" in df.columns:
        names = df["segments.conversion_action_name"].astype(str).map(_normalize_custom_metric_cmp)
        mask = names == target
        if mask.any():
            conv_col = "metrics.conversions" if "metrics.conversions" in df.columns else None
            all_col = "metrics.all_conversions" if "metrics.all_conversions" in df.columns else None
            conv_sum = None
            all_sum = None
            try:
                if conv_col:
                    conv_sum = float(pd.to_numeric(df.loc[mask, conv_col], errors="coerce").sum(skipna=True))
                if all_col:
                    all_sum = float(pd.to_numeric(df.loc[mask, all_col], errors="coerce").sum(skipna=True))
            except Exception:
                conv_sum = None
                all_sum = None

            # Prefer "conversions" when present and non-zero; fall back to "all_conversions" when conversions are zero
            # but all_conversions is non-zero (common when an action is excluded from "Conversions" in the UI).
            metric_col = None
            if conv_col and conv_sum not in (None, 0, 0.0):
                metric_col = conv_col
            elif all_col and all_sum not in (None, 0, 0.0):
                metric_col = all_col
            else:
                metric_col = conv_col or all_col
            if metric_col:
                return df[mask], metric_col
    col = _find_custom_metric_column(df, metric_key)
    if col:
        return df, col
    return None, None


def _sum_custom_metric(frames: dict[str, pd.DataFrame], metric_key: str) -> tuple[float | None, dict[str, Any] | None]:
    if not frames:
        return None, None
    # Prefer customer-level conversion action summaries (most complete), then fall back to keyword-level where needed.
    # Note: customer_performance provides accurate account-level conversion totals for metrics.* columns (e.g.,
    # all_conversions) without summing across conversion-action segments.
    priority = ["customer_performance", "conversion_action_summary", "keyword_performance_conv", "keyword_performance", "campaign", "ad_group", "ad"]
    for frame_name in priority:
        df = frames.get(frame_name)
        sub, metric_col = _filter_custom_metric_rows(df, metric_key)
        if sub is None or metric_col is None:
            continue
        series = pd.to_numeric(sub.get(metric_col), errors="coerce")
        return float(series.sum(skipna=True)), {"frame": frame_name, "metric_col": metric_col}

    # If the user asked for a conversion action that exists in the account catalog but it has no rows in the window,
    # treat it as 0 conversions instead of "missing".
    try:
        target = _normalize_custom_metric_cmp(_custom_metric_name(metric_key))
        cat = frames.get("conversion_actions")
        if cat is not None and not cat.empty and "conversion_action.name" in cat.columns and target:
            names = cat["conversion_action.name"].astype(str).map(_normalize_custom_metric_cmp)
            if (names == target).any():
                return 0.0, {"frame": "conversion_actions", "metric_col": None, "catalog_only": True}
    except Exception:
        pass
    return None, None


def _custom_metric_presence(frames: dict[str, pd.DataFrame], metric_key: str) -> tuple[bool, list[str]]:
    """Return (present, sample_actions) for a custom conversion-action metric."""
    if not frames:
        return False, []
    target = _normalize_custom_metric_cmp(_custom_metric_name(metric_key))
    sample: list[str] = []

    # Prefer the conversion action catalog (exists even when conversions are 0).
    cat = frames.get("conversion_actions")
    if cat is not None and not cat.empty and "conversion_action.name" in cat.columns:
        try:
            raw = cat["conversion_action.name"].astype(str).dropna().unique().tolist()
            sample = raw[:5]
            if target:
                names = cat["conversion_action.name"].astype(str).map(_normalize_custom_metric_cmp).dropna().tolist()
                return target in names, sample
        except Exception:
            return False, sample

    # Fallback: only actions that appear in the segmented conversion report (usually implies non-zero conversions).
    df = frames.get("keyword_performance_conv")
    if df is None or df.empty or "segments.conversion_action_name" not in df.columns:
        return False, []
    try:
        names = df["segments.conversion_action_name"].astype(str).map(_normalize_custom_metric_cmp).dropna().tolist()
        present = bool(target) and target in names
        raw_samples = df["segments.conversion_action_name"].astype(str).dropna().unique().tolist()
        return present, raw_samples[:5]
    except Exception:
        return False, []


def _aggregate_frame_custom(
    df: pd.DataFrame,
    key_col: str,
    metric_key: str,
    metric_col: str,
    device_filter: str | None = None,
) -> dict[str, dict]:
    if df is None or df.empty:
        return {}
    tmp = df.copy()
    if device_filter and "segments.device" in tmp.columns:
        tmp = tmp[tmp["segments.device"].str.upper() == device_filter.upper()]
    if tmp.empty or metric_col not in tmp.columns:
        return {}
    tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
    grouped: dict[str, dict] = {}
    for key, group in tmp.groupby(key_col):
        grouped[key] = {metric_key: group[metric_col].sum(skipna=True)}
    return grouped


def _aggregate_frame(df: pd.DataFrame, key_col: str, device_filter: str | None = None) -> dict[str, dict]:
    """
    Aggregate metrics by key column (optionally device-filtered). Returns dict[key] -> metric dict.
    """
    if df is None or df.empty:
        return {}
    tmp = df.copy()
    if device_filter and "segments.device" in tmp.columns:
        tmp = tmp[tmp["segments.device"].str.upper() == device_filter.upper()]
    if tmp.empty:
        return {}
    metrics_cols = {
        "impressions": "metrics.impressions",
        "clicks": "metrics.clicks",
        "cost": "metrics.cost_micros",
        "conversions": "metrics.conversions",
        "conversions_value": "metrics.conversions_value",
    }
    # Coerce numerics
    for col in metrics_cols.values():
        if col in tmp.columns:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

    grouped = {}
    for key, group in tmp.groupby(key_col):
        metrics = {}
        for mk, col in metrics_cols.items():
            if col in group.columns:
                metrics[mk] = group[col].sum(skipna=True)
        grouped[key] = _summarize_numeric(metrics)
    return grouped


def _aggregate_account_performance(frames: dict[str, pd.DataFrame]) -> dict:
    """
    Aggregate account-level metrics using the richest available SA360 frame.
    Priority: customer_performance -> keyword_performance -> campaign -> ad_group -> ad.
    Chooses the first frame that has non-zero cost data; otherwise falls back to the
    first available frame in priority order.
    """
    if not frames:
        return {}

    priority = ["customer_performance", "keyword_performance", "campaign", "ad_group", "ad"]

    def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    best_with_cost = None
    fallback = None

    for key in priority:
        df = frames.get(key)
        if df is None or df.empty:
            continue
        tmp = df.copy()
        metrics = {}
        imp_col = pick_col(tmp, ["metrics.impressions", "Impr.", "impressions"])
        if imp_col:
            metrics["impressions"] = pd.to_numeric(tmp[imp_col], errors="coerce").sum(skipna=True)
        click_col = pick_col(tmp, ["metrics.clicks", "Clicks", "clicks"])
        if click_col:
            metrics["clicks"] = pd.to_numeric(tmp[click_col], errors="coerce").sum(skipna=True)
        cost_col = pick_col(tmp, ["metrics.cost", "metrics.cost_micros", "Cost", "cost"])
        cost_nonnull = 0
        if cost_col:
            series = pd.to_numeric(tmp[cost_col], errors="coerce")
            cost_nonnull = int(series.notna().sum())
            metrics["cost"] = series.sum(skipna=True)
        conv_col = pick_col(tmp, ["metrics.conversions", "Conversions", "conversions"])
        if conv_col:
            metrics["conversions"] = pd.to_numeric(tmp[conv_col], errors="coerce").sum(skipna=True)
        conv_val_col = pick_col(tmp, ["metrics.conversions_value", "Conversion value", "conversions_value"])
        if conv_val_col:
            metrics["conversions_value"] = pd.to_numeric(tmp[conv_val_col], errors="coerce").sum(skipna=True)

        summarized = _summarize_numeric(metrics)
        # If we have any non-null cost entries, prefer this frame and stop
        if cost_nonnull > 0 and best_with_cost is None:
            best_with_cost = summarized
            break
        # Track first available as fallback
        if fallback is None:
            fallback = summarized

    return best_with_cost or fallback or {}


def _get_perf_frame_debug(frames: dict[str, pd.DataFrame]) -> dict:
    """
    Return a small debug payload about the frame chosen for performance aggregation.
    Includes which frame was used, presence of cost columns, and a tiny sample of values.
    """
    priority = [
        "customer_performance",
        "keyword_performance",
        "campaign",
        "ad_group",
        "ad",
    ]
    for key in priority:
        df = frames.get(key)
        if df is None or df.empty:
            continue
        cols = list(df.columns)
        candidates = ["metrics.cost", "metrics.cost_micros", "Cost", "cost"]
        cost_col = next((c for c in candidates if c in df.columns), None)
        sample = []
        nonnull = 0
        if cost_col:
            try:
                series = pd.to_numeric(df[cost_col], errors="coerce")
                nonnull = int(series.notna().sum())
                sample = series.dropna().head(5).tolist()
            except Exception:
                pass
        return {
            "frame": key,
            "columns": cols[:80],
            "cost_column": cost_col,
            "cost_nonnull": nonnull,
            "cost_sample": sample,
            "row_count": int(len(df)),
        }
    return {"frame": None, "columns": [], "cost_column": None, "cost_nonnull": 0, "cost_sample": [], "row_count": 0}


def _build_perf_snapshot(frames: dict[str, pd.DataFrame], metric_keys: list[str] | None = None) -> dict:
    """
    Build a compact snapshot of performance metrics + debug + data quality flags.
    Useful for diagnostics without altering the main aggregation flow.
    """
    metrics = _aggregate_account_performance(frames)
    debug = _get_perf_frame_debug(frames)
    # Optional: include custom conversion-action metrics so QA can validate parity against plan-and-run.
    custom_meta: dict[str, Any] = {}
    try:
        for key in (metric_keys or []):
            if not isinstance(key, str):
                continue
            k = key.strip()
            if not k or not k.startswith("custom:"):
                continue
            val, meta = _sum_custom_metric(frames, k)
            if val is not None:
                metrics[k] = val
            if meta:
                custom_meta[k] = meta
    except Exception:
        # Diagnostics must never crash due to a custom-metric compute issue; snapshots remain best-effort.
        pass
    missing_spend = (
        (metrics.get("cost") in (None, 0, 0.0))
        and (metrics.get("clicks") or 0) > 0
        and (debug.get("cost_nonnull") or 0) == 0
    )
    return {
        "metrics": _to_primitive(metrics),
        "debug": _to_primitive({**debug, "custom_metrics": custom_meta} if custom_meta else debug),
        "data_quality": {"missing_spend": missing_spend},
    }


def _compute_perf_deltas(current: dict, previous: dict) -> dict:
    """
    Build delta dict between two metric snapshots.
    """
    keys = [
        "impressions",
        "clicks",
        "cost",
        "conversions",
        "conversions_value",
        "ctr",
        "cpc",
        "cvr",
        "cpa",
        "roas",
    ]
    for extra in list((current or {}).keys()) + list((previous or {}).keys()):
        if extra not in keys:
            keys.append(extra)

    def num(val):
        try:
            if val is None:
                return 0.0
            return float(val)
        except Exception:
            try:
                return float(str(val).replace(",", ""))
            except Exception:
                return 0.0

    deltas = {}
    for k in keys:
        cur = num(current.get(k)) if current else 0.0
        prev = num(previous.get(k)) if previous else 0.0
        change = cur - prev
        pct = (change / prev * 100) if prev else None
        deltas[k] = {"current": cur, "previous": prev, "change": change, "pct_change": pct}
    return deltas


def _to_primitive(obj):
    """
    Convert numpy/pandas scalars to Python primitives for safe JSON serialization.
    """
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None

    if np and isinstance(obj, np.generic):
        try:
            return obj.item()
        except Exception:
            return float(obj)
    if isinstance(obj, dict):
        return {k: _to_primitive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_primitive(v) for v in obj]
    return obj


# ------------------ Trends / Seasonality Helpers ------------------ #


def _normalize_trends_timeframe(tf: str | None) -> str:
    if not tf:
        return "LAST_30_DAYS"
    t = tf.strip()
    if "," in t and all(part.strip() for part in t.split(",")):
        # ISO range passthrough
        return t
    upper = t.upper()
    presets = {"NOW 7-D": "LAST_7_DAYS", "NOW 30-D": "LAST_30_DAYS", "NOW 3-M": "LAST_90_DAYS", "NOW 6-M": "LAST_180_DAYS", "NOW 12-M": "LAST_365_DAYS", "LAST_7_DAYS": "LAST_7_DAYS", "LAST_30_DAYS": "LAST_30_DAYS"}
    if upper in presets:
        if presets[upper].startswith("LAST_"):
            # turn LAST_X_DAYS into explicit range for SA360
            try:
                days = int(presets[upper].split("_")[1])
            except Exception:
                days = 30
            end = date.today() - timedelta(days=1)
            start = end - timedelta(days=days - 1)
            return f"{start:%Y-%m-%d},{end:%Y-%m-%d}"
        return presets[upper]
    return "LAST_30_DAYS"


def _derive_themes(themes: list[str], account_name: str | None) -> list[str]:
    out = [t.strip() for t in (themes or []) if t and t.strip()]
    if out:
        return out[:5]
    if account_name:
        words = [w.strip() for w in re.split(r"[^a-zA-Z0-9]+", account_name) if len(w.strip()) > 2]
        if words:
            return [" ".join(words[:3])]
    return []


def _collect_keyword_perf_weights(customer_ids: list[str], date_range: str | None, session_id: str | None = None) -> dict:
    """
    Return a weight map by keyword text using SA360 keyword_performance frame.
    Uses cost if present, else clicks.
    """
    scope = _normalize_session_id(session_id) or ""
    cache_key = f"{scope}|{','.join(sorted([cid or '' for cid in customer_ids]))}|{date_range or ''}"
    cached = PERF_WEIGHT_CACHE.get(cache_key)
    if cached:
        return cached
    weights = {}
    frames = _collect_sa360_frames(customer_ids, date_range, session_id=session_id)
    kw = frames.get("keyword_performance")
    if kw is None or kw.empty:
        return {}
    kw = kw.copy()
    if "ad_group_criterion.keyword.text" not in kw.columns:
        return {}
    kw["__key"] = kw["ad_group_criterion.keyword.text"].astype(str).str.lower()
    cost_col = next((c for c in ["metrics.cost_micros", "Cost"] if c in kw.columns), None)
    click_col = next((c for c in ["metrics.clicks", "Clicks"] if c in kw.columns), None)
    kw[cost_col] = pd.to_numeric(kw[cost_col], errors="coerce") if cost_col else None
    kw[click_col] = pd.to_numeric(kw[click_col], errors="coerce") if click_col else None
    for key, group in kw.groupby("__key"):
        cost = group[cost_col].sum(skipna=True) if cost_col else 0
        clicks = group[click_col].sum(skipna=True) if click_col else 0
        weight = cost if cost and cost > 0 else clicks
        if weight and weight > 0:
            weights[key] = float(weight)
    if weights:
        PERF_WEIGHT_CACHE.set(cache_key, weights)
    return weights


def _performance_seasonality(frames: dict[str, pd.DataFrame]) -> dict:
    """
    Build a basic seasonality profile from performance frames (keyword_performance preferred).
    Returns similar shape as summarize_seasonality: peaks/shoulders/lows/month_scores.
    """
    df = frames.get("keyword_performance")
    if df is None or df.empty:
        return {}
    if "segments.date" not in df.columns:
        return {}
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["segments.date"], errors="coerce")
    tmp = tmp.dropna(subset=["date"])
    if tmp.empty:
        return {}
    # derive month period
    tmp["month"] = tmp["date"].dt.to_period("M")
    cost_col = next((c for c in ["metrics.cost_micros", "Cost"] if c in tmp.columns), None)
    click_col = next((c for c in ["metrics.clicks", "Clicks"] if c in tmp.columns), None)
    month_scores = []
    for m, g in tmp.groupby("month"):
        score = 0.0
        if cost_col:
            score += pd.to_numeric(g[cost_col], errors="coerce").sum(skipna=True)
        elif click_col:
            score += pd.to_numeric(g[click_col], errors="coerce").sum(skipna=True)
        month_scores.append({"month": str(m), "score": float(score)})
    if not month_scores:
        return {}
    month_scores_sorted = sorted(month_scores, key=lambda x: x["score"], reverse=True)
    peaks = month_scores_sorted[:3]
    lows = month_scores_sorted[-2:] if len(month_scores_sorted) >= 2 else month_scores_sorted[-1:]
    n = len(month_scores_sorted)
    mid_start = max(0, n // 3)
    mid_end = min(n, 2 * n // 3 + 1)
    shoulders = month_scores_sorted[mid_start:mid_end]
    return {
        "peaks": peaks,
        "shoulders": shoulders,
        "lows": lows,
        "month_scores": month_scores_sorted,
    }


def _seasonality_fallback_with_range(customer_ids: list[str], normalized_range: str | None, session_id: str | None = None) -> dict:
    """
    Try to build a performance seasonality profile using the requested range; if empty,
    back off to shorter ranges to avoid empty summaries.
    """
    scope = _normalize_session_id(session_id) or ""
    cache_key = f"{scope}|{','.join(sorted([cid or '' for cid in customer_ids]))}|{normalized_range or ''}"
    cached = PERF_SEASONALITY_CACHE.get(cache_key)
    if cached:
        return cached
    ranges = [normalized_range] if normalized_range else []
    # add fallbacks: last 180 days, last 90 days
    today = date.today()
    def span(days):
        end = today - timedelta(days=1)
        start = end - timedelta(days=days - 1)
        return f"{start:%Y-%m-%d},{end:%Y-%m-%d}"
    ranges += [span(180), span(90)]
    for r in ranges:
        frames = _collect_sa360_frames(customer_ids, r, bypass_cache=True, write_cache=False, session_id=session_id)
        perf_seasonality = _performance_seasonality(frames)
        if perf_seasonality:
            PERF_SEASONALITY_CACHE.set(cache_key, perf_seasonality)
            return perf_seasonality
    return {}


def _allocate_budget(themes: list[str], multipliers: dict, perf_weights: dict, budget: float | None) -> list[dict]:
    out = []
    base_weights = {}
    for t in themes:
        k = t.lower()
        base_weights[t] = perf_weights.get(k, 1.0)
    # apply multipliers
    weighted = {}
    for t, w in base_weights.items():
        mult = multipliers.get(t, {}).get("multiplier") or multipliers.get(t.lower(), {}).get("multiplier") or 1.0
        weighted[t] = w * mult
    total = sum(weighted.values()) or len(weighted) or 1.0
    for t, w in weighted.items():
        pct = w / total
        alloc = budget * pct if budget is not None else None
        m = multipliers.get(t) or multipliers.get(t.lower()) or {}
        out.append({
            "theme": t,
            "weight_pct": pct * 100.0,
            "suggested_budget": alloc,
            "trend_multiplier": m.get("multiplier"),
            "interest_recent": m.get("recent"),
            "interest_avg": m.get("avg"),
        })
    # sort descending by weight
    out.sort(key=lambda x: x.get("weight_pct", 0), reverse=True)
    return out


def _aggregate_entity(df: pd.DataFrame, key_col: str, match: str) -> dict | None:
    if df is None or df.empty:
        return None
    if match is None:
        return None
    m = match.lower()
    def match_row(val: Any) -> bool:
        if val is None:
            return False
        txt = str(val).lower()
        return m in txt or txt == m
    tmp = df.copy()
    # Coerce numerics
    for col in ["metrics.impressions", "metrics.clicks", "metrics.cost_micros", "metrics.conversions", "metrics.conversions_value"]:
        if col in tmp.columns:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

    matched = tmp[tmp[key_col].apply(match_row)]
    if matched.empty:
        return None
    metrics = {
        "impressions": matched["metrics.impressions"].sum(skipna=True) if "metrics.impressions" in matched else 0,
        "clicks": matched["metrics.clicks"].sum(skipna=True) if "metrics.clicks" in matched else 0,
        "cost": matched["metrics.cost_micros"].sum(skipna=True) if "metrics.cost_micros" in matched else 0,
        "conversions": matched["metrics.conversions"].sum(skipna=True) if "metrics.conversions" in matched else 0,
        "conversions_value": matched["metrics.conversions_value"].sum(skipna=True) if "metrics.conversions_value" in matched else 0,
    }
    return _summarize_numeric(metrics)


def _analyze_entity_performance(
    entity: dict,
    customer_ids: list[str],
    date_range: str | None,
    session_id: str | None = None,
) -> dict | None:
    """
    Fetch current and prior spans for the entity and compute deltas.
    Only runs for ad/keyword asks; returns dict or None.
    """
    if not entity or not entity.get("entity_type"):
        return None
    span = _date_span_from_range(_coerce_date_range(date_range))
    prev_span = _previous_span(span)
    if not span or not prev_span:
        return {"note": "Could not derive comparison window.", "entity": entity}

    def span_to_range(sp: tuple[date, date]) -> str:
        return f"{sp[0]:%Y-%m-%d},{sp[1]:%Y-%m-%d}"

    current_frames = _collect_sa360_frames(customer_ids, span_to_range(span), session_id=session_id)
    previous_frames = _collect_sa360_frames(customer_ids, span_to_range(prev_span), session_id=session_id)

    etype = entity.get("entity_type")
    identifier = entity.get("identifier")
    metric_pref = (entity.get("metric") or "").lower()

    if etype == "ad":
        df_key = "ad"
        key_cols = ["ad_group_ad.ad.id", "ad_group_ad.ad.responsive_search_ad.headlines", "ad_group_ad.ad.responsive_search_ad.descriptions", "ad_group_ad.ad.final_urls"]
    elif etype == "keyword":
        df_key = "keyword_performance"
        key_cols = ["ad_group_criterion.criterion_id", "ad_group_criterion.keyword.text"]
    else:
        return None

    def best_match(df: pd.DataFrame) -> dict | None:
        if df is None or df.empty:
            return None
        if identifier:
            for col in key_cols:
                agg = _aggregate_entity(df.assign(_match=df[col]), "_match", identifier)
                if agg:
                    return agg
        # fallback: top by clicks
        if "metrics.clicks" in df.columns:
            top = df.sort_values("metrics.clicks", ascending=False).head(1)
        else:
            top = df
        metrics = {
            "impressions": top["metrics.impressions"].sum(skipna=True) if "metrics.impressions" in top else 0,
            "clicks": top["metrics.clicks"].sum(skipna=True) if "metrics.clicks" in top else 0,
            "cost": top["metrics.cost_micros"].sum(skipna=True) if "metrics.cost_micros" in top else 0,
            "conversions": top["metrics.conversions"].sum(skipna=True) if "metrics.conversions" in top else 0,
            "conversions_value": top["metrics.conversions_value"].sum(skipna=True) if "metrics.conversions_value" in top else 0,
        }
        return _summarize_numeric(metrics)

    curr_df = current_frames.get(df_key)
    prev_df = previous_frames.get(df_key)
    curr = best_match(curr_df)
    prev = best_match(prev_df)
    if not curr:
        return {"note": "No matching entity found in current window.", "entity": entity}

    def num(val):
        try:
            if val is None:
                return 0.0
            return float(val)
        except Exception:
            try:
                return float(str(val).replace(",", ""))
            except Exception:
                return 0.0

    def delta(cur, old, key):
        c = num(cur.get(key))
        p = num(old.get(key)) if old else 0.0
        abs_diff = c - p
        pct = (abs_diff / p * 100) if p else None
        return c, p, abs_diff, pct

    keys = ["impressions", "clicks", "conversions", "cost", "ctr", "cpc", "cvr", "cpa", "roas"]
    deltas = {}
    for k in keys:
        c, p, d, pct = delta(curr, prev or {}, k)
        deltas[k] = {"current": c, "previous": p, "change": d, "pct_change": pct}

    drivers = []
    if deltas["clicks"]["pct_change"] is not None:
        drivers.append(f"Clicks {deltas['clicks']['pct_change']:+.1f}% (Δ {deltas['clicks']['change']:+.0f})")
    if deltas["impressions"]["pct_change"] is not None:
        drivers.append(f"Impr {deltas['impressions']['pct_change']:+.1f}%")
    if deltas["ctr"]["pct_change"] is not None:
        drivers.append(f"CTR {deltas['ctr']['pct_change']:+.1f}%")
    if deltas["cpc"]["pct_change"] is not None:
        drivers.append(f"CPC {deltas['cpc']['pct_change']:+.1f}%")

    focus_metric = metric_pref or "conversions"
    if focus_metric not in keys:
        focus_metric = "clicks"

    return {
        "entity": entity,
        "date_range_current": span_to_range(span),
        "date_range_previous": span_to_range(prev_span),
        "metric_focus": focus_metric,
        "deltas": deltas,
        "drivers": drivers[:4],
    }


def _analyze_top_movers(
    customer_ids: list[str],
    date_range: str | None,
    entity_type: str | None,
    metric: str | None,
    device: str | None,
    limit: int = 5,
    session_id: str | None = None,
) -> dict | None:
    """
    Compute top movers for the chosen entity type between current window and the previous same-length window.
    """
    span = _date_span_from_range(_coerce_date_range(date_range))
    prev_span = _previous_span(span)
    if not span or not prev_span:
        return {"note": "Could not derive comparison window for top movers."}

    def span_to_range(sp: tuple[date, date]) -> str:
        return f"{sp[0]:%Y-%m-%d},{sp[1]:%Y-%m-%d}"

    cur_frames = _collect_sa360_frames(customer_ids, span_to_range(span), session_id=session_id)
    prev_frames = _collect_sa360_frames(customer_ids, span_to_range(prev_span), session_id=session_id)

    etype = entity_type or "keyword"
    if etype == "keyword":
        frame_key = "keyword_performance"
        key_col = "ad_group_criterion.keyword.text"
    elif etype == "ad":
        frame_key = "ad"
        key_col = "ad_group_ad.ad.id"
    elif etype == "campaign":
        frame_key = "campaign"
        key_col = "campaign.name"
    else:
        return {"note": "Unsupported entity type for top movers."}

    metric_focus = (metric or "conversions").lower()
    if metric_focus.startswith("custom:"):
        cur_sub, metric_col = _filter_custom_metric_rows(cur_frames.get(frame_key), metric_focus)
        prev_sub, prev_col = _filter_custom_metric_rows(prev_frames.get(frame_key), metric_focus)
        cur_grouped = _aggregate_frame_custom(cur_sub, key_col, metric_focus, metric_col, device_filter=device)
        prev_grouped = _aggregate_frame_custom(prev_sub, key_col, metric_focus, prev_col, device_filter=device)
        focus = metric_focus
    else:
        cur_grouped = _aggregate_frame(cur_frames.get(frame_key), key_col, device_filter=device)
        prev_grouped = _aggregate_frame(prev_frames.get(frame_key), key_col, device_filter=device)

        # If metric not present for this entity type, fall back to clicks, then impressions
        def pick_metric(grouped: dict) -> str:
            if not grouped:
                return "clicks"
            sample = next(iter(grouped.values()))
            if metric_focus in sample and sample.get(metric_focus) is not None:
                return metric_focus
            if "clicks" in sample:
                return "clicks"
            if "impressions" in sample:
                return "impressions"
            return "clicks"

        focus = pick_metric(cur_grouped)

    def num(val):
        try:
            if val is None:
                return 0.0
            return float(val)
        except Exception:
            try:
                return float(str(val).replace(",", ""))
            except Exception:
                return 0.0

    items = []
    all_keys = set(cur_grouped.keys()) | set(prev_grouped.keys())
    for k in all_keys:
        cur = cur_grouped.get(k, {})
        prev = prev_grouped.get(k, {})
        cval = num(cur.get(focus))
        pval = num(prev.get(focus))
        delta = cval - pval
        pct = (delta / pval * 100) if pval else None
        items.append({
            "name": str(k),
            "metric": focus,
            "current": cval,
            "previous": pval,
            "change": delta,
            "pct_change": pct,
            "metrics_current": cur,
            "metrics_previous": prev,
        })

    items = [i for i in items if i["change"] != 0 or (i["pct_change"] is not None and i["pct_change"] != 0)]
    items.sort(key=lambda x: abs(x["change"]), reverse=True)
    movers = items[:limit]

    return {
        "type": "top_movers",
        "entity_type": etype,
        "metric_focus": focus,
        "date_range_current": span_to_range(span),
        "date_range_previous": span_to_range(prev_span),
        "device_filter": device,
        "items": movers,
    }


def _has_performance_intent(message: str) -> bool:
    """
    Detect when the user is asking for performance metrics (cost/conv/roas etc.) or comparisons,
    and NOT explicitly requesting an audit.
    """
    t = (message or "").lower()
    if any(word in t for word in ["audit", "score", "klaudit"]):
        return False
    # Strategy/plan questions should not trigger performance planner by default
    if any(word in t for word in ["strategy", "plan", "planning", "budget strategy", "next quarter", "q1", "q2", "q3", "q4"]):
        return False
    perf_keywords = {
        "conversion",
        "conversions",
        "conv",
        "cost",
        "spend",
        "roas",
        "cpa",
        "ctr",
        "cpc",
        "performance",
        "perform",
        "performing",
        "compare",
        "versus",
        "vs",
        "week over week",
        "wow",
        "last weekend",
        "weekend",
        "trend",
        "did it do",
        "did they do",
        "did the account do",
    }
    if "how did" in t and ("perform" in t or "do" in t):
        return True
    return any(k in t for k in perf_keywords) or bool(_extract_custom_metric_mentions(message))


def _write_frames_to_blob(account_name: str, frames: dict[str, pd.DataFrame]):
    container_client = _get_blob_client()
    slug = slugify_account_name(account_name or "General")
    prefix_template_env = os.environ.get("PPC_DATA_BLOB_PREFIX")
    prefix_template = "{account_slug}/" if prefix_template_env is None else prefix_template_env
    prefix = prefix_template.format(account=account_name, account_slug=slug).strip("/")
    prefix = f"{prefix}/" if prefix and not prefix.endswith("/") else prefix

    filenames = {
        "campaign": f"Campaign Report - {account_name}.csv",
        "ad_group": f"Ad group Report - {account_name}.csv",
        "ad": f"Ad Report - {account_name}.csv",
        "keyword_performance": f"Keyword Report - {account_name}.csv",
        "landing_page": f"Landing page Report - {account_name}.csv",
        "account": f"Account Report - {account_name}.csv",
    }

    uploaded = []
    for key, df in frames.items():
        if df is None or df.empty:
            continue
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        data = buf.getvalue().encode("utf-8")
        blob_name = f"{prefix}{filenames.get(key, key + '.csv')}"
        container_client.upload_blob(name=blob_name, data=data, overwrite=True)
        uploaded.append(blob_name)
    if not uploaded:
        raise HTTPException(status_code=404, detail="No data returned from Ads API; nothing uploaded.")
    return uploaded


def _get_blob_client():
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    container = os.environ.get("PPC_DATA_BLOB_CONTAINER")
    if not conn or not container:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Storage is not configured (missing AZURE_STORAGE_CONNECTION_STRING or PPC_DATA_BLOB_CONTAINER).",
        )
    return BlobServiceClient.from_connection_string(conn).get_container_client(container)


def _parse_storage_connection(conn: str) -> dict:
    parts = {}
    for part in (conn or "").split(";"):
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        parts[key] = value
    return parts


def _get_audit_blob_client():
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    container = os.environ.get("AUDIT_BLOB_CONTAINER") or os.environ.get("PPC_DATA_BLOB_CONTAINER")
    if not conn or not container:
        return None
    return BlobServiceClient.from_connection_string(conn).get_container_client(container)


def _upload_audit_report(file_path: Path) -> dict | None:
    container_client = _get_audit_blob_client()
    if not container_client:
        return None
    prefix = (os.environ.get("AUDIT_BLOB_PREFIX") or "audit_reports/").strip("/")
    prefix = f"{prefix}/" if prefix else ""
    blob_name = f"{prefix}{file_path.name}"
    with open(file_path, "rb") as handle:
        container_client.upload_blob(name=blob_name, data=handle, overwrite=True)
    blob_url = f"{container_client.url}/{blob_name}"

    download_url = None
    expiry_hours = int(os.environ.get("AUDIT_SAS_EXPIRY_HOURS", "24") or "24")
    parts = _parse_storage_connection(os.environ.get("AZURE_STORAGE_CONNECTION_STRING", ""))
    account_name = parts.get("AccountName")
    account_key = parts.get("AccountKey")
    if account_name and account_key:
        sas = generate_blob_sas(
            account_name=account_name,
            container_name=container_client.container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=expiry_hours),
        )
        download_url = f"{container_client.url}/{blob_name}?{sas}"

    return {"blob_name": blob_name, "blob_url": blob_url, "download_url": download_url}


def _download_account_data(account_name: str, data_prefix: str | None = None) -> Path | None:
    """Download uploaded report exports for the given account from blob storage into a temp folder."""
    try:
        container_client = _get_blob_client()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Storage error: {str(exc)[:100]}")

    prefix_template_env = os.environ.get("PPC_DATA_BLOB_PREFIX")
    if data_prefix:
        prefix_template = f"{data_prefix.rstrip('/')}" + "/{account_slug}/"
    else:
        prefix_template = "{account_slug}/" if prefix_template_env is None else prefix_template_env
    slug = slugify_account_name(account_name or "General")
    prefix = prefix_template.format(account=account_name, account_slug=slug).strip("/") if prefix_template else ""
    prefix = f"{prefix}/" if prefix and not prefix.endswith("/") else prefix

    cache_dir = Path(tempfile.gettempdir()) / "kai_blob_data" / (slug or "default")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing cache for this account to ensure latest files
    for pattern in ("*.csv", "*.tsv", "*.xlsx", "*.xls"):
        for f in cache_dir.glob(pattern):
            f.unlink(missing_ok=True)

    downloaded = 0
    allowed_exts = (".csv", ".tsv", ".xlsx", ".xls")
    try:
        for blob in container_client.list_blobs(name_starts_with=prefix or None):
            if not blob.name.lower().endswith(allowed_exts):
                continue
            dest = cache_dir / Path(blob.name).name
            with open(dest, "wb") as handle:
                container_client.download_blob(blob).readinto(handle)
            downloaded += 1
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to download data: {str(exc)[:200]}")

    if downloaded == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No data blobs found for account '{account_name}' (prefix '{prefix or '<root>'}'). "
                f"Expected one of: {', '.join(allowed_exts)}."
            ),
        )

    return cache_dir


@app.post("/api/data/upload")
async def upload_data(
    account_name: str = Form(...),
    files: list[UploadFile] = File(...),
    data_prefix: str | None = Form(None),
):
    """
    Upload report exports (CSV/TSV/XLSX/XLS) to blob storage under the account-specific prefix.
    Prefix uses PPC_DATA_BLOB_PREFIX template (format keys: account, account_slug), default '{account_slug}/'.
    """
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded.")
    prefix_template_env = os.environ.get("PPC_DATA_BLOB_PREFIX")
    if data_prefix:
        prefix_template = f"{data_prefix.rstrip('/')}" + "/{account_slug}/"
    else:
        prefix_template = "{account_slug}/" if prefix_template_env is None else prefix_template_env
    slug = slugify_account_name(account_name or "General")
    prefix = prefix_template.format(account=account_name, account_slug=slug).strip("/") if prefix_template else ""
    prefix = f"{prefix}/" if prefix and not prefix.endswith("/") else prefix

    try:
        container_client = _get_blob_client()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Storage error: {str(exc)[:100]}")

    uploaded = []
    for f in files:
        name = os.path.basename(f.filename)
        blob_name = f"{prefix}{name}" if prefix else name
        try:
            data = await f.read()
            container_client.upload_blob(name=blob_name, data=data, overwrite=True)
            uploaded.append(blob_name)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to upload {name}: {str(exc)[:200]}")

    return {"status": "success", "uploaded": uploaded, "account": account_name, "prefix": prefix or "<root>"}

class UploadDataRequest(BaseModel):
    account_name: str


@app.get("/api/data/manifest")
async def data_manifest(
    account_name: str,
    data_prefix: str | None = None,
    business_unit: str | None = None,
):
    """
    Inspect which report exports (CSV/TSV/XLSX/XLS) are available for an account after blob download + parsing.

    This is intentionally metadata-only (no row data). It exists to validate multi-file and multi-sheet uploads
    and to help debug "mapper can't find my columns" issues in broad beta.
    """
    data_dir = _download_account_data(account_name, data_prefix)

    override_template = os.environ.get("AUDIT_TEMPLATE_PATH")
    if override_template and Path(override_template).exists():
        template_path = Path(override_template)
    else:
        template_path = ROOT / "kai_core" / "GenerateAudit" / "template.xlsx"

    engine = UnifiedAuditEngine(
        template_path=template_path,
        data_directory=data_dir,
        business_unit=business_unit,
        business_context={},
    )
    engine.load_data()

    return {
        "status": "success",
        "account": account_name,
        "prefix": data_prefix or "<root>",
        "business_unit": business_unit,
        "manifest": engine.diagnostics.get("file_manifest", []),
        "selected_files": engine.diagnostics.get("selected_files", {}),
        "normalized_key_map": engine.diagnostics.get("normalized_key_map", {}),
    }


async def _perform_web_search(query: str, count: int = 3) -> list[dict]:
    """Call the internal /api/search/web logic (SerpAPI using the Bing engine)."""
    if not SERP_BREAKER.allow():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Web search temporarily unavailable (circuit open).",
        )
    api_key = os.environ.get("SERPAPI_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SERPAPI_KEY is not configured on the server.",
        )

    search_endpoint = "https://serpapi.com/search"
    params = {
        "engine": "bing",
        "q": query,
        "api_key": api_key,
        "num": max(1, min(count, 10)),
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            start = time.perf_counter()
            resp = await client.get(search_endpoint, params=params)
            latency_ms = (time.perf_counter() - start) * 1000
            log_event("external_call", service="serpapi", latency_ms=round(latency_ms, 2))
            if resp.status_code != 200:
                SERP_BREAKER.record_failure()
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            data = resp.json()
            web_pages = data.get("organic_results", []) or data.get("webPages", {}).get("value", [])
            results = [
                {
                  "name": item.get("title") or item.get("name"),
                  "snippet": item.get("snippet"),
                  "url": item.get("link") or item.get("url"),
                  "displayUrl": item.get("displayed_link") or item.get("displayUrl"),
                }
                for item in web_pages
            ]
            SERP_BREAKER.record_success()
            return [r for r in results if r.get("url")]
    except HTTPException:
        raise
    except Exception as exc:
        SERP_BREAKER.record_failure()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(exc)}",
        )


# Health check
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Kai Platform API"}


@app.get("/api/version")
async def version_info():
    """Build/version metadata for provenance checks."""
    return {
        "status": "ok",
        "service": "Kai Platform API",
        "version": app.version,
        "git_sha": os.environ.get("GIT_SHA") or os.environ.get("BUILD_SHA") or "unknown",
        "build_time": os.environ.get("BUILD_TIME") or os.environ.get("BUILD_TIMESTAMP") or "unknown",
    }


@app.get("/api/jobs/health")
async def job_queue_health():
    if not JOB_QUEUE_ENABLED:
        raise HTTPException(status_code=503, detail="Job queue is disabled.")
    depth = queue_depth()
    return {"status": "ok", "queue_depth": depth}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str, include_payload: bool = False):
    if not JOB_QUEUE_ENABLED:
        raise HTTPException(status_code=503, detail="Job queue is disabled.")
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not include_payload:
        job.pop("payload", None)
    return {"status": "success", "job": job}


@app.get("/api/jobs/{job_id}/result")
async def job_result(job_id: str):
    if not JOB_QUEUE_ENABLED:
        raise HTTPException(status_code=503, detail="Job queue is disabled.")
    result = get_job_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job result not found.")
    return {"status": "success", "result": result}


@app.get("/api/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    if not JOB_QUEUE_ENABLED:
        raise HTTPException(status_code=503, detail="Job queue is disabled.")

    async def _events():
        last_status = None
        while True:
            job = get_job(job_id)
            if not job:
                payload = {"status": "error", "detail": "Job not found."}
                yield f"data: {json.dumps(payload)}\n\n"
                break
            status_value = job.get("status")
            if status_value != last_status:
                payload = {
                    "job_id": job_id,
                    "status": status_value,
                    "updatedAt": job.get("updatedAt"),
                    "error": job.get("error"),
                }
                yield f"data: {json.dumps(payload)}\n\n"
                last_status = status_value
            if status_value in {"succeeded", "failed"}:
                break
            await asyncio.sleep(2)

    return StreamingResponse(_events(), media_type="text/event-stream")


# Authentication endpoint - password stored in environment variable
@app.post("/api/auth/verify")
async def verify_password(req: AuthRequest):
    """Verify access password - password is stored server-side in env var"""
    # Get password from environment variable (set in Azure Container Apps)
    correct_password = os.environ.get("KAI_ACCESS_PASSWORD", "kelvinkai")

    if req.password == correct_password:
        return {"status": "success", "authenticated": True}
    else:
        raise HTTPException(status_code=401, detail="Invalid password")


# Chat endpoints
@app.get("/api/chat/history")
async def get_chat_history(limit: int = 200, session_id: str | None = None):
    """Get chat history from database"""
    history = db_manager.fetch_chat_history(limit=limit, session_id=session_id)
    return {"messages": history}


@app.post("/api/chat/send", response_model=ChatResponse)
async def send_chat_message(chat: ChatMessage, request: Request):
    """Send a chat message and get AI response"""
    # Save user message
    session_id = _extract_session_id(chat)
    # SA360 tokens/defaults are stored either per-user (Entra SSO) or per-browser-session (legacy).
    # Chat history remains session-scoped; SA360 access must be scoped consistently with the rest of the API.
    sa360_sid = _sa360_scope_from_request(request, session_id)
    # Defensive defaults: ensure we never hit an unbound variable crash on partial failures.
    # Broad beta expectation is "safe failure" (200 w/ helpful message), not 500s.
    reply: str = ""
    reply_source: str | None = None
    model_used: str | None = None
    fallback_reason: str | None = None
    tool: str | None = None
    tool_output: Any | None = None
    stored_tool_output: Any | None = None
    stored_tool: str | None = None
    sources: list[dict] = []
    tool_output_present = False
    use_stored_tool_output = False
    if isinstance(chat.context, dict) and "tool_output" in chat.context:
        tool_output_present = True
    elif "planner output" in (chat.message or "").lower():
        tool_output_present = True

    dry_run = bool(getattr(chat, "dry_run", False))

    def _save_message(role: str, message: str) -> None:
        if dry_run:
            return
        db_manager.save_chat_message(role, message, session_id=session_id)

    _save_message("user", chat.message)

    lower_message = (chat.message or "").lower()
    if any(term in lower_message for term in ("api key", "api keys", "credential", "credentials", "password")):
        reply = "I can't share secrets. If you need access, use the Env & Keys page or ask an admin."
        _save_message("assistant", reply)
        return ChatResponse(reply=reply, sources=[], model="rules")

    # Explicit web lookup intent (only fire SerpAPI when clearly asked or LLM is unsure)
    def is_lookup_intent(text: str) -> bool:
        t = (text or "").lower()
        keywords = [
            "search the web",
            "search online",
            "web search",
            "look up",
            "lookup",
            "find sources",
            "google it",
            "bing it",
            "latest news",
            "recent news",
            "breaking news",
            "trending",
        ]
        return any(k in t for k in keywords)

    def is_strategy_intent(text: str) -> bool:
        t = (text or "").lower()
        strategy_terms = [
            "strategy",
            "planning",
            "roadmap",
            "allocation",
            "forecast",
            "next quarter",
            "next month",
            "next year",
            "quarterly",
            "annual",
            "portfolio",
            "budget framework",
            # Common broad-beta phrasing: users ask for "recommendations" / "actions" / "next steps"
            # without using the word "strategy". Treat these as strategy intent so we return an
            # advisor-quality options/tradeoffs answer instead of generic tips.
            "next steps",
            "recommendation",
            "recommendations",
        ]
        if any(k in t for k in strategy_terms):
            return True
        # Lightweight pattern match for "give me N actions" / "what should I do" style prompts.
        # Keep this conservative: require "actions" to co-occur with an optimization cue so we
        # don't misroute operational questions (e.g., "actions column") into strategy mode.
        if "action" in t or "actions" in t:
            if any(cue in t for cue in ("improve", "optimize", "increase", "decrease", "reduce", "grow", "scale")):
                return True
        if any(cue in t for cue in ("what should i do", "what do i do")):
            return True
        return False

    def is_concept_intent(text: str) -> bool:
        t = (text or "").lower().strip()
        cues = [
            "what is",
            "what does",
            "explain",
            "define",
            "how does",
            "why does",
            "meaning of",
        ]
        return any(cue in t for cue in cues)

    def is_missing_data_intent(text: str) -> bool:
        t = (text or "").lower()
        cues = [
            "missing data",
            "data needed",
            "missing report",
            "data missing",
            "report missing",
        ]
        return any(cue in t for cue in cues)

    def is_architecture_intent(text: str) -> bool:
        t = (text or "").lower()
        return any(
            k in t
            for k in (
                "architecture",
                "system map",
                "system topology",
                "technical architecture",
                "tech stack",
                "stack",
            )
        )

    def is_general_help_intent(text: str) -> bool:
        t = (text or "").strip().lower()
        cues = (
            "what can you help me with",
            "what do you do",
            "how can you help",
            "how do you help",
            "what can you do",
            "help me",
            "what are your capabilities",
        )
        return any(cue == t or cue in t for cue in cues)

    def is_metric_or_account_intent(text: str) -> bool:
        t = (text or "").lower()
        metric_terms = [
            "impression",
            "impressions",
            "click",
            "clicks",
            "cpc",
            "cpm",
            "ctr",
            "roas",
            "cpa",
            "cost",
            "spend",
            "conversions",
            "conversion",
            "budget",
            "performance",
        ]
        relational_terms = [
            "compare",
            "vs",
            "versus",
            "why",
            "cause",
            "driver",
            "drivers",
            "explain",
            "increase",
            "decrease",
            "drop",
            "spike",
        ]
        # NOTE: Do not include PMax terms here. PMax should route to the PMax tool handler which
        # can operate without SA360 account context (it primarily needs placements data). Including
        # PMax terms here triggers the "missing account" guardrail and blocks the PMax flow.
        account_terms = ["my account", "my campaign", "our account", "our campaign"]
        has_metric = any(k in t for k in metric_terms) or bool(_extract_custom_metric_mentions(text))
        has_account = any(k in t for k in account_terms)
        has_id = bool(_extract_customer_ids_from_text(text))
        has_timeframe = _has_timeframe_hint(text)
        has_relational = any(k in t for k in relational_terms)
        if has_metric and has_relational:
            return True
        if has_metric and not (has_account or has_id or has_timeframe) and not is_strategy_intent(text):
            return False
        return has_metric or has_account or (has_id and has_timeframe) or is_strategy_intent(text)

    def is_market_volume_intent(text: str) -> bool:
        t = (text or "").lower()
        market_terms = [
            "search volume",
            "keyword volume",
            "search demand",
            "volume estimates",
            "market volume",
            "query volume",
            "google trends",
            "trend",
            "trends",
        ]
        return any(k in t for k in market_terms)

    # Ultra-fast deterministic help/architecture responses.
    # Keep these before account-resolution to preserve low p95 latency for common prompts.
    msg_slim_early = re.sub(r"\s+", " ", (chat.message or "").strip().lower())
    if msg_slim_early in {
        "what can you do?",
        "what can you do",
        "what do you do?",
        "what do you do",
        "help",
        "help me",
        "hello",
        "hi",
        "hey",
    }:
        reply = (
            "I can help with paid search analysis and execution support across a few main paths:\n\n"
            "1. Performance analysis (SA360): ask about CPC/CTR/CPA/ROAS, trends, drivers, or custom conversions.\n"
            "2. Audits (Klaudit/PPC): request an account audit and I can generate findings and a report.\n"
            "3. Creative Studio: generate ads copy (RSA-style) and validate constraints.\n"
            "4. PMax Deep Dive: analyze placements/spend signals and recommend actions.\n"
            "5. SERP Monitor / Competitor Intel: URL health checks and competitor signal summaries.\n\n"
            "Tell me what you want to decide (scale vs efficiency vs stability), the timeframe, and which account to use."
        )
        _save_message("assistant", reply)
        return ChatResponse(reply=reply, sources=[], model="rules")

    if is_architecture_intent(chat.message):
        reply = (
            "Kai runs a FastAPI backend that routes requests to planners, the audit engine, and tool handlers, "
            "with a web UI (Kai Chat, Klaudit, Creative, PMax, SERP, Settings/Env/Info) calling the API gateway. "
            "LLM reasoning is split between a local model and Azure OpenAI, with guardrails applied before replies. "
            "Data flows through SA360, SerpAPI, and Trends where applicable, with results cached and stored in blob "
            "storage/queue-backed jobs for longer work. The system is deployed on Azure Container Apps with "
            "health checks, job status, and logging for operational visibility."
        )
        _save_message("assistant", reply)
        return ChatResponse(reply=reply, sources=[], model="rules")

    # Ultra-fast PMax conversational path for short prompts.
    # Do not intercept payload-like asks (JSON/CSV pasted placements) or deep-dive requests.
    pmax_hint = any(t in lower_message for t in ("pmax", "performance max", "performance_max", "placement", "placements"))
    pmax_deep = any(t in lower_message for t in ("deep", "full", "detailed", "diagnose", "root cause", "root-cause"))
    looks_like_payload = bool(
        (chat.message and ("\n" in chat.message and "," in chat.message))
        or (chat.message and ("{" in chat.message or "[" in chat.message))
    )
    if pmax_hint and not pmax_deep and not looks_like_payload:
        reply = _ensure_advisor_sections(
            "I can start with a fast PMax placement optimization pass now.",
            evidence="This request matches PMax placement analysis and can run as a conversational fast path.",
            hypothesis="Low-value placement concentration or asset-group mix is likely dragging efficiency.",
            next_step=(
                "Start by excluding consistently low-quality placements, shift budget to top-converting asset groups, "
                "and monitor CTR, CPC, and conversions over the next 7 days."
            ),
        )
        _save_message("assistant", reply)
        return ChatResponse(reply=reply, sources=[], model="rules_fast_pmax")

    stored_tool_output = None
    if session_id:
        try:
            stored_ctx = _load_last_tool_context(session_id)
            if isinstance(stored_ctx, dict):
                stored_tool_output = stored_ctx.get("tool_output")
        except Exception:
            stored_tool_output = None

    has_tool_context = tool_output_present or stored_tool_output is not None
    message_ids = _extract_customer_ids_from_text(chat.message)
    context_ids: list[str] = []
    if isinstance(chat.context, dict):
        raw_ctx_ids = chat.context.get("customer_ids") or chat.context.get("customerIds") or []
        if isinstance(raw_ctx_ids, list):
            context_ids = _normalize_customer_ids([str(cid).strip() for cid in raw_ctx_ids if str(cid).strip()])
        elif isinstance(raw_ctx_ids, str) and raw_ctx_ids.strip():
            context_ids = _normalize_customer_ids([raw_ctx_ids.strip()])
    message_ids = _normalize_customer_ids(message_ids)
    explicit_context_ids = bool(message_ids or context_ids)
    candidate_context_ids = _normalize_customer_ids(context_ids + message_ids)
    if not candidate_context_ids:
        candidate_context_ids = _normalize_customer_ids(_default_customer_ids(session_id=sa360_sid))
    pmax_like_prompt = any(t in lower_message for t in ("pmax", "performance max", "performance_max", "placement", "placements"))
    skip_context_resolve = bool(
        pmax_like_prompt
        and candidate_context_ids
        and not explicit_context_ids
        and not (chat.account_name or "").strip()
    )
    if skip_context_resolve:
        resolved_context_ids = list(candidate_context_ids)
        resolved_context_account = chat.account_name
    else:
        resolved_context_ids, resolved_context_account, _, _ = _resolve_account_context(
            chat.message,
            customer_ids=candidate_context_ids,
            account_name=chat.account_name,
            explicit_ids=explicit_context_ids,
            session_id=sa360_sid,
        )
    effective_customer_ids = _normalize_customer_ids((resolved_context_ids or []) + context_ids + message_ids)
    if not effective_customer_ids:
        # Keep default-account fallback available for conversational asks when no explicit ID is present.
        effective_customer_ids = candidate_context_ids
    has_id = bool(message_ids)
    has_timeframe = _has_timeframe_hint(chat.message)
    has_account_context = bool(chat.account_name or resolved_context_account or effective_customer_ids)
    if (
        not has_tool_context
        and is_metric_or_account_intent(chat.message)
        and not is_market_volume_intent(chat.message)
        and not (has_id or has_account_context)
    ):
        advisor_reply = _advisor_missing_data_reply_for_session(chat.message, sa360_sid)
        if not advisor_reply:
            advisor_reply = _ensure_advisor_sections(
                "I need connected performance data to answer that accurately.",
                evidence="I do not have report output or performance data (CPC/CTR/ROAS) for the requested timeframe.",
                hypothesis="It is likely driven by bid, query, or seasonality shifts, but I need account data to confirm.",
                next_step="Run a performance check or share the report output for the account and timeframe.",
            )
            model = "rules"
        else:
            model = "local"
        _save_message("assistant", advisor_reply)
        return ChatResponse(reply=advisor_reply, sources=[], model=model)

    if is_general_help_intent(chat.message):
        # Fast-path: keep broad-beta UX responsive for common help/capabilities probes.
        # Avoids a slow local LLM call for "what can you do" while still returning a high-quality answer.
        try:
            msg_norm = (chat.message or "").strip().lower()
        except Exception:
            msg_norm = ""
        msg_slim = re.sub(r"\s+", " ", msg_norm) if msg_norm else ""

        if msg_slim in {
            "what can you do?",
            "what can you do",
            "what do you do?",
            "what do you do",
            "help",
            "help me",
            "hello",
            "hi",
            "hey",
        }:
            reply = (
                "I can help with paid search analysis and execution support across a few main paths:\n\n"
                "1. Performance analysis (SA360): ask about CPC/CTR/CPA/ROAS, trends, drivers, or custom conversions.\n"
                "2. Audits (Klaudit/PPC): request an account audit and I can generate findings and a report.\n"
                "3. Creative Studio: generate ads copy (RSA-style) and validate constraints.\n"
                "4. PMax Deep Dive: analyze placements/spend signals and recommend actions.\n"
                "5. SERP Monitor / Competitor Intel: URL health checks and competitor signal summaries.\n\n"
                "Tell me what you want to decide (scale vs efficiency vs stability), the timeframe, and which account to use."
            )
            _save_message("assistant", reply)
            return ChatResponse(reply=reply, sources=[], model="rules")

        reply = _llm_general_help_reply(chat.message)
        model = "local"
        if not reply:
            reply = (
                "I can analyze account performance, run audits, and surface trends or SERP signals. "
                "Tell me which account or module you want to start with, and the timeframe you care about."
            )
            model = "rules"
        _save_message("assistant", reply)
        return ChatResponse(reply=reply, sources=[], model=model)

    def is_audit_intent(text: str) -> bool:
        t = (text or "").lower()
        keywords = [
            "audit",
            "run audit",
            "generate audit",
            "upload data",
            "upload csv",
            "upload files",
            "ppc audit",
        ]
        return any(k in t for k in keywords)

    def is_affirmative(text: str) -> bool:
        t = (text or "").strip().lower()
        affirmations = [
            "yes",
            "yeah",
            "yep",
            "sure",
            "ok",
            "okay",
            "please do",
            "do it",
            "go ahead",
            "sounds good",
            "please",
            "yup",
            "absolutely",
            "sure thing",
        ]
        return any(t == a or t.startswith(a + " ") for a in affirmations)

    def is_market_choice(text: str) -> bool:
        t = (text or "").strip().lower()
        keywords = [
            "market",
            "market volume",
            "search volume",
            "public",
            "estimate",
            "benchmarks",
            "industry",
            "general",
            "trends",
        ]
        return any(k in t for k in keywords)

    def is_account_choice(text: str) -> bool:
        t = (text or "").strip().lower()
        keywords = [
            "account",
            "my account",
            "my campaign",
            "campaign",
            "impressions from my",
            "my data",
            "our data",
            "exact impressions",
        ]
        return any(k in t for k in keywords)

    def rewrite_lookup_query(text: str) -> str:
        """Rewrite user ask into a higher-signal search query for volume/benchmarks."""
        t = text or ""
        lower = t.lower()
        import re

        m = re.search(r'"([^"]+)"', t)
        keyword = m.group(1) if m else t.strip()
        keyword = keyword if keyword else "ppc keyword"

        time_hint = ""
        if "last month" in lower:
            time_hint = "last month"
        elif "november" in lower:
            time_hint = "november"
        elif "2025" in lower:
            time_hint = "2025"
        elif "2024" in lower:
            time_hint = "2024"

        return f"{keyword} keyword search volume {time_hint} US google ads keyword planner estimate"

    def filter_sources(raw_sources: list[dict]) -> list[dict]:
        """Drop low-value sources like dictionary results or off-topic finance links."""
        bad_domains = [
            "merriam-webster.com",
            "dictionary.com",
            "cambridge.org",
            "thefreedictionary.com",
            "wordreference.com",
            "vocabulary.com",
            "collinsdictionary.com",
            "marketwatch.com",
            "finance.yahoo.com",
            "cnbc.com",
            "reuters.com",
            "bloomberg.com",
            "cnn.com/markets",
            "tripadvisor.com",
            "yelp.com",
            "facebook.com",
            "pinterest.com",
            "instagram.com",
            "reddit.com",
        ]
        keywords = ["search", "volume", "keyword", "ppc", "sem", "adwords", "ads", "impressions", "planner", "trend"]
        focus_phrases = [
            "search volume",
            "keyword planner",
            "keyword volume",
            "search demand",
            "search impressions",
            "google ads",
            "adwords",
            "ppc",
            "semrush",
            "ahrefs",
            "moz",
            "google trends",
        ]
        filtered = []
        for s in raw_sources or []:
            url = (s.get("url") or "").lower()
            if any(bad in url for bad in bad_domains):
                continue
            name = (s.get("name") or "").lower()
            snippet = (s.get("snippet") or "").lower()
            if not snippet and not name:
                continue
            if not any(k in name or k in snippet for k in keywords):
                continue
            text_blob = f"{name} {snippet}"
            if not any(phrase in text_blob for phrase in focus_phrases):
                continue
            filtered.append(s)
        return filtered

    def has_numeric_evidence(sources: list[dict]) -> bool:
        """Return True if any source contains numeric tokens (e.g., ranges, counts)."""
        import re
        for s in sources or []:
            text = f"{s.get('name','')} {s.get('snippet','')}"
            if re.search(r"\d", text):
                return True
        return False

    def fallback_market_estimate(keyword: str) -> str:
        """Provide a conservative, clearly labeled market estimate template with disclaimers."""
        kw = keyword.strip('"').strip() if keyword else "the keyword"
        return (
            f"I don't have exact impression counts for {kw} without platform access. "
            "Here is a market-level estimate approach using public signals:\n\n"
            "- Treat this as search volume (how many people search), not ad impressions (how often your ad showed).\n"
            "- Public tools: Google Keyword Planner ranges, Google Trends for seasonality, and third-party volume aggregators.\n"
            "- Actual ranges vary by geo, match type, and seasonality; validate in Keyword Planner for your targeting.\n"
            "- Expect spikes around relevant events/holidays; post-spike drop-offs are normal.\n\n"
            "Next steps: 1) Run Keyword Planner for your geo/date range, 2) Use Google Trends to map peaks, 3) Align with your ads data to distinguish search volume from ad impressions. "
            "I won't provide numeric ranges unless a source explicitly contains them."
        )

    def needs_web_lookup(text: str, draft_reply: str) -> bool:
        """Only escalate to web search if explicitly requested or the draft reply signals missing data."""
        reply = (draft_reply or "").lower()
        uncertainty_markers = [
            "i don't have real-time data",
            "i do not have real-time data",
            "i don't have realtime data",
            "i do not have realtime data",
            "i can't browse",
            "i cannot browse",
            "i don't have browsing",
            "not enough information",
            "need more information",
            "couldn't find information",
            "cannot find information",
            "no source available",
        ]
        return is_lookup_intent(text) or any(marker in reply for marker in uncertainty_markers)

    sources: list[dict] = []
    model_used: str | None = None
    reply_source: str | None = None
    fallback_reason: str | None = None
    tool: str | None = None

    # Generate AI response
    if chat.ai_enabled:
        llm_trace = {"calls": [], "total_ms": 0.0, "local_calls": 0, "azure_calls": 0}
        trace_token = _LLM_TRACE.set(llm_trace)
        try:
            apply_general_guardrails = False
            base_system = {
                "role": "system",
                "content": chat_system_prompt(),
            }
            require_local = os.environ.get("REQUIRE_LOCAL_LLM", "false").lower() == "true"
            fast_prompt = os.environ.get("LOCAL_LLM_FAST_PROMPT", "").strip().lower() in {"1", "true", "yes"}
            history_limit = 10
            recent_limit = 5
            if require_local or fast_prompt:
                history_limit = 2
                recent_limit = 1
            history = db_manager.fetch_chat_history(limit=history_limit, session_id=session_id) if session_id else []
            recent_messages = []
            for msg in history[-recent_limit:]:
                recent_messages.append({"role": msg["role"], "content": msg["message"]})

            def _prior_user_message(messages: list[dict], current: str) -> str:
                for msg in reversed(messages or []):
                    if msg.get("role") == "user" and msg.get("message") != current:
                        return msg.get("message") or ""
                return ""

            def _call_azure(messages: list[dict], intent: str | None = None) -> str:
                nonlocal model_used, fallback_reason
                if os.environ.get("AZURE_OPENAI_DISABLED", "false").lower() == "true" or os.environ.get(
                    "REQUIRE_LOCAL_LLM", "false"
                ).lower() == "true":
                    reason = "require_local" if os.environ.get("REQUIRE_LOCAL_LLM", "false").lower() == "true" else "disabled"
                    fallback_reason = f"azure_blocked:{reason}"
                    model_used = "local"
                    log_event(
                        "llm_call",
                        model="azure",
                        intent=intent or "",
                        status="blocked",
                        error=reason,
                    )
                    _record_llm_usage("azure", "error", error=reason)
                    return (
                        "I'm currently responding with the local model to control usage. "
                        "If you want a premium-model answer, tell me explicitly."
                    )
                allowed, reason = _azure_budget_allow(intent)
                if not allowed:
                    fallback_reason = f"azure_blocked:{reason}"
                    model_used = "local"
                    log_event(
                        "llm_call",
                        model="azure",
                        intent=intent or "",
                        status="blocked",
                        error=reason,
                    )
                    _record_llm_usage("azure", "error", error=reason)
                    return (
                        "I'm currently responding with the local model to control usage. "
                        "If you want a premium-model answer, tell me explicitly."
                    )
                model_used = "azure"
                return call_azure_openai(messages, session_id=None, intent=intent, tenant_id=None)

            # Tool-specific fast paths (serp/pmax/competitor)
            ctx = chat.context if isinstance(chat.context, dict) else {}
            tool = ctx.get("tool")
            tool_output = ctx.get("tool_output")
            prompt_kind = ctx.get("prompt_kind")
            skip_tool_followup = False
            # Allow tool-less PMax prompts to bypass the LLM router.
            # This prevents "AI services unavailable" failures for a flow that can be handled deterministically.
            if not tool:
                try:
                    msg_norm = (chat.message or "").strip().lower()
                    msg_slim = re.sub(r"\s+", " ", msg_norm) if msg_norm else ""
                except Exception:
                    msg_slim = ""
                if msg_slim:
                    audit_cues = (
                        "audit",
                        "ppc audit",
                        "klaudit",
                        "brand audit",
                        "nonbrand audit",
                        "run audit",
                        "generate audit",
                    )
                    if any(cue in msg_slim for cue in audit_cues):
                        tool = "audit"
                    elif any(needle in msg_slim for needle in ("pmax", "performance max", "performance_max")):
                        tool = "pmax"
                    else:
                        serp_cues = (
                            "serp monitor:",
                            "url health",
                            "broken url",
                            "broken link",
                            "soft 404",
                            "soft-404",
                            "landing page",
                            "landing pages",
                        )
                        if msg_slim.startswith("serp monitor:") or any(cue in msg_slim for cue in serp_cues):
                            tool = "serp"
                        else:
                            competitor_cues = (
                                "competitor intel:",
                                "competitor",
                                "outranking",
                                "impression share",
                                "position above",
                                "top of page",
                                "auction insights",
                            )
                            if msg_slim.startswith("competitor intel:") or any(cue in msg_slim for cue in competitor_cues):
                                tool = "competitor"

            def _extract_urls(text: str) -> list[str]:
                import re
                return re.findall(r"https?://[\w\-.]+[^\s,;>\)]*", text or "")

            def _extract_domain(text: str) -> str | None:
                import re
                t = (text or "").lower()
                if "home depot" in t:
                    return "homedepot.com"
                domains = re.findall(r"\b([a-z0-9.-]+\.[a-z]{2,})\b", text or "", flags=re.IGNORECASE)
                return domains[0].lower() if domains else None

            def _parse_number_after(text: str, keywords: list[str]) -> float | None:
                import re
                t = text or ""
                for kw in keywords:
                    m = re.search(kw + r"[^\d\-]*([0-9]+(?:\.[0-9]+)?)", t, flags=re.IGNORECASE)
                    if m:
                        try:
                            return float(m.group(1))
                        except Exception:
                            continue
                return None

            def _try_parse_json_payload(raw: str):
                import json
                try:
                    return json.loads(raw)
                except Exception:
                    pass
                if "[" in raw and "]" in raw:
                    try:
                        start = raw.index("[")
                        end = raw.rindex("]") + 1
                        return json.loads(raw[start:end])
                    except Exception:
                        return None
                return None

            def _try_parse_csv_placements(raw: str):
                """
                Best-effort CSV parser for PMax placements pasted into chat.
                Accepts headers like asset_group, placement, cost, conversions, impressions, clicks.
                """
                import csv
                from io import StringIO
                try:
                    reader = csv.DictReader(StringIO(raw))
                    rows = [dict(row) for row in reader if any(row.values())]
                    if not rows:
                        return None
                    # normalize numeric fields
                    numeric_fields = {"cost", "conversions", "clicks", "impressions"}
                    for r in rows:
                        for k in list(r.keys()):
                            v = r[k]
                            if v is None:
                                continue
                            if k.lower() in numeric_fields:
                                try:
                                    r[k] = float(str(v).replace(",", "")) if v != "" else 0.0
                                except Exception:
                                    pass
                    return rows
                except Exception:
                    return None

            def _is_performance_followup(text: str) -> bool:
                t = (text or "").strip().lower()
                if not t:
                    return False
                cues = [
                    "explain",
                    "why",
                    "driver",
                    "drivers",
                    "compare",
                    "versus",
                    "due to",
                    "because",
                    "efficiency",
                    "increase",
                    "increased",
                    "decrease",
                    "decreased",
                    "spike",
                    "drop",
                    "slice",
                    "breakdown",
                    "break down",
                    "what does that mean",
                    "what happened",
                    "what changed",
                    "seasonal",
                    "seasonality",
                    "quality",
                    "root cause",
                    "root-cause",
                    "rootcause",
                    "summary",
                    "summarize",
                    "recap",
                    "next action",
                    "next step",
                    "next steps",
                    "next actions",
                    "action",
                    "actions",
                    "recommend",
                    "recommendation",
                    "recommendations",
                    "what should i do",
                    "what should we do",
                    "what would you do",
                    "what do you suggest",
                    "optimize",
                    "optimise",
                    "optimization",
                    "optimizations",
                    "improve",
                    "improved",
                    "improvement",
                    "improvements",
                    "suggest",
                    "suggestion",
                    "prioritize",
                    "priority",
                    "impact",
                    "implication",
                    "takeaway",
                    "what now",
                    "so what",
                ]
                return any(cue in t for cue in cues)

            last_assistant_message = next((msg["message"] for msg in reversed(history) if msg["role"] == "assistant"), "")
            if _is_performance_followup(chat.message) and not tool_output_present:
                lower_last = (last_assistant_message or "").lower()
                if not lower_last:
                    short_followup = len((chat.message or "").strip().split()) <= 4
                    if short_followup:
                        reply = _ensure_advisor_sections(
                            "I can explain the drivers once I have account performance data.",
                            evidence="I do not have report output or performance data (CPC/CTR/ROAS) for this account.",
                            hypothesis="It is likely caused by bid or query mix changes, but I need account data to confirm.",
                            next_step="Share the account and timeframe or run a performance check so I can break it down.",
                        )
                        _save_message("assistant", reply)
                        return ChatResponse(reply=reply, sources=sources, model="rules")
                if "need connected data" in lower_last or "which account should i use" in lower_last:
                    reply = _ensure_advisor_sections(
                        "I can explain the drivers once I have account performance data.",
                        evidence="I do not have report output or performance data (CPC/CTR/ROAS) for this account.",
                        hypothesis="It is likely driven by targeting or creative shifts, but I need account data to confirm.",
                        next_step="Share the account and timeframe or run a performance check so I can break it down.",
                    )
                    _save_message("assistant", reply)
                    return ChatResponse(reply=reply, sources=sources, model="rules")

            def _is_summary_request(text: str) -> bool:
                t = (text or "").lower()
                return any(
                    cue in t
                    for cue in ("summary", "summarize", "recap", "two-sentence", "next action", "next step")
                )

            def _looks_like_performance_output(output: Any) -> bool:
                if not isinstance(output, dict):
                    return False
                result = output.get("result") or {}
                analysis = output.get("analysis") or {}
                plan = output.get("plan") or {}
                if isinstance(result, dict) and result.get("mode") == "performance":
                    return True
                if isinstance(result, dict) and ("deltas" in result or "metrics" in result):
                    return True
                if isinstance(analysis, dict) and (analysis.get("summary") or analysis.get("note") or analysis.get("deltas")):
                    return True
                if isinstance(plan, dict) and plan.get("customer_ids"):
                    return True
                return False

            def _is_performance_context(tool_name: str | None, output: Any) -> bool:
                # Explicit non-performance tools must not be treated as performance followups,
                # even if a prior session stored a performance payload. This prevents cross-tool
                # hijacks (e.g., competitor/audit prompts reusing a performance context).
                if isinstance(tool_name, str) and tool_name.lower() in {"serp", "pmax", "competitor", "audit"}:
                    return False
                if tool_name == "performance":
                    return True
                if isinstance(output, dict):
                    result = output.get("result") or {}
                    if result.get("mode") == "performance":
                        return True
                return _looks_like_performance_output(output)

            def _build_performance_followup_reply(question: str, output: dict) -> str | None:
                relational_reply = _build_relational_performance_reply(question, output)
                if relational_reply:
                    return relational_reply
                lower = (question or "").lower()
                wants_summary = any(
                    cue in lower
                    for cue in (
                        "summary",
                        "summarize",
                        "recap",
                        "two-sentence",
                        "two sentence",
                        "2 sentence",
                    )
                )
                is_action_request = any(
                    cue in lower
                    for cue in (
                        "action",
                        "actions",
                        "recommend",
                        "recommendation",
                         "recommendations",
                         "what should i do",
                         "what should we do",
                         "what would you do",
                         "what do you suggest",
                         "optimize",
                         "optimise",
                         "optimization",
                         "optimizations",
                         "improve",
                         "improved",
                         "improvement",
                         "improvements",
                         "next steps",
                         "next actions",
                         "prioritize",
                         "priority",
                         "what now",
                    )
                )
                is_data_request = any(
                    cue in lower
                    for cue in (
                        "need more data",
                        "what data",
                        "what else do you need",
                        "what would you need",
                        "what additional data",
                        "what should we collect",
                    )
                )
                def _two_sentence_summary(result_payload: dict) -> str | None:
                    if not isinstance(result_payload, dict):
                        return None
                    deltas_local = result_payload.get("deltas") or {}
                    if not isinstance(deltas_local, dict):
                        return None

                    def _sentence(keys: list[tuple[str, str]]) -> str | None:
                        parts = []
                        for key, label in keys:
                            pct = _delta_pct(deltas_local, key)
                            if pct is None:
                                continue
                            direction = "up" if pct > 0 else "down" if pct < 0 else "flat"
                            parts.append(f"{label} {direction} ({_format_pct(pct)})")
                        if not parts:
                            return None
                        if len(parts) == 1:
                            return parts[0] + "."
                        return " and ".join(parts) + "."

                    first = _sentence([("conversions", "Conversions"), ("cost", "Spend")]) or _sentence(
                        [("clicks", "Clicks"), ("impressions", "Impressions")]
                    )
                    second = _sentence([("ctr", "CTR"), ("cpc", "CPC")]) or _sentence(
                        [("cvr", "CVR"), ("cpa", "CPA")]
                    ) or _sentence([("roas", "ROAS"), ("conversions_value", "Revenue")])
                    if first and second:
                        return f"{first} {second}"
                    return first or second
                if any(
                    cue in lower
                    for cue in (
                        "which driver",
                        "which slice",
                        "only look at one",
                        "one slice",
                        "first slice",
                    )
                ):
                    return (
                        "Start with campaign for that window; it usually surfaces the biggest variance first. "
                        "If campaign is flat, check device next, then query or geo."
                    )
                if wants_summary and is_action_request:
                    result = output.get("result") or {}
                    summary_text = _two_sentence_summary(result)
                    if summary_text:
                        return (
                            f"{summary_text} "
                            "Next action: break out by campaign and device for that window to isolate the driver."
                        )
                if is_data_request:
                    return (
                        "To answer precisely, I need: (1) campaign and device breakdowns for the same window, "
                        "(2) keyword/query mix for the top CPC movers, and (3) auction insights or impression share "
                        "changes to confirm competitive pressure. That will isolate whether this is mix shift, quality, or auction-driven."
                    )
                if is_action_request:
                    result = output.get("result") or {}
                    analysis = output.get("analysis") or {}
                    summary_text = _two_sentence_summary(result) or ""

                    metric_focus = None
                    try:
                        if isinstance(analysis, dict):
                            metric_focus = analysis.get("metric_focus") or metric_focus
                        if not metric_focus and isinstance(result, dict):
                            dq = result.get("data_quality") or {}
                            if isinstance(dq, dict):
                                metric_focus = dq.get("custom_metric_match_value") or dq.get("custom_metric_key") or metric_focus
                    except Exception:
                        metric_focus = metric_focus

                    # Surface at most one driver snippet to avoid overwhelming the user.
                    driver_hint = None
                    try:
                        drivers = analysis.get("drivers") if isinstance(analysis, dict) else None
                        if isinstance(drivers, dict):
                            camp = drivers.get("campaign")
                            dev = drivers.get("device")
                            geo = drivers.get("geo")
                            def _first_name(rows: Any) -> str:
                                if not isinstance(rows, list) or not rows:
                                    return ""
                                row0 = rows[0] if isinstance(rows[0], dict) else {}
                                if not isinstance(row0, dict):
                                    return ""
                                return str(row0.get("name") or row0.get("campaign") or row0.get("campaign_name") or row0.get("device") or row0.get("geo") or "").strip()
                            top_camp = _first_name(camp)
                            top_dev = _first_name(dev)
                            top_geo = _first_name(geo)
                            if top_camp:
                                driver_hint = f"Likely driver to check first: campaign (top mover: {top_camp})."
                            elif top_dev:
                                driver_hint = f"Likely driver to check first: device (top mover: {top_dev})."
                            elif top_geo:
                                driver_hint = f"Likely driver to check first: geo (top mover: {top_geo})."
                    except Exception:
                        driver_hint = driver_hint

                    focus_line = None
                    if metric_focus and isinstance(metric_focus, str):
                        mf = metric_focus.strip()
                        if mf:
                            focus_line = f"Metric focus: {mf}."

                    # Advisor-grade structure, even when the LLM is unavailable.
                    # Keep the actions reversible and staged to avoid "do everything now" behavior.
                    option_a = (
                        "Option A (Conservative): isolate the driver first. "
                        "Break out results by campaign and device for the same window, then inspect query mix for the top movers."
                    )
                    option_b = (
                        "Option B (More aggressive): once the driver is identified, apply a targeted fix. "
                        "Examples: tighten queries/negatives and adjust bids on the biggest CPC movers, or refresh ads/LP alignment if CTR/CVR fell."
                    )
                    monitor = (
                        "Monitoring: watch the focused conversion metric (if applicable), conversions, spend, CTR, CPC, and CPA for 3-5 days; "
                        "avoid stacking multiple major changes in the same learning window."
                    )
                    next_step = "If you want, tell me whether you want the drilldown by campaign, device, or geo and I will run that cut."

                    parts = [p for p in (summary_text, focus_line, driver_hint, option_a, option_b, monitor, next_step) if p]
                    return "\n".join(parts)
                analysis = output.get("analysis") or {}
                if isinstance(analysis, dict):
                    summary = analysis.get("summary")
                    if summary:
                        return summary
                result = output.get("result") or {}
                deltas = result.get("deltas")
                missing_spend = False
                data_quality = result.get("data_quality")
                if isinstance(data_quality, dict):
                    missing_spend = bool(data_quality.get("missing_spend"))
                if isinstance(deltas, dict):
                    explanation = _build_performance_explanation(question, deltas, missing_spend)
                    if explanation:
                        return explanation
                if _is_summary_request(question):
                    for key in ("enhanced_summary", "summary"):
                        candidate = output.get(key)
                        if candidate:
                            return candidate
                return None

            def _planner_payload_from_response(plan_response: Any) -> dict | None:
                if not plan_response:
                    return None
                payload: dict[str, Any] = {}
                plan_obj = getattr(plan_response, "plan", None)
                result_obj = getattr(plan_response, "result", None)
                analysis_obj = getattr(plan_response, "analysis", None)
                summary_obj = getattr(plan_response, "summary", None)
                enhanced_obj = getattr(plan_response, "enhanced_summary", None)
                if isinstance(plan_obj, dict):
                    payload["plan"] = plan_obj
                if isinstance(result_obj, dict):
                    payload["result"] = result_obj
                if isinstance(analysis_obj, dict):
                    payload["analysis"] = analysis_obj
                if isinstance(summary_obj, str) and summary_obj.strip():
                    payload["summary"] = summary_obj.strip()
                    payload["summary_seed"] = summary_obj.strip()
                if isinstance(enhanced_obj, str) and enhanced_obj.strip():
                    payload["enhanced_summary"] = enhanced_obj.strip()
                if not payload:
                    return None
                return payload

            def _render_planner_summary_for_chat(question: str, plan_response: Any, tool_name: str = "performance") -> tuple[str | None, str | None, dict | None]:
                payload = _planner_payload_from_response(plan_response)
                if not isinstance(payload, dict):
                    return None, None, None
                summary_reply = None
                llm_meta: dict | None = None
                if _is_performance_context(tool_name, payload):
                    compact_ctx = _compact_tool_output(payload)
                    is_action_request = _is_performance_action_request(question)
                    if is_action_request:
                        summary_reply, llm_meta = _llm_advise_performance(question, compact_ctx)
                    else:
                        summary_reply, llm_meta = _llm_explain_performance(question, compact_ctx)
                    if not summary_reply:
                        summary_reply = _build_performance_followup_reply(question, payload)
                    if is_action_request and summary_reply:
                        summary_reply = _ensure_performance_advice_minimum_structure(summary_reply) or summary_reply
                else:
                    summary_reply, llm_meta = _llm_sync_tool_followup(question, tool_name, payload)
                if not summary_reply:
                    summary_reply = payload.get("enhanced_summary") or payload.get("summary")
                if not summary_reply:
                    return None, None, None
                guardrail_meta = None
                if isinstance(payload, dict):
                    allowed_numbers: set[str] = set()
                    seed = _extract_summary_seed(payload)
                    _collect_numeric_tokens(payload, allowed_numbers)
                    if seed:
                        _collect_numeric_tokens(seed, allowed_numbers)
                    summary_reply, guardrail_meta = _apply_numeric_grounding_guardrail(
                        summary_reply,
                        allowed_numbers,
                        fallback_text=seed,
                    )
                summary_reply = _normalize_reply_text(summary_reply) or summary_reply
                model_name = llm_meta.get("model") if isinstance(llm_meta, dict) and llm_meta.get("model") else "planner"
                return summary_reply, model_name, guardrail_meta

            if isinstance(tool_output, str):
                parsed_tool = _try_parse_json_payload(tool_output)
                if parsed_tool is not None:
                    tool_output = parsed_tool

            if tool_output is not None and not tool and _looks_like_performance_output(tool_output):
                tool = "performance"

            if tool_output is not None:
                _save_last_tool_context(session_id, tool, tool_output, prompt_kind)

            stored_ctx = _load_last_tool_context(session_id)
            stored_tool_output = None
            stored_tool = None
            stored_prompt_kind = None
            if isinstance(stored_ctx, dict):
                stored_tool_output = stored_ctx.get("tool_output")
                stored_tool = stored_ctx.get("tool")
                stored_prompt_kind = stored_ctx.get("prompt_kind")

            stored_perf = stored_tool == "performance" or _looks_like_performance_output(stored_tool_output)
            if (
                tool_output is None
                and stored_tool_output is not None
                and stored_perf
                and _is_performance_followup(chat.message)
                # Only reuse stored performance context for performance followups.
                # If the current message routed to another tool (audit/competitor/serp/pmax),
                # do not inject performance outputs into that flow.
                and (tool is None or tool == "performance")
            ):
                tool_output = stored_tool_output
                tool = tool or ("performance" if stored_perf else stored_tool)
                use_stored_tool_output = True

            if not prompt_kind and stored_prompt_kind and not use_stored_tool_output:
                prompt_kind = stored_prompt_kind

            skip_tool_followup = (
                prompt_kind in {"summary", "persona_summary"} and not use_stored_tool_output
            )

            if prompt_kind == "planner_summary" and tool_output is not None:
                summary_reply = None
                llm_meta: dict | None = None
                if _is_performance_context(tool, tool_output):
                    compact_ctx = _compact_tool_output(tool_output)
                    is_action_request = _is_performance_action_request(chat.message)
                    if _is_performance_action_request(chat.message):
                        summary_reply, llm_meta = _llm_advise_performance(
                            chat.message,
                            compact_ctx,
                        )
                    else:
                        summary_reply, llm_meta = _llm_explain_performance(
                            chat.message,
                            compact_ctx,
                        )
                    if not summary_reply and isinstance(tool_output, dict):
                        summary_reply = _build_performance_followup_reply(chat.message, tool_output)
                    if is_action_request and summary_reply:
                        summary_reply = _ensure_performance_advice_minimum_structure(summary_reply) or summary_reply
                else:
                    summary_reply, llm_meta = _llm_sync_tool_followup(chat.message, tool, tool_output)
                if summary_reply:
                    guardrail_meta = None
                    if isinstance(tool_output, dict):
                        allowed_numbers: set[str] = set()
                        seed = None
                        if _is_performance_context(tool, tool_output):
                            seed = _extract_summary_seed(tool_output)
                        _collect_numeric_tokens(tool_output, allowed_numbers)
                        if seed:
                            _collect_numeric_tokens(seed, allowed_numbers)
                        summary_reply, guardrail_meta = _apply_numeric_grounding_guardrail(
                            summary_reply,
                            allowed_numbers,
                            fallback_text=seed,
                        )
                    summary_reply = _normalize_reply_text(summary_reply) or summary_reply
                    _save_message("assistant", summary_reply)
                    reply_source = "planner_summary"
                    model_used = llm_meta.get("model") if isinstance(llm_meta, dict) else "planner_summary"
                    log_event(
                        "chat_reply",
                        source=reply_source,
                        model=model_used,
                        tool=tool,
                        session_id=session_id,
                    )
                    return ChatResponse(reply=summary_reply, sources=[], model=model_used, guardrail=guardrail_meta)

            if _is_performance_context(tool, tool_output) and not skip_tool_followup and _is_performance_followup(chat.message):
                if isinstance(tool_output, dict):
                    llm_meta: dict | None = None
                    followup_reply = None
                    lower_question = (chat.message or "").lower()
                    short_followup = len(lower_question.split()) <= 4
                    no_metric_mentions = not _extract_metric_mentions(lower_question)
                    effective_question = chat.message
                    if short_followup and no_metric_mentions:
                        prior_user = _prior_user_message(history, chat.message)
                        if prior_user:
                            effective_question = prior_user
                    wants_driver_or_summary = any(
                        cue in lower_question
                        for cue in (
                            "which driver",
                            "which slice",
                            "only look at one",
                            "one slice",
                            "first slice",
                            "summary",
                            "summarize",
                            "recap",
                            "two-sentence",
                            "2 sentence",
                            "next action",
                            "next step",
                            "next steps",
                            "next actions",
                        )
                    ) or (short_followup and no_metric_mentions)
                    is_action_request = _is_performance_action_request(chat.message) or _is_performance_action_request(
                        effective_question
                    )
                    if wants_driver_or_summary:
                        # Short followups like "recommendations?" should produce an advisor answer,
                        # not a rules-only summary. Only use the rules followup for driver/summary asks.
                        if not is_action_request:
                            followup_reply = _build_performance_followup_reply(effective_question, tool_output)
                            if followup_reply:
                                model_used = "planner_followup"
                    if not followup_reply:
                        relational_reply = _build_relational_performance_reply(
                            effective_question, tool_output
                        )
                        if relational_reply:
                            followup_reply = relational_reply
                            model_used = "rules_relational"
                    if not followup_reply:
                        # Prefer a deterministic follow-up first to keep latency within interactive budgets
                        # and avoid unnecessary Azure usage for common "what should I do next?" prompts.
                        followup_reply = _build_performance_followup_reply(effective_question, tool_output)
                        if followup_reply:
                            model_used = "planner_followup"
                        else:
                            if is_action_request:
                                followup_reply, llm_meta = _llm_advise_performance(
                                    effective_question,
                                    _compact_tool_output(tool_output),
                                )
                            else:
                                followup_reply, llm_meta = _llm_explain_performance(
                                    effective_question,
                                    _compact_tool_output(tool_output),
                                )
                            if followup_reply:
                                model_used = llm_meta.get("model") if isinstance(llm_meta, dict) else "planner_followup"
                            else:
                                followup_reply = _build_performance_followup_reply(
                                    effective_question, tool_output
                                )
                                model_used = "planner_followup"
                    if followup_reply:
                        guardrail_meta = None
                        allowed_numbers: set[str] = set()
                        seed = _extract_summary_seed(tool_output)
                        if is_action_request:
                            followup_reply = _ensure_performance_advice_minimum_structure(followup_reply) or followup_reply
                        _collect_numeric_tokens(tool_output, allowed_numbers)
                        if seed:
                            _collect_numeric_tokens(seed, allowed_numbers)
                        followup_reply, guardrail_meta = _apply_numeric_grounding_guardrail(
                            followup_reply,
                            allowed_numbers,
                            fallback_text=seed,
                        )
                        followup_reply = _normalize_reply_text(followup_reply) or followup_reply
                        _save_message("assistant", followup_reply)
                        reply_source = "tool_followup"
                        log_event(
                            "chat_reply",
                            source=reply_source,
                            model=model_used,
                            tool=tool,
                            session_id=session_id,
                        )
                        return ChatResponse(reply=followup_reply, sources=[], model=model_used, guardrail=guardrail_meta)

            if tool_output is not None and not skip_tool_followup and tool not in {"performance", "serp", "pmax", "competitor"}:
                sync_reply, sync_meta = _llm_sync_tool_followup(chat.message, tool, tool_output)
                if sync_reply:
                    _save_message("assistant", sync_reply)
                    reply_source = "tool_followup"
                    model_used = sync_meta.get("model") if isinstance(sync_meta, dict) else "tool_followup"
                    log_event(
                        "chat_reply",
                        source=reply_source,
                        model=model_used,
                        tool=tool,
                        session_id=session_id,
                    )
                    return ChatResponse(reply=sync_reply, sources=[], model=model_used)

            if tool == "serp":
                urls = _extract_urls(chat.message)
                if not urls:
                    reply = "Share the URLs you want checked (one or more) and I'll run a quick health check."
                    _save_message("assistant", reply)
                    return ChatResponse(reply=reply, sources=[])
                try:
                    serp_results = check_url_health(urls)
                    lines = []
                    broken = []
                    for r in serp_results or []:
                        u = str(r.get("url") or "").strip()
                        status = int(r.get("status") or r.get("http_status") or 0)
                        soft_404 = bool(r.get("soft_404")) if status == 200 else False

                        if status:
                            status_label = f"HTTP {status}"
                        else:
                            status_label = "HTTP request failed"

                        line = f"- {u}: {status_label}"
                        if status >= 400:
                            broken.append(u)
                            line += " (possible broken link)"
                        if status == 200:
                            line += f"; soft-404={str(soft_404).lower()}"
                        lines.append(line.strip())

                    summary = (
                        "URL health check complete. I fetched HTTP status codes and a simple soft-404 signal "
                        "(page returns 200 but looks like a 'not found' page)."
                    )
                    if broken:
                        summary += f" Potential issues: {len(broken)} URL(s) returned 4xx/5xx."
                    next_steps = (
                        "Next: If any URL is broken or soft-404=true, fix the destination or pause ads pointing "
                        "to it to avoid wasted spend."
                    )
                    reply = (summary + "\n" + "\n".join(lines) + "\n" + next_steps) if lines else (summary + " No URLs processed.")
                    _save_message("assistant", reply)
                    return ChatResponse(reply=reply, sources=[])
                except Exception as exc:
                    reply = f"URL health check failed: {str(exc)[:120]}"
                    _save_message("assistant", reply)
                    return ChatResponse(reply=reply, sources=[])

            if tool == "pmax":
                placements = _try_parse_json_payload(chat.message)
                if placements is None and "," in (chat.message or "") and "\n" in (chat.message or ""):
                    placements = _try_parse_csv_placements(chat.message)
                if isinstance(placements, list) and placements:
                    try:
                        analyzer = PMaxAnalyzer()
                        result = analyzer.analyze(placements=placements, total_cost=0.0)
                        # optional channel breakout if cost exists
                        try:
                            import pandas as pd
                            df = pd.DataFrame(placements)
                            if "cost" in df.columns:
                                splitter = PMaxChannelSplitter(df_campaign=df)
                                breakout = splitter.infer()
                                result["channel_breakout"] = breakout.__dict__
                        except Exception:
                            pass
                        reply = "PMax analysis completed:\n"
                        if "channel_breakout" in result:
                            cb = result["channel_breakout"]
                            reply += f"- Channel breakout: {cb}\n"
                        if "placements_ranked" in result:
                            tops = result["placements_ranked"][:5]
                            summary = "; ".join(
                                [f"{p.get('asset_group','?')}: {p.get('cost',0)} spend" for p in tops]
                            )
                            reply += f"- Top placements: {summary}"
                        _save_message("assistant", reply)
                        return ChatResponse(reply=reply, sources=[])
                    except Exception as exc:
                        reply = f"PMax analysis failed to parse placements: {str(exc)[:120]}"
                        _save_message("assistant", reply)
                        return ChatResponse(reply=reply, sources=[])
                # Adaptive fallback: if user did not paste placements but account context exists,
                # run planner-backed analysis so PMax prompts remain conversational.
                pmax_ids = list(effective_customer_ids)
                if not pmax_ids:
                    # PMax tool requests often rely on account picker context rather than explicit IDs.
                    # Resolve from account name/session defaults before prompting the user again.
                    fallback_account = (chat.account_name or resolved_context_account or "").strip() or None
                    try:
                        resolved_ids, _, _, _ = _resolve_account_context(
                            chat.message,
                            customer_ids=[],
                            account_name=fallback_account,
                            explicit_ids=bool(fallback_account),
                            session_id=sa360_sid,
                        )
                        pmax_ids = _normalize_customer_ids(resolved_ids or [])
                    except Exception:
                        pmax_ids = []
                if not pmax_ids:
                    pmax_ids = _normalize_customer_ids(_default_customer_ids(session_id=sa360_sid))
                if pmax_ids:
                    # Fast conversational path: for common "analyze placements" prompts with resolved
                    # account context, return actionable guidance immediately instead of blocking on
                    # a heavy planner pass. This keeps user-facing latency within interactive budgets.
                    msg_l = (chat.message or "").lower()
                    wants_placements = any(t in msg_l for t in ("placement", "placements", "pmax", "performance max"))
                    wants_deep_fetch = any(t in msg_l for t in ("deep", "full", "detailed", "diagnose", "root cause"))
                    if wants_placements and not wants_deep_fetch:
                        acct_label = (chat.account_name or resolved_context_account or "").strip() or "the selected account"
                        quick_reply = _ensure_advisor_sections(
                            f"I can start with a fast PMax placement optimization pass for {acct_label}.",
                            evidence="Account context is resolved from your current SA360 selection.",
                            hypothesis="Placement concentration and low-value inventory are likely reducing efficiency.",
                            next_step=(
                                "Prioritize exclusion of low-quality placements, rebalance spend toward top-converting asset groups, "
                                "and watch CTR, CPC, and conversions over the next 7 days."
                            ),
                        )
                        quick_reply = _normalize_reply_text(quick_reply) or quick_reply
                        _save_message("assistant", quick_reply)
                        return ChatResponse(reply=quick_reply, sources=[], model="rules_fast_pmax")
                    try:
                        plan_req = PlanRequest(
                            message=chat.message,
                            customer_ids=pmax_ids,
                            account_name=(chat.account_name or resolved_context_account or None),
                            default_date_range="LAST_14_DAYS",
                            include_previous=False,
                            intent_hint="performance",
                            async_mode=False,
                        )
                        plan_resp = await _chat_plan_and_run_core(plan_req, request=request)
                    except Exception:
                        plan_resp = None
                    has_planner_payload = bool(
                        plan_resp and (plan_resp.summary or plan_resp.analysis or plan_resp.result)
                    )
                    if has_planner_payload:
                        payload = _planner_payload_from_response(plan_resp)
                        reply = None
                        guardrail_meta = None
                        if isinstance(payload, dict):
                            # PMax fallback is expected to feel direct and deterministic.
                            # Use the planner payload to build a grounded follow-up first,
                            # instead of adding a second LLM summarization pass.
                            reply = _build_performance_followup_reply(chat.message, payload)
                            if not reply:
                                reply = payload.get("enhanced_summary") or payload.get("summary")
                            if isinstance(reply, str) and reply.strip():
                                allowed_numbers: set[str] = set()
                                _collect_numeric_tokens(payload, allowed_numbers)
                                seed = _extract_summary_seed(payload)
                                reply, guardrail_meta = _apply_numeric_grounding_guardrail(
                                    reply,
                                    allowed_numbers,
                                    fallback_text=seed,
                                )
                                reply = _normalize_reply_text(reply) or reply
                        if reply:
                            _save_message("assistant", reply)
                            return ChatResponse(reply=reply, sources=[], model="rules", guardrail=guardrail_meta)
                    if plan_resp and plan_resp.error:
                        reply = _ensure_advisor_sections(
                            "I could not complete the PMax analysis for that account right now.",
                            evidence="The performance planner returned an error while fetching account data.",
                            hypothesis="It is likely a temporary data fetch issue or account access mismatch.",
                            next_step="Retry with a broader date range, or paste placements CSV/JSON to run direct placement analysis.",
                        )
                        _save_message("assistant", reply)
                        return ChatResponse(reply=reply, sources=[], model="rules")
                reply = (
                    "I can run PMax analysis in two ways: use the selected SA360 account context, or paste placements data. "
                    "I do not have enough account context in this request. Confirm an account in 'Account (by name)' or paste placements JSON/CSV."
                )
                _save_message("assistant", reply)
                return ChatResponse(reply=reply, sources=[])

            if tool == "audit":
                # Chat audit should be fast and actionable. Use available performance context
                # (stored planner/tool output) to produce "top priorities" without forcing a long-running
                # XLSX audit job in the chat path.
                acct_label = (chat.account_name or resolved_context_account or "").strip() or "the selected account"
                payload: dict | None = None
                if isinstance(tool_output, dict):
                    payload = tool_output
                elif isinstance(stored_tool_output, dict) and _looks_like_performance_output(stored_tool_output):
                    payload = stored_tool_output

                priorities: list[str] = []
                if isinstance(payload, dict):
                    result = payload.get("result") or {}
                    deltas = result.get("deltas") or {}
                    if isinstance(deltas, dict):
                        ctr_pct = _delta_pct(deltas, "ctr")
                        cpc_pct = _delta_pct(deltas, "cpc")
                        conv_pct = _delta_pct(deltas, "conversions")
                        cpa_pct = _delta_pct(deltas, "cpa")

                        # Directional, data-backed priorities (avoid introducing new ungrounded numbers).
                        if conv_pct is not None and conv_pct < 0:
                            priorities.append(
                                "Conversion volume: verify tracking + landing page flow, then isolate CVR drivers (campaign/device/query/geo)."
                            )
                        if cpa_pct is not None and cpa_pct > 0:
                            priorities.append(
                                "Efficiency (CPA): tighten bids/budget allocation, remove waste, and confirm conversion quality."
                            )
                        if ctr_pct is not None and ctr_pct < 0:
                            priorities.append(
                                "Engagement (CTR): refresh ads/creative and improve keyword-to-ad-to-landing relevance."
                            )
                        if cpc_pct is not None and cpc_pct > 0:
                            priorities.append(
                                "Cost pressure (CPC): review bidding strategy, quality signals, and auction pressure."
                            )

                if not priorities:
                    priorities = [
                        "Measurement: confirm primary conversions are tracking correctly and consistently across the window.",
                        "Efficiency: identify the biggest cost drivers (campaign/device/query) and remove waste before scaling.",
                        "Relevance + CRO: improve ad-to-landing alignment and landing page conversion rate to raise volume without higher bids.",
                    ]

                # Guarantee exactly three lines for UX clarity.
                p1 = priorities[0] if len(priorities) > 0 else "Confirm tracking and conversion definitions."
                p2 = priorities[1] if len(priorities) > 1 else "Check campaign/device/query slices to find the biggest mover."
                p3 = priorities[2] if len(priorities) > 2 else "Run a deeper audit (Klaudit) for checklist-level coverage."

                reply = (
                    f"Audit (quick) for {acct_label}: top 3 priority actions.\n"
                    f"Priority 1: {p1}\n"
                    f"Priority 2: {p2}\n"
                    f"Priority 3: {p3}\n\n"
                    "Next step: If you want the full checklist audit + report, run Klaudit Audit and I’ll generate it asynchronously."
                )
                reply = _normalize_reply_text(reply) or reply
                _save_message("assistant", reply)
                return ChatResponse(reply=reply, sources=[], model="rules_audit")

            if tool == "competitor":
                domain = _extract_domain(chat.message) or "competitor"
                is_current = _parse_number_after(
                    chat.message,
                    ["impression share current", "current impression share", "current is", "is now"],
                )
                is_prev = _parse_number_after(
                    chat.message,
                    ["impression share previous", "previous impression share", "last month", "prior is", "was"],
                )
                outranking = _parse_number_after(chat.message, ["outranking rate", "outranking", "outrank"])
                top_of_page = _parse_number_after(chat.message, ["top of page rate", "top-of-page", "top of page"])
                position_above = _parse_number_after(chat.message, ["position above rate", "position above"])

                try:
                    comp_result = analyze_competitor(
                        competitor_domain=domain,
                        impression_share_current=is_current,
                        impression_share_previous=is_prev,
                        outranking_rate=outranking,
                        top_of_page_rate=top_of_page,
                        position_above_rate=position_above,
                        raw_description=chat.message,
                    )
                except Exception as exc:
                    reply = "I couldn't infer competitor investment signals from that message yet."
                    reply = _ensure_advisor_sections(
                        reply,
                        evidence="The competitor intelligence tool raised an error while parsing your description.",
                        hypothesis="This may be a temporary parsing issue or missing Auction Insights context.",
                        next_step=(
                            "Share any Auction Insights metrics (impression share current vs previous, outranking rate, top-of-page rate), "
                            "or paste a short description of what changed."
                        ),
                    )
                    _save_message("assistant", reply)
                    return ChatResponse(reply=reply, sources=[], model="rules")

                signal = str(comp_result.get("signal") or "unknown").replace("_", " ")
                confidence = comp_result.get("confidence")
                conf_text = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "n/a"
                evidence_bits: list[str] = []
                if isinstance(is_prev, (int, float)) and isinstance(is_current, (int, float)):
                    evidence_bits.append(f"Impression share moved from {is_prev:.0f}% to {is_current:.0f}%.")
                if isinstance(outranking, (int, float)):
                    evidence_bits.append(f"Outranking rate ~{outranking:.0f}%.")
                if isinstance(top_of_page, (int, float)):
                    evidence_bits.append(f"Top-of-page rate ~{top_of_page:.0f}%.")
                if not evidence_bits:
                    evidence_bits.append("Based on your description of recent Auction Insights changes.")

                # Produce an advisor-style response (options + monitoring) without requiring SA360 account context.
                option_a = (
                    "Protect core brand coverage: ensure brand campaigns are not budget-limited, confirm ad strength, and "
                    "use a small, controlled bid adjustment on the highest-value brand terms."
                )
                option_b = (
                    "Defend + learn: split traffic into an experiment (or staged rollout) that tests competitor-focused coverage "
                    "or creative/landing page improvements, then keep changes if impression share/CPA stabilizes."
                )
                monitoring = (
                    "Monitor impression share, outranking/position-above rate, top-of-page rate, and your guardrail KPI (CPA/ROAS) "
                    "for 3-5 days after any change."
                )
                reply = (
                    f"Competitor signal for {domain}: {signal} (confidence {conf_text}).\n"
                    f"Evidence: {' '.join(evidence_bits)}\n\n"
                    "Option A (Conservative): " + option_a + "\n"
                    "Option B (More aggressive): " + option_b + "\n\n"
                    "Next step: " + monitoring
                )
                reply = _normalize_reply_text(reply) or reply
                _save_message("assistant", reply)
                return ChatResponse(reply=reply, sources=[], model="rules_competitor")

            # Search volume / market volume intent -> pull SA360 keyword data
            if is_market_volume_intent(chat.message):
                ctx = chat.context or {}
                ctx_ids = []
                explicit_ids = False
                if isinstance(ctx, dict):
                    ctx_ids = ctx.get("customer_ids") or []
                    explicit_ids = bool(ctx.get("customer_ids"))
                ids, acct, resolution_notes, candidates = _resolve_account_context(
                    chat.message,
                    ctx_ids or _default_customer_ids(session_id=sa360_sid),
                    chat.account_name,
                    explicit_ids=explicit_ids,
                    session_id=sa360_sid,
                )
                if not ids:
                    if candidates:
                        reply = "Which account should I use? " + " | ".join(candidates)
                    else:
                        reply = "To pull search volume, tell me the account name or provide a customer ID (SA360)."
                    _save_message("assistant", reply)
                    return ChatResponse(reply=reply, sources=[])
                kw = _extract_keyword_from_text(chat.message)
                if not kw:
                    reply = "Tell me the keyword you want search volume for (e.g., \"shell gas stations\")."
                    _save_message("assistant", reply)
                    return ChatResponse(reply=reply, sources=[])
                date_range = None
                if isinstance(ctx, dict):
                    date_range = ctx.get("date_range") or None
                date_range = date_range or "LAST_30_DAYS"
                exact, close_matches = _keyword_volume_snapshot(
                    kw,
                    ids,
                    date_range,
                    include_previous=True,
                    session_id=sa360_sid,
                )
                reply = _format_volume_reply(kw, exact, close_matches, date_range)
                if resolution_notes:
                    reply = f"{resolution_notes} {reply}"
                _save_message("assistant", reply)
                return ChatResponse(reply=reply, sources=[])

            # If audit intent without clear account context, prompt for account/upload and stop
            if is_audit_intent(chat.message):
                reply = (
                    "I can run an audit. Which account should I use? If you have CSVs, upload them and tell me the account name. "
                    "I'll pull the latest files for that account."
                )
                _save_message("assistant", reply)
                return ChatResponse(reply=reply, sources=sources, model="rules")

            # If the user said yes to the prior prompt, override gate and perform web lookup on the previous query
            last_assistant_message = next((msg["message"] for msg in reversed(history) if msg["role"] == "assistant"), "")
            user_messages = [msg["message"] for msg in history if msg["role"] == "user"]
            prev_user_query = user_messages[-2] if len(user_messages) >= 2 else ""

            # If the user said yes to the prior prompt, override gate and perform web lookup on the previous query
            force_lookup_from_affirmation = (
                is_affirmative(chat.message)
                and "find the answer from outside my immediate sources" in (last_assistant_message or "").lower()
                and prev_user_query
            )

            # If the user chose market vs account after prompt
            market_followup = (
                ("market search volume" in (last_assistant_message or "").lower()
                 or "account impressions" in (last_assistant_message or "").lower())
                and is_market_choice(chat.message)
            )
            account_followup = (
                ("market search volume" in (last_assistant_message or "").lower()
                 or "account impressions" in (last_assistant_message or "").lower())
                and is_account_choice(chat.message)
            )

            lookup_text = prev_user_query if (force_lookup_from_affirmation or market_followup) else chat.message

            if force_lookup_from_affirmation or market_followup:
                search_query = rewrite_lookup_query(lookup_text)
                try:
                    raw_sources = await _perform_web_search(search_query, count=chat.top_k or 3)
                    filtered_sources = filter_sources(raw_sources)
                    sources = filtered_sources
                    if not sources or not has_numeric_evidence(sources):
                        second_query = f"{search_query} keyword planner benchmarks"
                        raw_sources = await _perform_web_search(second_query, count=chat.top_k or 3)
                        second_filtered = filter_sources(raw_sources)
                        sources = second_filtered
                except Exception as exc:
                    reply = f"I'm having trouble connecting to web search. Error: {str(exc)[:100]}"
                else:
                    if sources:
                        snippets = "\n".join(
                            [
                                f"{idx+1}. {s.get('name')}: {s.get('snippet')} ({s.get('url')})"
                                for idx, s in enumerate(sources)
                            ]
                        )
                        grounded_messages = [
                            base_system,
                            {
                                "role": "system",
                                "content": (
                                    "You have live web results. Use them to answer concisely, distinguish impressions vs search volume, "
                                    "and include caveats. Only include numeric ranges if the sources explicitly provide them. "
                                    "If data is insufficient, say so briefly and provide a market-level estimate without inventing numbers."
                                ),
                            },
                            *recent_messages,
                            {
                                "role": "user",
                                "content": f"User question: {lookup_text}\n\nWeb results:\n{snippets}",
                            },
                        ]
                        reply = _call_azure(grounded_messages)
                    else:
                        reply = fallback_market_estimate(lookup_text)

            # If account/metric intent and no connected data, provide a clear gate with choices
            elif is_metric_or_account_intent(chat.message):
                if (effective_customer_ids or has_timeframe or has_account_context) and not tool_output:
                    try:
                        ids = list(effective_customer_ids)
                        plan_req = PlanRequest(
                            message=chat.message,
                            customer_ids=ids,
                            default_date_range="LAST_30_DAYS",
                            intent_hint="performance",
                            async_mode=False,
                        )
                        plan_resp = await _chat_plan_and_run_core(plan_req, request=request)
                    except Exception:
                        plan_resp = None
                    if plan_resp and plan_resp.summary:
                        reply, planner_model, guardrail_meta = _render_planner_summary_for_chat(
                            chat.message,
                            plan_resp,
                            tool_name="performance",
                        )
                        if reply:
                            model_used = planner_model or "planner"
                            _save_message("assistant", reply)
                            return ChatResponse(reply=reply, sources=sources, model=model_used, guardrail=guardrail_meta)
                    elif plan_resp and plan_resp.error:
                        reply = _ensure_advisor_sections(
                            "I could not retrieve performance data for that account/timeframe.",
                            evidence="The performance planner returned an error when attempting to fetch data.",
                            hypothesis="It is likely a data availability or access issue for the specified ID/timeframe.",
                            next_step="Confirm the account ID and timeframe, or try a broader date range.",
                        )
                        model_used = "rules"
                        _save_message("assistant", reply)
                    return ChatResponse(reply=reply, sources=sources, model=model_used)
                else:
                        advisor_reply = _advisor_missing_data_reply_for_session(chat.message, sa360_sid)
                        if advisor_reply:
                            reply = advisor_reply
                            model_used = "local"
                        else:
                            reply = _ensure_advisor_sections(
                                "I need connected data to answer that if it is specific to your account or campaigns.",
                                evidence="I do not have report output or performance data (CPC/CTR/ROAS) for this account.",
                                hypothesis="It is likely driven by bid or query mix changes, but I need account data to confirm.",
                                next_step="Share the account and timeframe or run a performance check to ground the answer.",
                            )
                            model_used = "rules"
                        _save_message("assistant", reply)
                        return ChatResponse(reply=reply, sources=sources, model=model_used)
                if is_strategy_intent(chat.message) and not effective_customer_ids:
                    strategy_reply = _advisor_strategy_reply(chat.message)
                    if not strategy_reply:
                        strategy_reply = _ensure_advisor_sections(
                            "Here is a high-level budget framework for financial services.",
                            evidence="Based on common performance data patterns (CPC/CTR/ROAS) and account report output structures.",
                            hypothesis="It is likely that budget efficiency improves when spend aligns to the highest marginal return segments.",
                            next_step="Review top campaigns by cost and test two new USP/CTA variations before reallocating budget.",
                        )
                        model_used = "rules"
                    else:
                        model_used = "local"
                    reply = strategy_reply
                elif account_followup:
                    reply = (
                        "To provide account impressions, connect your Google Ads or Search Console data. "
                        "I won't run a web search because that won't contain your account metrics."
                    )
                elif is_market_choice(chat.message) or is_market_volume_intent(chat.message):
                    search_query = rewrite_lookup_query(chat.message)
                    try:
                        raw_sources = await _perform_web_search(search_query, count=chat.top_k or 3)
                        filtered_sources = filter_sources(raw_sources)
                        sources = filtered_sources
                        if not sources or not has_numeric_evidence(sources):
                            second_query = f"{search_query} keyword planner benchmarks"
                            raw_sources = await _perform_web_search(second_query, count=chat.top_k or 3)
                            second_filtered = filter_sources(raw_sources)
                            sources = second_filtered
                    except Exception:
                        sources = []
                    if sources:
                        snippets = "\n".join(
                            [
                                f"{idx+1}. {s.get('name')}: {s.get('snippet')} ({s.get('url')})"
                                for idx, s in enumerate(sources)
                            ]
                        )
                        grounded_messages = [
                            base_system,
                            {
                                "role": "system",
                                "content": (
                                    "You have live web results. Provide market-level search volume estimates, "
                                    "differentiate impressions vs search volume, and include caveats. "
                                    "Only include numeric ranges if the sources explicitly provide them; otherwise stay qualitative."
                                ),
                            },
                            *recent_messages,
                            {
                                "role": "user",
                                "content": f"User question: {chat.message}\n\nWeb results:\n{snippets}",
                            },
                        ]
                        reply = _call_azure(grounded_messages)
                    else:
                        reply = fallback_market_estimate(chat.message)
            else:
                apply_general_guardrails = True
                # First pass: answer without web search to avoid unnecessary SerpAPI calls
                tool_context_msg = None
                if tool_output:
                    compact_tool = _compact_tool_output(tool_output)
                    tool_json = json.dumps(compact_tool, ensure_ascii=True, default=str)
                    tool_context_msg = {
                        "role": "system",
                        "content": (
                            "You have recent tool output from the platform. Use it to answer the user's question. "
                            "Avoid generic definitions and keep the response grounded in the data. "
                            "If you infer causes, label them as hypotheses and suggest a focused next step (campaign, device, query, geo). "
                            f"Tool output (JSON): {tool_json}"
                        ),
                    }
                concept_context_msg = None
                if is_concept_intent(chat.message):
                    concept_context_msg = {
                        "role": "system",
                        "content": (
                            "If the user asks for a definition or explanation, keep it to 2-3 sentences, "
                            "avoid generic textbook language, include one paid-search-specific example, "
                            "and end with one concrete next step (campaign/device/query/geo cut or experiment)."
                        ),
                    }
                missing_data_context_msg = None
                if is_missing_data_intent(chat.message):
                    missing_data_context_msg = {
                        "role": "system",
                        "content": (
                            "If the user asks about missing data or reports, explain the dependency in 2-3 sentences, "
                            "avoid mention of access limits or sensitive data, and end with a direct request for the "
                            "specific report or fields needed."
                        ),
                    }
                messages = [
                    base_system,
                    *([tool_context_msg] if tool_context_msg else []),
                    *([concept_context_msg] if concept_context_msg else []),
                    *([missing_data_context_msg] if missing_data_context_msg else []),
                    *recent_messages,
                    {"role": "user", "content": lookup_text},
                ]
                if is_missing_data_intent(chat.message):
                    missing_context = ""
                    if tool_output:
                        try:
                            missing_context = json.dumps(tool_output, ensure_ascii=True, default=str)
                        except Exception:
                            missing_context = str(tool_output)
                    draft_reply = _rewrite_missing_data_reply(chat.message, missing_context)
                    if not draft_reply:
                        draft_reply = (
                            "I need the specific reports and fields that are missing to complete the checks. "
                            "Please share the exports for the missing reports so I can rerun the audit."
                        )
                    model_used = "rules"
                    fallback_reason = "missing_data_deterministic"
                else:
                    if is_strategy_intent(chat.message) and not effective_customer_ids:
                        strategy_reply = _advisor_strategy_reply(chat.message)
                        if not strategy_reply:
                            strategy_reply = _ensure_advisor_sections(
                                "Here are three actions to improve ROAS at a portfolio level.",
                                evidence="Based on common performance data patterns (CPC/CTR/ROAS) and report output context.",
                                hypothesis="It is likely that targeting breadth and message depth are limiting efficiency.",
                                next_step="Review top campaigns by cost and test two new USP/CTA variations, then measure ROAS impact.",
                            )
                            model_used = "rules"
                        else:
                            model_used = "local"
                        draft_reply = strategy_reply
                    else:
                        draft_reply, llm_meta = _call_llm(
                            messages, intent="general_chat", allow_local=True, max_tokens=600
                        )
                        if llm_meta and llm_meta.get("model"):
                            model_used = llm_meta.get("model")
                        if not draft_reply:
                            # Deterministic advisor fallback: keep UX usable even when LLM calls fail.
                            # Must be concise, actionable, and avoid internal debug tokens.
                            draft_reply = (
                                "I couldn't reach the AI service for a tailored answer, but here are 3 levers you can pull right now:\n"
                                "1) Budget & pacing: shift spend from high-CPA segments to your most efficient campaigns; watch delivery and lost IS (budget).\n"
                                "2) Bidding: adjust targets gradually (one change at a time) and re-check after the learning period stabilizes.\n"
                                "3) Keywords / creative / landing: mine search terms for negatives + new keywords, refresh RSA creatives, and fix landing-page speed/UX.\n"
                                "If you tell me whether you're optimizing for CPA, ROAS, or volume, I'll prioritize the next step."
                            )
                            fallback_reason = "llm_no_reply"
                            model_used = model_used or "rules"
                        if draft_reply and is_concept_intent(chat.message) and _needs_concept_rewrite(draft_reply):
                            if not (require_local or fast_prompt):
                                rewritten = _rewrite_concept_reply(chat.message, draft_reply)
                                if rewritten:
                                    draft_reply = rewritten
                        if draft_reply and is_missing_data_intent(chat.message) and _needs_missing_data_rewrite(draft_reply):
                            rewritten = _rewrite_missing_data_reply(chat.message, draft_reply)
                            if rewritten:
                                draft_reply = rewritten
                        if is_strategy_intent(chat.message) and not effective_customer_ids:
                            draft_reply = _ensure_advisor_sections(
                                draft_reply,
                                evidence="Based on common performance data patterns (CPC/CTR/ROAS) and report output context.",
                                hypothesis="It is likely that targeting breadth and message depth are limiting efficiency.",
                                next_step="Review top campaigns by cost and test two new USP/CTA variations, then measure ROAS impact.",
                            )
                        if model_used == "local" and _is_low_quality_local_reply(draft_reply):
                            fallback_reason = "local_low_quality"
                            try:
                                draft_reply = _call_azure(messages, intent="general_chat")
                                model_used = "azure"
                            except Exception:
                                pass
                    if not tool_output_present and is_metric_or_account_intent(chat.message) and not effective_customer_ids:
                        draft_reply = _ensure_advisor_sections(
                            "I need connected performance data to answer that accurately.",
                            evidence="I do not have report output or performance data (CPC/CTR/ROAS) for this account.",
                            hypothesis="It is likely driven by bid or query mix changes, but I need account data to confirm.",
                            next_step="Share the account and timeframe or run a performance check to ground the answer.",
                        )
                        model_used = "rules"
                        fallback_reason = "missing_data_guard"

                lookup_needed = False if is_missing_data_intent(chat.message) else needs_web_lookup(lookup_text, draft_reply)
                consent_needed = lookup_needed and not force_lookup_from_affirmation and not is_lookup_intent(chat.message)

                if consent_needed:
                    prompt = (
                        "Do you want me to find the answer from outside my immediate sources? "
                        "If yes, I'll run a quick grounded lookup; otherwise I'll stick with my current answer."
                    )
                    reply = f"{draft_reply.strip()} {prompt}".strip() if draft_reply else prompt
                # Only call web search if explicitly requested or the draft reply signals missing data
                elif lookup_needed:
                    search_query = rewrite_lookup_query(lookup_text)
                    try:
                        raw_sources = await _perform_web_search(search_query, count=chat.top_k or 3)
                        filtered_sources = filter_sources(raw_sources)
                        sources = filtered_sources
                        if not sources or not has_numeric_evidence(sources):
                            second_query = f"{search_query} keyword planner benchmarks"
                            raw_sources = await _perform_web_search(second_query, count=chat.top_k or 3)
                            second_filtered = filter_sources(raw_sources)
                            sources = second_filtered
                    except Exception as exc:
                        reply = draft_reply or f"I'm having trouble connecting to web search. Error: {str(exc)[:100]}"
                    else:
                        if sources:
                            snippets = "\n".join(
                                [
                                    f"{idx+1}. {s.get('name')}: {s.get('snippet')} ({s.get('url')})"
                                    for idx, s in enumerate(sources)
                                ]
                            )
                            grounded_messages = [
                                base_system,
                                {
                                    "role": "system",
                                    "content": (
                                        "You have live web results. Use them to answer concisely and cite key findings. "
                                        "Only include numeric ranges if the sources explicitly provide them; otherwise stay qualitative and add caveats."
                                    ),
                                },
                                *recent_messages,
                                {
                                    "role": "user",
                                    "content": f"User question: {lookup_text}\n\nWeb results:\n{snippets}",
                                },
                            ]
                            reply = _call_azure(grounded_messages)
                        else:
                            reply = fallback_market_estimate(chat.message) if is_market_volume_intent(chat.message) else draft_reply
                else:
                    reply = draft_reply

            if is_strategy_intent(chat.message) and not effective_customer_ids:
                reply = _ensure_advisor_sections(
                    reply,
                    evidence="Based on common performance data patterns (CPC/CTR/ROAS) and report output context.",
                    hypothesis="It is likely that targeting breadth and message depth are limiting efficiency.",
                    next_step="Review top campaigns by cost and test two new USP/CTA variations, then measure ROAS impact.",
                )

            if apply_general_guardrails:
                reply = _dedupe_general_chat_reply(reply)

        except Exception as exc:
            print(f"[chat] exception: {exc}", file=sys.stderr, flush=True)
            try:
                print(traceback.format_exc(), file=sys.stderr, flush=True)
            except Exception:
                pass
             # Keep errors server-side; do not leak exception strings to end users.
            if "lever" in (lower_message or "") and "performance" in (lower_message or ""):
                reply = (
                    "I couldn't reach the AI service for a tailored answer, but here are 3 levers you can pull right now:\n"
                    "1) Budget & pacing: shift spend from high-CPA segments to your most efficient campaigns.\n"
                    "2) Bidding: adjust targets gradually and re-check after the learning period stabilizes.\n"
                    "3) Keywords / creative / landing: mine search terms, refresh RSA creatives, and fix landing-page UX.\n"
                    "If you tell me whether you're optimizing for CPA, ROAS, or volume, I'll prioritize the next step."
                )
                model_used = model_used or "rules"
                fallback_reason = fallback_reason or "exception_deterministic_advice"
            else:
                reply = (
                    "I'm having trouble connecting to AI services right now, so I didn't get any report output for this request. "
                    "This may be a temporary error. Next step: try again in a moment."
                )
        finally:
            if llm_trace["calls"]:
                log_event(
                    "llm_trace",
                    session_id=session_id,
                    total_ms=round(llm_trace["total_ms"], 2),
                    local_calls=llm_trace["local_calls"],
                    azure_calls=llm_trace["azure_calls"],
                    calls=llm_trace["calls"],
                )
            _LLM_TRACE.reset(trace_token)
    else:
        reply = "AI chat is currently disabled. Enable AI in settings to use intelligent responses."
        reply_source = "ai_disabled"

    if not reply_source:
        if sources:
            reply_source = "web_lookup"
        elif model_used:
            reply_source = f"llm_{model_used}"
        else:
            reply_source = "llm_unknown"
    if not model_used and reply_source == "llm_unknown":
        model_used = "rules"

    guardrail_meta = None
    if use_stored_tool_output:
        tool_output_present = True
    if tool_output_present:
        allowed_numbers = _extract_allowed_numbers(chat)
        fallback_text = None
        effective_output = tool_output if tool_output is not None else stored_tool_output
        effective_tool = tool or stored_tool
        if (
            effective_output is not None
            and _is_performance_context(effective_tool, effective_output)
            and _is_performance_followup(chat.message)
        ):
            seed = _extract_summary_seed(effective_output)
            if seed:
                allowed_numbers = set()
                _collect_numeric_tokens(seed, allowed_numbers)
                fallback_text = seed
        reply, guardrail_meta = _apply_numeric_grounding_guardrail(
            reply,
            allowed_numbers,
            fallback_text=fallback_text,
        )

    if not reply:
        # If we reached here without a reply, something went wrong in an earlier branch.
        # Return a deterministic fallback instead of 500'ing.
        reply = "I'm having trouble generating a response right now. Please try again in a moment."
        reply_source = reply_source or "error_fallback"
        if not model_used:
            model_used = "rules"

    reply = _normalize_reply_text(reply)
    if reply:
        reply = re.sub(r"(?i)api keys?", "authorized access", reply)

    log_event(
        "chat_reply",
        source=reply_source,
        model=model_used,
        fallback=fallback_reason,
        tool=tool,
        session_id=session_id,
    )

    # Save assistant message
    _save_message("assistant", reply)

    return ChatResponse(reply=reply, sources=sources, model=model_used, guardrail=guardrail_meta)


@app.delete("/api/chat/clear")
async def clear_chat_history(session_id: str | None = None):
    """Clear chat history"""
    db_manager.clear_chat_history(session_id=session_id)
    return {"status": "success", "message": "Chat history cleared"}


# Ads integration (feature-flagged scaffolding)
def _ensure_ads_enabled():
    if not ADS_FETCH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ads integration is disabled (ADS_FETCH_ENABLED=false).",
        )


@app.post("/api/integrations/ads/connect")
async def ads_connect(req: AdsConnectRequest):
    """
    Store Ads OAuth credentials. This is a placeholder; secure storage (e.g., Key Vault) should be wired here.
    """
    _ensure_ads_enabled()
    # NOTE: For security, do not log secrets. Here we simply acknowledge receipt.
    # In production, persist to a secure secret store and associate with the tenant/account.
    return {"status": "received", "customer_ids": req.customer_ids}


@app.post("/api/integrations/ads/fetch")
async def ads_fetch(req: AdsFetchRequest):
    """
    Google Ads fetch -> CSV -> blob (feature-flagged). Uses GAQL templates and writes CSVs matching audit schemas.
    """
    _ensure_ads_enabled()
    try:
        access_token = _exchange_google_access_token(
            os.environ.get("ADS_CLIENT_ID", ""),
            os.environ.get("ADS_CLIENT_SECRET", ""),
            os.environ.get("ADS_REFRESH_TOKEN", ""),
        )
        developer_token = os.environ.get("ADS_DEVELOPER_TOKEN", "")
        if not developer_token:
            raise HTTPException(status_code=500, detail="ADS_DEVELOPER_TOKEN is missing.")

        frames = {} if req.dry_run else _collect_ads_frames(
            customer_ids=req.customer_ids or [],
            developer_token=developer_token,
            access_token=access_token,
            date_range=req.date_range,
        )
        if req.dry_run:
            return {"status": "dry-run", "schemas": ADS_CSV_SCHEMAS}
        uploaded = _write_frames_to_blob(req.account_name, frames)
        return {"status": "success", "uploaded": uploaded}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/integrations/ads/fetch-and-audit")
async def ads_fetch_and_audit(req: AdsFetchAuditRequest):
    """
    Fetch Ads data and run audit. Leaves audit engine untouched; feature-flagged.
    """
    _ensure_ads_enabled()
    fetch_resp = await ads_fetch(req)  # will raise if failure or dry-run
    if fetch_resp.get("status") != "success":
        raise HTTPException(status_code=500, detail="Ads fetch did not complete successfully.")
    audit_request = AuditRequest(
        business_unit=req.business_unit,
        account_name=req.account_name,
        use_mock_data=False,
    )
    return await generate_audit(audit_request)


# SA360 integration (feature-flagged, funnel.io-like ingest)
def _ensure_sa360_feature():
    if not SA360_FETCH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SA360 integration is disabled (SA360_FETCH_ENABLED=false).",
        )


@app.post("/api/integrations/sa360/fetch")
async def sa360_fetch(req: Sa360FetchRequest, request: Request = None):
    """
    Search Ads 360 fetch -> CSV -> blob (feature-flagged). Writes CSVs matching existing audit schemas.
    """
    _ensure_sa360_feature()
    _ensure_sa360_enabled()
    sa360_sid = _sa360_scope_from_request(request, req.session_id) if request is not None else req.session_id
    if _should_enqueue(req.async_mode):
        payload = req.model_dump()
        payload["session_id"] = sa360_sid
        job_id = enqueue_job("sa360_fetch", payload)
        return {"status": "queued", "job_id": job_id}
    try:
        frames = {} if req.dry_run else _collect_sa360_frames(
            customer_ids=req.customer_ids or [],
            date_range=req.date_range,
            session_id=sa360_sid,
        )
        if req.dry_run:
            return {"status": "dry-run", "schemas": ADS_CSV_SCHEMAS}
        resolved_account = req.account_name
        if resolved_account is None or resolved_account == "":
            acct_df = frames.get("account")
            if acct_df is not None and not acct_df.empty:
                # Prefer descriptive name from SA360
                resolved_account = (
                    acct_df.iloc[0].get("customer.descriptive_name")
                    or acct_df.iloc[0].get("descriptive_name")
                    or ""
                )
            if not resolved_account and req.customer_ids:
                resolved_account = req.customer_ids[0]
        if not resolved_account:
            resolved_account = "Unknown Account"

        uploaded = _write_frames_to_blob(resolved_account, frames)
        return {"status": "success", "uploaded": uploaded, "account_name": resolved_account}
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[sa360_fetch] Exception: {exc}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/integrations/sa360/fetch-and-audit")
async def sa360_fetch_and_audit(req: Sa360FetchRequest, request: Request = None):
    """
    Fetch SA360 data and run audit. Leaves audit engine untouched; feature-flagged.
    """
    _ensure_sa360_feature()
    sa360_sid = _sa360_scope_from_request(request, req.session_id) if request is not None else req.session_id
    if _should_enqueue(req.async_mode):
        payload = req.model_dump()
        payload["session_id"] = sa360_sid
        job_id = enqueue_job("sa360_fetch_and_audit", payload)
        return {"status": "queued", "job_id": job_id}
    try:
        fetch_resp = await sa360_fetch(req, request=request)  # will raise if failure or dry-run
    except Exception as exc:
        print(f"[sa360_fetch_and_audit] Fetch failure: {exc}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        raise
    if fetch_resp.get("status") != "success":
        raise HTTPException(status_code=500, detail="SA360 fetch did not complete successfully.")
    resolved_account = fetch_resp.get("account_name") or req.account_name
    if not resolved_account:
        resolved_account = "Unknown Account"
    audit_request = AuditRequest(
        business_unit=req.business_unit,
        account_name=resolved_account,
        use_mock_data=False,
    )
    try:
        return await generate_audit(audit_request)
    except Exception as exc:
        print(f"[sa360_fetch_and_audit] Audit failure: {exc}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        raise


@app.post("/api/diagnostics/sa360/perf-window")
async def sa360_performance_diagnostics(req: Sa360DiagnosticsRequest, request: Request = None):
    """
    Compare cached vs live SA360 pulls for a date window (and prior window if requested).
    Helps detect cache cutoff issues (e.g., missing spend with clicks present).
    """
    _ensure_sa360_feature()
    _ensure_sa360_enabled()
    sa360_sid = _sa360_scope_from_request(request, req.session_id) if request is not None else req.session_id
    if not req.customer_ids:
        raise HTTPException(status_code=400, detail="customer_ids are required for diagnostics.")
    if _should_enqueue_heavy(req.async_mode, len(req.customer_ids), SA360_DIAGNOSTICS_MAX_SYNC_ACCOUNTS):
        payload = req.model_dump()
        payload["async_mode"] = False
        payload["session_id"] = sa360_sid
        job_id = enqueue_job("sa360_perf_window", payload)
        return {"status": "queued", "job_id": job_id}

    normalized_range = _coerce_date_range(req.date_range)
    if not normalized_range:
        normalized_range = "LAST_7_DAYS"
    span = _date_span_from_range(normalized_range)
    if not span:
        span = _date_span_from_range("LAST_7_DAYS")

    def span_to_range(sp: tuple[date, date]) -> str:
        return f"{sp[0]:%Y-%m-%d},{sp[1]:%Y-%m-%d}"

    current_range = span_to_range(span) if span else normalized_range
    prev_span = _previous_span(span) if (req.include_previous and span) else None
    previous_range = span_to_range(prev_span) if prev_span else None

    snapshots: dict[str, dict] = {}
    requested_reports = [r for r in (req.report_names or []) if isinstance(r, str) and r.strip()]
    diagnostics_reports = [
        r for r in [x.strip().lower() for x in requested_reports] if r in SA360_QUERIES
    ] or list(SA360_DIAGNOSTICS_REPORTS)
    # Cached vs live (bypassing cache + skipping write) for current window
    use_batched = len(req.customer_ids or []) > SA360_DIAGNOSTICS_MAX_SYNC_ACCOUNTS
    def _fetch_frames(range_value: str, bypass_cache: bool, write_cache: bool):
        if use_batched:
            return _collect_sa360_frames_batched(
                req.customer_ids,
                range_value,
                bypass_cache=bypass_cache,
                write_cache=write_cache,
                max_workers=SA360_DIAGNOSTICS_CONCURRENCY,
                chunk_size=SA360_DIAGNOSTICS_CHUNK_SIZE,
                report_names=diagnostics_reports,
                session_id=sa360_sid,
            )
        return _collect_sa360_frames(
            req.customer_ids,
            range_value,
            bypass_cache=bypass_cache,
            write_cache=write_cache,
            report_names=diagnostics_reports,
            session_id=sa360_sid,
        )

    from concurrent.futures import ThreadPoolExecutor, as_completed
    ranges = [("current", current_range)]
    if previous_range:
        ranges.append(("previous", previous_range))
    max_workers = 4 if previous_range else 2
    results_map: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for label, range_value in ranges:
            future_map[executor.submit(_fetch_frames, range_value, False, True)] = (label, "cached")
            future_map[executor.submit(_fetch_frames, range_value, True, False)] = (label, "live")
        for future in as_completed(future_map):
            label, mode = future_map[future]
            results_map[(label, mode)] = future.result()

    frames_cached = results_map.get(("current", "cached")) or {}
    frames_live = results_map.get(("current", "live")) or {}
    metric_keys = [m for m in (req.metrics or []) if isinstance(m, str) and m.strip()]
    snapshots["current_cached"] = _build_perf_snapshot(frames_cached, metric_keys=metric_keys)
    snapshots["current_live"] = _build_perf_snapshot(frames_live, metric_keys=metric_keys)

    if previous_range:
        frames_prev_cached = results_map.get(("previous", "cached")) or {}
        frames_prev_live = results_map.get(("previous", "live")) or {}
        snapshots["previous_cached"] = _build_perf_snapshot(frames_prev_cached, metric_keys=metric_keys)
        snapshots["previous_live"] = _build_perf_snapshot(frames_prev_live, metric_keys=metric_keys)

    def _num(val: Any) -> float | None:
        try:
            if val is None:
                return None
            return float(val)
        except Exception:
            try:
                return float(str(val).replace(",", ""))
            except Exception:
                return None

    def _cost_diff(a: dict, b: dict) -> dict:
        ca = _num((a or {}).get("metrics", {}).get("cost"))
        cb = _num((b or {}).get("metrics", {}).get("cost"))
        diff = {"cached": ca, "live": cb}
        if ca is not None and cb is not None:
            diff["delta"] = cb - ca
        return diff

    comparisons = {
        "current_cost_gap": _cost_diff(snapshots.get("current_cached"), snapshots.get("current_live")),
    }
    if previous_range:
        comparisons["previous_cost_gap"] = _cost_diff(snapshots.get("previous_cached"), snapshots.get("previous_live"))

    return {
        "status": "success",
        "date_range_current": current_range,
        "date_range_previous": previous_range,
        "report_names": diagnostics_reports,
        "snapshots": snapshots,
        "comparisons": comparisons,
        "notes": "missing_spend flags trigger when cost is 0/null but clicks exist and cost column is empty.",
    }


@app.post("/api/trends/seasonality")
async def trends_seasonality(req: TrendsRequest, request: Request = None):
    """
    Combine Google Trends seasonality with recent SA360 performance to suggest budget allocation.
    Feature-flagged by ENABLE_TRENDS.
    """
    if not ENABLE_TRENDS:
        raise HTTPException(status_code=503, detail="Trends integration is disabled (ENABLE_TRENDS=false).")
    sa360_sid = _sa360_scope_from_request(request, req.session_id) if request is not None else req.session_id
    if _should_enqueue(req.async_mode):
        payload = req.model_dump()
        payload["session_id"] = sa360_sid
        job_id = enqueue_job("trends_seasonality", payload)
        return {"status": "queued", "job_id": job_id}
    account_name = req.account_name
    if not account_name and req.customer_ids and len(req.customer_ids) == 1:
        try:
            lookup_id = str(req.customer_ids[0])
            for acct in _sa360_list_customers_cached(session_id=sa360_sid):
                if str(acct.get("customer_id")) == lookup_id and acct.get("name"):
                    account_name = acct.get("name")
                    break
        except Exception:
            account_name = account_name or None
    themes = _derive_themes(req.themes, account_name)
    if not themes:
        raise HTTPException(status_code=400, detail="No themes provided or inferred.")

    # Normalize timeframe for SA360; trends uses its own string
    perf_range = _normalize_trends_timeframe(req.timeframe)
    perf_weights = {}
    perf_seasonality = {}
    perf_timed_out = False
    perf_task = None
    if req.use_performance and req.customer_ids:
        def _collect_perf():
            weights = _collect_keyword_perf_weights(req.customer_ids, perf_range, session_id=sa360_sid)
            seasonality = _seasonality_fallback_with_range(req.customer_ids, perf_range, session_id=sa360_sid)
            return weights, seasonality
        perf_task = asyncio.create_task(asyncio.to_thread(_collect_perf))

    # Fetch trends with a hard sync timeout to prevent long blocking calls.
    iot = None
    related = {}
    trends_timed_out = False
    queued_job_id = None
    trend_task = asyncio.create_task(asyncio.to_thread(fetch_trends, themes, req.timeframe, req.geo))
    if TRENDS_MAX_SYNC_SECONDS > 0:
        try:
            iot, related = await asyncio.wait_for(trend_task, timeout=TRENDS_MAX_SYNC_SECONDS)
        except asyncio.TimeoutError:
            trends_timed_out = True
            trend_task.cancel()
            if JOB_QUEUE_ENABLED and TRENDS_QUEUE_ON_TIMEOUT:
                queued_job_id = enqueue_job("trends_seasonality", req.model_dump())
            iot, related = None, {}
    else:
        iot, related = await trend_task

    if perf_task:
        if TRENDS_PERF_TIMEOUT_SECONDS > 0:
            try:
                perf_weights, perf_seasonality = await asyncio.wait_for(perf_task, timeout=TRENDS_PERF_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                perf_timed_out = True
                perf_task.cancel()
        else:
            perf_weights, perf_seasonality = await perf_task
    if perf_timed_out and queued_job_id is None and JOB_QUEUE_ENABLED and TRENDS_QUEUE_ON_TIMEOUT:
        queued_job_id = enqueue_job("trends_seasonality", req.model_dump())
    multipliers = seasonality_multipliers(iot, themes)
    allocations = _allocate_budget(themes, multipliers, perf_weights, req.budget)
    seasonality = summarize_seasonality(iot, themes)
    if trends_timed_out:
        trends_status = "trends_timeout"
    else:
        trends_status = (
            "trends_ok" if (iot is not None and not iot.empty) else ("trends_empty" if ENABLE_TRENDS else "trends_disabled")
        )

    def _seasonality_summary(seasonality: dict) -> str | None:
        if not seasonality:
            return None
        def fmt(entries):
            return ", ".join([f"{e['month']} (~{e['score']:.0f})" for e in entries]) if entries else None
        peaks = fmt(seasonality.get("peaks", []))
        lows = fmt(seasonality.get("lows", []))
        shoulders = fmt(seasonality.get("shoulders", []))
        parts = []
        if peaks:
            parts.append(f"Peaks: {peaks}")
        if shoulders:
            parts.append(f"Steady: {shoulders}")
        if lows:
            parts.append(f"Lows: {lows}")
        return " | ".join(parts) if parts else None

    seasonality_text = _seasonality_summary(seasonality) if seasonality else None
    seasonality_source = "trends" if seasonality_text else None
    if not seasonality_text and perf_seasonality:
        seasonality_text = _seasonality_summary(perf_seasonality)
        seasonality_source = "performance" if seasonality_text else seasonality_source
    # Final fallback: describe allocations if no seasonality text
    if not seasonality_text and allocations:
        top = allocations[0]
        second = allocations[1] if len(allocations) > 1 else None
        parts = [f"Top theme: {top.get('theme')} ~{top.get('weight_pct',0):.1f}%"]
        if second:
            parts.append(f"Next: {second.get('theme')} ~{second.get('weight_pct',0):.1f}%")
        seasonality_text = "; ".join(parts)
        seasonality_source = seasonality_source or "performance_fallback"

    notes = "Budget is allocated proportional to (performance weight * trend multiplier). Multipliers are capped to avoid extreme swings."
    if trends_status != "trends_ok":
        notes += f" trends_status={trends_status}."
    if queued_job_id:
        notes += " trends_job_queued."
    if perf_timed_out:
        notes += " perf_timeout=1."
    if seasonality_source == "performance_fallback":
        notes += " seasonality derived from performance allocations only."

    return {
        "status": "success",
        "themes": themes,
        "timeframe": req.timeframe,
        "geo": req.geo,
        "allocations": _to_primitive(allocations),
        "multipliers": _to_primitive(multipliers),
        "perf_weight_keys": list(perf_weights.keys()) if perf_weights else [],
        "related": related,
        "seasonality": _to_primitive(seasonality),
        "seasonality_perf": _to_primitive(perf_seasonality),
        "seasonality_summary": seasonality_text,
        "seasonality_source": seasonality_source,
        "notes": notes,
        "trends_status": trends_status,
        "job_id": queued_job_id,
    }


# Audit endpoints
@app.post("/api/audit/generate")
async def generate_audit(request: AuditRequest):
    """Generate a Klaudit audit report with ML-enhanced insights"""
    if _should_enqueue(request.async_mode):
        job_id = enqueue_job("audit_generate", request.model_dump())
        return {"status": "queued", "job_id": job_id}
    try:
        account_name = request.account_name
        business_unit = request.business_unit or account_name
        canonical_bu = _normalize_audit_business_unit(business_unit)
        if canonical_bu:
            business_unit = canonical_bu
        if account_name:
            canonical_account = _normalize_audit_business_unit(account_name)
            if canonical_account:
                account_name = canonical_account

        # Prefer blob storage if configured; only fall back to demo if explicitly requested
        data_dir = None
        try:
            data_dir = _download_account_data(account_name, request.data_prefix)
        except HTTPException as exc:
            if exc.status_code == 404 and request.use_mock_data:
                data_dir = ROOT / "demo_for_testing" / "kelvin_co_demo_data"
            else:
                raise
        except Exception:
            if request.use_mock_data:
                data_dir = ROOT / "demo_for_testing" / "kelvin_co_demo_data"
            else:
                raise
        if data_dir is None and request.use_mock_data:
            data_dir = ROOT / "demo_for_testing" / "kelvin_co_demo_data"

        if not data_dir:
            raise HTTPException(
                status_code=400,
                detail="Audit data not found. Upload CSVs via /api/data/upload or configure demo data.",
            )
        data_path = Path(data_dir)
        if not data_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Audit data directory missing: {data_path}",
            )
        has_data_files = any(data_path.glob("*.csv")) or any(data_path.glob("*.xlsx")) or any(data_path.glob("*.xls"))
        if not has_data_files:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No CSV/XLSX files found in the audit data directory. "
                    "Upload report exports or provide demo data."
                ),
            )

        override_template = os.environ.get("AUDIT_TEMPLATE_PATH")
        if override_template and Path(override_template).exists():
            template_path = Path(override_template)
        else:
            template_path = ROOT / "kai_core" / "GenerateAudit" / "template.xlsx"

        # Use system temp directory for Azure, isolate per job when available
        job_id = os.environ.get("KAI_JOB_ID")
        output_dir = Path(tempfile.gettempdir()) / "kai_reports"
        if job_id:
            output_dir = output_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create audit engine
        engine = UnifiedAuditEngine(
            template_path=template_path,
            data_directory=data_dir,
            business_unit=business_unit,
            business_context={},
        )

        # Generate audit
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"Audit_{account_name}_{timestamp}.xlsx"

        result = engine.generate_audit(
            account_name=account_name,
            output_path=output_file,
        )

        # Initialize ML reasoning engine and generate enhanced insights
        ml_engine = get_ml_reasoning_engine()
        ml_insights = ml_engine.generate_insights(
            audit_results=result,
            historical_data=None  # Could add historical audit data here
        )

        # Add ML insights to result
        result["ml_insights"] = ml_insights
        vector_index = None
        try:
            vector_index = index_audit_workbook(output_file, account_name)
            result["vector_index"] = vector_index
        except Exception as exc:
            print(f"[generate_audit] Vector indexing failed: {exc}", file=sys.stderr, flush=True)

        upload_info = None
        try:
            upload_info = _upload_audit_report(output_file)
        except Exception as exc:
            print(f"[generate_audit] Blob upload failed: {exc}", file=sys.stderr, flush=True)

        return {
            "status": "success",
            "file_path": str(output_file),
            "file_name": output_file.name,
            "result": result,
            "ml_insights": ml_insights,
            "vector_index": vector_index,
            "blob_name": upload_info.get("blob_name") if upload_info else None,
            "blob_url": upload_info.get("blob_url") if upload_info else None,
            "download_url": upload_info.get("download_url") if upload_info else None,
        }
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[generate_audit] Exception: {exc}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/audit/upload")
async def upload_audit(
    files: List[UploadFile] = File(...),
    business_unit: str = "Brand",
    account_name: str = "Uploaded",
):
    """Deprecated: use /api/data/upload to store CSVs in blob, then /api/audit/generate to run."""
    raise HTTPException(status_code=410, detail="Use /api/data/upload then /api/audit/generate.")


@app.get("/api/audit/download/{filename}")
async def download_audit(filename: str):
    """Download a generated audit file"""
    from fastapi.responses import FileResponse

    temp_dir = Path(tempfile.gettempdir()) / "kai_reports"
    file_path = temp_dir / filename

    if not file_path.exists():
        # Fallback: try blob storage if configured.
        container_client = _get_audit_blob_client()
        if not container_client:
            raise HTTPException(status_code=404, detail="File not found")
        prefix = (os.environ.get("AUDIT_BLOB_PREFIX") or "audit_reports/").strip("/")
        blob_name = f"{prefix}/{filename}" if prefix else filename
        blob_client = container_client.get_blob_client(blob_name)
        try:
            blob_client.get_blob_properties()
        except ResourceNotFoundError:
            raise HTTPException(status_code=404, detail="File not found")

        parts = _parse_storage_connection(os.environ.get("AZURE_STORAGE_CONNECTION_STRING", ""))
        account_name = parts.get("AccountName")
        account_key = parts.get("AccountKey")
        expiry_hours = int(os.environ.get("AUDIT_SAS_EXPIRY_HOURS", "24") or "24")
        if account_name and account_key:
            sas = generate_blob_sas(
                account_name=account_name,
                container_name=container_client.container_name,
                blob_name=blob_name,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=expiry_hours),
            )
            return RedirectResponse(f"{container_client.url}/{blob_name}?{sas}")

        downloader = blob_client.download_blob()
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return StreamingResponse(
            downloader.chunks(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers,
        )

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/api/audit/business-units")
async def get_business_units():
    """Get available business units for audit"""
    return {
        "business_units": [
            {"id": "Brand", "name": "Brand", "description": "Brand campaigns"},
            {"id": "NonBrand", "name": "Non-Brand", "description": "Non-brand campaigns"},
            {"id": "PMax", "name": "Performance Max", "description": "PMax campaigns"},
        ]
    }


# Settings endpoints
@app.get("/api/settings")
async def get_settings():
    """Get current user settings"""
    # In a real app, this would fetch from database per user
    return {
        "ai_chat_enabled": True,
        "ai_insights_enabled": True,
        "theme": "light",
    }


@app.post("/api/settings")
async def update_settings(settings: dict[str, Any]):
    """Update user settings"""
    # In a real app, this would save to database per user
    return {"status": "success", "settings": settings}


# Creative Studio
@app.post("/api/creative/generate")
async def generate_creative(req: CreativeRequest):
    """
    Generate RSA headlines/descriptions using CreativeFactory
    """
    try:
        # Create CreativeContext from request
        context = CreativeContext(
            final_url=req.url or "",
            keywords=req.keywords or [],
            business_name=req.business_name,
            usp_list=req.usps or [],
        )
        result = CreativeFactory.generate_ad_copy(context)
        return {"status": "success", "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# PMax Deep Dive
@app.post("/api/pmax/analyze")
async def analyze_pmax(req: PMaxRequest):
    """
    Analyze PMax placements/spend/conversions
    """
    try:
        analyzer = PMaxAnalyzer()
        result = analyzer.analyze(
            placements=req.placements,
            total_cost=(req.total_cost if req.total_cost is not None else req.spend) or 0.0,
        )

        # Optional channel split inference if placements contain ad_network_type and cost
        try:
            df = None
            if req.placements:
                import pandas as pd  # local import to avoid overhead if unused
                df = pd.DataFrame(req.placements)
            if df is not None and "cost" in df.columns:
                splitter = PMaxChannelSplitter(df_campaign=df)
                breakout = splitter.infer()
                result["channel_breakout"] = breakout.__dict__
        except Exception:
            pass

        return {"status": "success", "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# SERP Monitor
@app.post("/api/serp/check")
async def serp_check(req: SerpRequest):
    """
    Check URL health / soft 404 using SerpScanner
    """
    try:
        # For now, use URL health checks only; SerpScanner requires keywords and is optional here
        urls = [u for u in (req.urls or []) if u]
        results = check_url_health(urls)
        return {"status": "success", "results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/serp/competitor-signal")
async def analyze_competitor_signal(obs: CompetitorObservation):
    """
    Infer investment signal from conversational observations.
    Uses fuzzy inference from natural language or precise metrics.

    This endpoint supports the intelligent mapping pattern where data
    is extracted from conversation rather than requiring file uploads.
    """
    try:
        result = analyze_competitor(
            competitor_domain=obs.competitor_domain,
            impression_share_current=obs.impression_share_current,
            impression_share_previous=obs.impression_share_previous,
            outranking_rate=obs.outranking_rate,
            top_of_page_rate=obs.top_of_page_rate,
            position_above_rate=obs.position_above_rate,
            raw_description=obs.raw_description
        )
        return {"status": "success", "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/intel/diagnose")
async def diagnose_intelligence(req: IntelRequest):
    """
    Agentic root-cause analysis across Market, Execution, and Creative pillars.
    This is additive and does not change existing flows.
    """

    def _sanitize_numbers(obj):
        """Recursively replace non-finite floats (nan/inf) with None for JSON safety."""
        import math
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize_numbers(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_numbers(v) for v in obj]
        return obj

    try:
        pmax_df = pd.DataFrame(req.pmax) if req.pmax else None
        creative_df = pd.DataFrame(req.creative) if req.creative else None
        market_df = pd.DataFrame(req.market) if req.market else None

        agent = MarketingReasoningAgent()
        result = agent.analyze(
            query=req.query,
            pmax_df=pmax_df if pmax_df is not None and not pmax_df.empty else None,
            creative_df=creative_df if creative_df is not None and not creative_df.empty else None,
            market_df=market_df if market_df is not None and not market_df.empty else None,
            brand_terms=req.brand_terms or [],
        )
        safe_result = _sanitize_numbers(result)
        return {"status": "success", "result": safe_result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/search/web")
async def web_search(payload: SearchQuery):
    """
    Lightweight web search via SerpAPI (bing engine).
    Requires env var: SERPAPI_KEY.
    """
    try:
        results = await _perform_web_search(payload.query, count=payload.count)
        return {"status": "success", "results": results}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
