"""
Google Trends helper using pytrends.
Feature-flagged via ENABLE_TRENDS; intended for lightweight seasonality signals.
"""
from __future__ import annotations

import os
import math
import time
from typing import List, Tuple
import pandas as pd
import requests

from utils.circuit_breaker import CircuitBreaker
from utils.ttl_cache import TTLCache
from utils.telemetry import log_event

ENABLE_TRENDS = os.environ.get("ENABLE_TRENDS", "false").lower() == "true"
_TRENDS_CACHE = TTLCache(
    ttl_seconds=int(os.environ.get("TRENDS_CACHE_TTL_SECONDS", "3600") or "3600"),
    max_items=int(os.environ.get("TRENDS_CACHE_MAX_ITEMS", "128") or "128"),
)
TRENDS_TOTAL_TIMEOUT_SECONDS = float(os.environ.get("TRENDS_TOTAL_TIMEOUT_SECONDS", "45") or "45")
TRENDS_CONNECT_TIMEOUT_SECONDS = float(os.environ.get("TRENDS_CONNECT_TIMEOUT_SECONDS", "5") or "5")
TRENDS_READ_TIMEOUT_SECONDS = float(os.environ.get("TRENDS_READ_TIMEOUT_SECONDS", "25") or "25")
TRENDS_SERP_TIMEOUT_SECONDS = float(os.environ.get("TRENDS_SERP_TIMEOUT_SECONDS", "25") or "25")
_BREAKER_ENABLED = os.environ.get("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
_TRENDS_BREAKER = CircuitBreaker(
    "trends",
    failure_threshold=int(os.environ.get("TRENDS_BREAKER_FAILURES", "3") or "3"),
    cooldown_seconds=int(os.environ.get("TRENDS_BREAKER_COOLDOWN_SECONDS", "60") or "60"),
    enabled=_BREAKER_ENABLED,
)


def _cache_key(keywords: List[str], timeframe: str, geo: str | None) -> str:
    parts = [timeframe or "", geo or "", ",".join(sorted([k.lower() for k in keywords if k]))]
    return "|".join(parts)


def _log(msg: str):
    print(msg, flush=True)


def _safe_import_pytrends():
    try:
        from pytrends.request import TrendReq  # type: ignore
        return TrendReq
    except Exception:
        return None


def _time_left(start: float, total_seconds: float) -> float | None:
    if total_seconds <= 0:
        return None
    return total_seconds - (time.perf_counter() - start)


def fetch_trends(keywords: List[str], timeframe: str = "now 12-m", geo: str | None = None) -> Tuple[pd.DataFrame, dict]:
    """
    Fetch interest-over-time and related queries for the given keywords.
    Returns (iot_df, related_queries) or (empty, {} ) if disabled/unavailable.
    """
    if not ENABLE_TRENDS:
        _log("[trends] ENABLE_TRENDS=false; returning empty trends.")
        return pd.DataFrame(), {}
    total_start = time.perf_counter()
    cache_key = _cache_key(keywords, timeframe, geo)
    cached = _TRENDS_CACHE.get(cache_key)
    if cached:
        try:
            iot_cached, related_cached = cached
            return iot_cached.copy(), related_cached
        except Exception:
            pass
    if not _TRENDS_BREAKER.allow():
        _log("[trends] circuit open; returning empty trends.")
        return pd.DataFrame(), {}
    TrendReq = _safe_import_pytrends()
    if TrendReq is None:
        _log("[trends] pytrends import failed; returning empty trends.")
        return pd.DataFrame(), {}
    # Normalize timeframe and try fallbacks (pytrends is picky: use today X-m/d)
    tf_primary = timeframe or "today 12-m"
    if tf_primary.lower().startswith("now "):
        tf_primary = "today " + tf_primary.split(" ", 1)[1]
    timeframes = [tf_primary, "today 12-m", "today 3-m", "today 1-m"]
    seen = set()
    for tf in timeframes:
        remaining = _time_left(total_start, TRENDS_TOTAL_TIMEOUT_SECONDS)
        if remaining is not None and remaining <= 0:
            _log("[trends] total timeout exceeded before timeframe attempts.")
            break
        tf_norm = (tf or "").strip().lower()
        if not tf_norm or tf_norm in seen:
            continue
        seen.add(tf_norm)
        try:
            start = time.perf_counter()
            pt = TrendReq(
                hl="en-US",
                tz=0,
                retries=2,
                backoff_factor=0.1,
                timeout=(TRENDS_CONNECT_TIMEOUT_SECONDS, TRENDS_READ_TIMEOUT_SECONDS),
                requests_args={"headers": {"User-Agent": "Mozilla/5.0 (compatible; KaiBot/1.0)"}},
            )
            pt.build_payload(keywords, timeframe=tf_norm, geo=geo or "")
            iot = pt.interest_over_time().reset_index()
            related = pt.related_queries()
            if iot is not None and not iot.empty:
                latency_ms = (time.perf_counter() - start) * 1000
                _TRENDS_BREAKER.record_success()
                log_event("external_call", service="trends_pytrends", latency_ms=latency_ms, timeframe=tf_norm)
                _log(f"[trends] pytrends rows={len(iot)} timeframe={tf_norm}")
                _TRENDS_CACHE.set(cache_key, (iot.copy(), related))
                return iot, related
            _log(f"[trends] empty result for timeframe={tf_norm}")
        except Exception as exc:
            _TRENDS_BREAKER.record_failure()
            _log(f"[trends] fetch_trends exception timeframe={tf_norm}: {exc}")
            continue
    # Fallback: SerpAPI Google Trends if available
    serp_key = os.environ.get("SERPAPI_KEY")
    if serp_key:
        try:
            serp_date = tf_primary
            if serp_date.lower().startswith("now "):
                serp_date = "today " + serp_date.split(" ", 1)[1]
            frames = []
            for kw in keywords:
                remaining = _time_left(total_start, TRENDS_TOTAL_TIMEOUT_SECONDS)
                if remaining is not None and remaining <= 0:
                    _log("[trends] total timeout exceeded before SerpAPI attempts.")
                    break
                start = time.perf_counter()
                params = {
                    "engine": "google_trends",
                    "q": kw,
                    "data_type": "TIMESERIES",
                    # SerpAPI expects `date` (not `time_range`)
                    "date": serp_date or "today 12-m",
                    "geo": geo or "US",
                    "tz": 0,
                    "api_key": serp_key,
                }
                resp = requests.get(
                    "https://serpapi.com/search.json",
                    params=params,
                    timeout=TRENDS_SERP_TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
                data = resp.json()
                timeline = (data.get("interest_over_time") or {}).get("timeline_data") or []
                rows = []
                for entry in timeline:
                    ts = entry.get("timestamp")
                    dt = None
                    if ts:
                        try:
                            dt = pd.to_datetime(int(ts), unit="s", utc=True)
                        except Exception:
                            dt = None
                    if dt is None:
                        dt = pd.to_datetime(entry.get("date"), errors="coerce")
                    vals = entry.get("values") or []
                    first = vals[0] if isinstance(vals, list) and vals else {}
                    val = None
                    if isinstance(first, dict):
                        val = first.get("extracted_value")
                        if val is None:
                            val = first.get("value")
                    elif isinstance(first, (int, float)):
                        val = first
                    if val is None:
                        continue
                    try:
                        val_num = float(val if not isinstance(val, list) else (val[0] if val else 0))
                    except Exception:
                        continue
                    rows.append({"date": dt, kw: val_num})
                if rows:
                    df = pd.DataFrame(rows)
                    df = df.dropna(subset=["date"])
                    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                    if not df.empty:
                        latency_ms = (time.perf_counter() - start) * 1000
                        log_event("external_call", service="trends_serpapi", latency_ms=latency_ms, keyword=kw)
                        _log(f"[trends] SerpAPI rows for '{kw}': {len(df)} timeframe={serp_date}")
                        frames.append(df[["date", kw]])
            if frames:
                merged = frames[0]
                for f in frames[1:]:
                    merged = pd.merge(merged, f, on="date", how="outer")
                merged = merged.sort_values("date").reset_index(drop=True)
                _log(f"[trends] SerpAPI fallback succeeded. rows={len(merged)}")
                _TRENDS_BREAKER.record_success()
                _TRENDS_CACHE.set(cache_key, (merged.copy(), {}))
                return merged, {}
            _log("[trends] SerpAPI fallback returned empty.")
        except Exception as exc:
            _TRENDS_BREAKER.record_failure()
            _log(f"[trends] SerpAPI fallback exception: {exc}")
    _log("[trends] all timeframes empty and no fallback data; returning empty trends.")
    return pd.DataFrame(), {}


def seasonality_multipliers(iot: pd.DataFrame, keywords: List[str]) -> dict:
    """
    Compute simple seasonality multipliers vs the mean for each keyword.
    Returns {keyword: {"multiplier": float, "recent": float, "avg": float}}
    """
    out = {}
    if iot is None or iot.empty:
        return out
    for kw in keywords:
        if kw not in iot.columns:
            continue
        series = pd.to_numeric(iot[kw], errors="coerce").dropna()
        if series.empty:
            continue
        recent = series.iloc[-1]
        avg = series.mean()
        if avg and math.isfinite(avg):
            mult = max(0.2, min(3.0, recent / avg))
        else:
            mult = 1.0
        out[kw] = {"multiplier": mult, "recent": float(recent), "avg": float(avg)}
    return out


def summarize_seasonality(iot: pd.DataFrame, keywords: List[str]) -> dict:
    """
    Build a simple seasonality summary from interest-over-time.
    Returns {peaks: [...], lows: [...], shoulder: [...], month_scores: [...]}
    """
    if iot is None or iot.empty:
        return {}
    df = iot.copy()
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    # reshape long: month -> sum of normalized interest across keywords
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    month_scores = []
    for m, g in df.groupby("month"):
        score = 0.0
        for kw in keywords:
            if kw in g.columns:
                score += pd.to_numeric(g[kw], errors="coerce").mean()
        month_scores.append({"month": str(m), "score": float(score)})
    if not month_scores:
        return {}
    # rank months
    month_scores_sorted = sorted(month_scores, key=lambda x: x["score"], reverse=True)
    peaks = month_scores_sorted[:3]
    lows = month_scores_sorted[-2:] if len(month_scores_sorted) >= 2 else month_scores_sorted[-1:]
    # shoulders: middle tercile
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
