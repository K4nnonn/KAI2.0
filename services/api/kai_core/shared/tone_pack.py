from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List, Optional

from kai_core.shared.vector_search import (
    build_vector_queries,
    load_vector_config,
    resolve_vector_filter_mode,
)

_CACHE = {"key": None, "expires": 0.0, "value": ""}


def _tone_pack_enabled() -> bool:
    return os.environ.get("ENABLE_TONE_PACK", "true").strip().lower() in {"1", "true", "yes"}


def _cache_seconds() -> float:
    raw = os.environ.get("TONE_PACK_CACHE_SECONDS", "180") or "180"
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 180.0


def _max_chars() -> int:
    raw = os.environ.get("TONE_PACK_MAX_CHARS", "1200") or "1200"
    try:
        return max(200, int(raw))
    except ValueError:
        return 1200


def _tone_pack_section() -> str:
    return (os.environ.get("TONE_PACK_SECTION") or "Tone Persona Pack").strip()


def _tone_pack_query(use_case: str, audience: Optional[str]) -> str:
    base = (os.environ.get("TONE_PACK_QUERY") or "tone persona guidance for Kai responses").strip()
    bits = [base, use_case or "general"]
    if audience:
        bits.append(str(audience))
    return " ".join(bits).strip()


def _get_search_client(cfg):
    try:
        from azure.search.documents import SearchClient  # type: ignore
        from azure.core.credentials import AzureKeyCredential  # type: ignore
    except Exception:
        return None
    if not cfg.search_endpoint or not cfg.search_key or not cfg.search_index:
        return None
    return SearchClient(
        endpoint=cfg.search_endpoint,
        index_name=cfg.search_index,
        credential=AzureKeyCredential(cfg.search_key),
    )


def _search_tone_docs(query: str, cfg, section: str) -> List[str]:
    client = _get_search_client(cfg)
    if not client:
        return []
    vector_queries = build_vector_queries(query, cfg) if cfg.enabled else None
    vector_filter_mode = resolve_vector_filter_mode(cfg) if vector_queries else None

    filter_expr = "source eq 'knowledge_base'"
    if section:
        escaped = section.replace("'", "''")
        filter_expr = f"{filter_expr} and section eq '{escaped}'"

    def _run_search(filter_override: Optional[str]):
        return client.search(
            search_text=query if cfg.hybrid else None,
            filter=filter_override,
            query_type="semantic" if cfg.hybrid else None,
            semantic_configuration_name="kai-semantic" if cfg.hybrid else None,
            vector_queries=vector_queries,
            vector_filter_mode=vector_filter_mode,
            top=cfg.k or 5,
        )

    try:
        results = _run_search(filter_expr)
    except Exception as exc:
        logging.warning("[tone_pack] vector search filter failed; retrying without section: %s", exc)
        try:
            results = _run_search("source eq 'knowledge_base'")
        except Exception as exc2:
            logging.warning("[tone_pack] vector search failed: %s", exc2)
            return []

    chunks = []
    for doc in results:
        content = (doc.get("content") or "").strip()
        if content:
            chunks.append(content)
    return chunks


def _load_local_tone_pack() -> str:
    env_path = os.environ.get("TONE_PACK_LOCAL_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")

    current = Path(__file__).resolve()
    for parent in current.parents:
        fallback = parent / "knowledge-base" / "tone_persona_pack.md"
        if fallback.exists():
            return fallback.read_text(encoding="utf-8", errors="ignore")
    return ""


def get_tone_guidance(use_case: str = "general", audience: Optional[str] = None) -> str:
    if not _tone_pack_enabled():
        return ""
    if os.environ.get("REQUIRE_LOCAL_LLM", "").strip().lower() in {"1", "true", "yes"}:
        return _load_local_tone_pack()[: _max_chars()]
    if os.environ.get("AZURE_OPENAI_DISABLED", "").strip().lower() in {"1", "true", "yes"}:
        return _load_local_tone_pack()[: _max_chars()]
    if os.environ.get("TONE_PACK_FORCE_LOCAL", "").strip().lower() in {"1", "true", "yes"}:
        return _load_local_tone_pack()[: _max_chars()]

    cache_key = f"{use_case}|{audience or ''}"
    now = time.time()
    if _CACHE["key"] == cache_key and now < _CACHE["expires"]:
        return _CACHE["value"]

    cfg = load_vector_config()
    section = _tone_pack_section()
    query = _tone_pack_query(use_case, audience)
    chunks = _search_tone_docs(query, cfg, section) if cfg.enabled else []
    if not chunks:
        guidance = _load_local_tone_pack()
    else:
        guidance = "\n\n".join(chunks)

    guidance = (guidance or "").strip()
    if len(guidance) > _max_chars():
        guidance = guidance[: _max_chars()].rstrip()

    _CACHE["key"] = cache_key
    _CACHE["expires"] = now + _cache_seconds()
    _CACHE["value"] = guidance
    return guidance


def append_tone_guidance(base_prompt: str, use_case: str, audience: Optional[str] = None) -> str:
    guidance = get_tone_guidance(use_case=use_case, audience=audience)
    if not guidance:
        return base_prompt
    return (
        base_prompt.rstrip()
        + "\n\nTone guidance (knowledge base):\n"
        + guidance
    )
