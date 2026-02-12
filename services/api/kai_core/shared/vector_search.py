from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import requests

try:
    from azure.search.documents.models import VectorFilterMode, VectorizedQuery
except Exception:  # pragma: no cover - optional dependency in some runtimes
    VectorizedQuery = None  # type: ignore[assignment]
    VectorFilterMode = None  # type: ignore[assignment]

from kai_core.telemetry import log_openai_usage
from kai_core.config import is_azure_embeddings_enabled
from kai_core.shared.azure_budget import allow_azure_usage


@dataclass(frozen=True)
class VectorSearchConfig:
    search_endpoint: str
    search_key: str
    search_index: str
    vector_field: str
    k: int
    hybrid: bool
    filter_mode: str
    embedding_endpoint: str
    embedding_key: str
    embedding_deployment: str
    embedding_api_version: str
    embedding_timeout: float
    local_embedding_endpoint: str
    local_embedding_model: str
    local_embedding_timeout: float

    @property
    def enabled(self) -> bool:
        has_search = all([self.search_endpoint, self.search_key, self.search_index, self.vector_field])
        has_azure = all([self.embedding_endpoint, self.embedding_key, self.embedding_deployment])
        has_local = all([self.local_embedding_endpoint, self.local_embedding_model])
        return has_search and (has_azure or has_local)


def load_vector_config() -> VectorSearchConfig:
    search_endpoint = os.environ.get("CONCIERGE_SEARCH_ENDPOINT", "").strip()
    search_key = os.environ.get("CONCIERGE_SEARCH_KEY", "").strip()
    search_index = os.environ.get("CONCIERGE_SEARCH_INDEX", "").strip()
    vector_field = os.environ.get("CONCIERGE_SEARCH_VECTOR_FIELD", "contentVector").strip() or "contentVector"
    k = int(os.environ.get("CONCIERGE_SEARCH_VECTOR_K", "5") or 5)
    hybrid = os.environ.get("CONCIERGE_SEARCH_HYBRID", "true").strip().lower() in {"1", "true", "yes"}
    filter_mode = os.environ.get("CONCIERGE_SEARCH_VECTOR_FILTER_MODE", "preFilter").strip()

    embedding_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    embedding_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY") or ""
    embedding_deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "").strip()
    embedding_api_version = (
        os.environ.get("AZURE_OPENAI_EMBEDDING_API_VERSION")
        or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    )
    embedding_timeout = float(os.environ.get("AZURE_OPENAI_EMBEDDING_TIMEOUT_SECONDS", "20"))
    local_embedding_endpoint = os.environ.get("LOCAL_EMBEDDINGS_ENDPOINT", "").strip()
    local_embedding_model = os.environ.get("LOCAL_EMBEDDINGS_MODEL", "").strip()
    local_embedding_timeout = float(os.environ.get("LOCAL_EMBEDDINGS_TIMEOUT_SECONDS", "20"))

    return VectorSearchConfig(
        search_endpoint=search_endpoint,
        search_key=search_key,
        search_index=search_index,
        vector_field=vector_field,
        k=k,
        hybrid=hybrid,
        filter_mode=filter_mode,
        embedding_endpoint=embedding_endpoint,
        embedding_key=embedding_key,
        embedding_deployment=embedding_deployment,
        embedding_api_version=embedding_api_version,
        embedding_timeout=embedding_timeout,
        local_embedding_endpoint=local_embedding_endpoint,
        local_embedding_model=local_embedding_model,
        local_embedding_timeout=local_embedding_timeout,
    )


def _embedding_url(cfg: VectorSearchConfig) -> str:
    base = cfg.embedding_endpoint.rstrip("/")
    return f"{base}/openai/deployments/{cfg.embedding_deployment}/embeddings?api-version={cfg.embedding_api_version}"


def _local_embedding_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/api/embeddings"):
        return base
    if base.endswith("/api"):
        return f"{base}/embeddings"
    if base.endswith("/api/chat"):
        return f"{base[:-len('/chat')]}/embeddings"
    return f"{base}/api/embeddings"


def _get_local_embedding(text: str, cfg: VectorSearchConfig) -> Optional[List[float]]:
    if not cfg.local_embedding_endpoint or not cfg.local_embedding_model:
        return None
    if not text:
        return None
    url = _local_embedding_url(cfg.local_embedding_endpoint)
    payload = {"model": cfg.local_embedding_model, "prompt": _normalize_text(text)}
    try:
        resp = requests.post(url, json=payload, timeout=cfg.local_embedding_timeout)
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding") or data.get("data", [{}])[0].get("embedding")
        if isinstance(embedding, list):
            return embedding
    except Exception as exc:
        logging.warning("Local embedding request failed: %s", exc)
    return None


def _normalize_text(text: str, max_chars: int = 4000) -> str:
    value = (text or "").strip()
    if len(value) > max_chars:
        value = value[:max_chars]
    return value


def get_embedding(text: str, cfg: Optional[VectorSearchConfig] = None) -> Optional[List[float]]:
    cfg = cfg or load_vector_config()
    if not cfg.enabled:
        return None

    if not text:
        return None

    if not is_azure_embeddings_enabled():
        logging.info("Vector embeddings disabled by configuration; attempting local embeddings")
        return _get_local_embedding(text, cfg)

    allowed, reason = allow_azure_usage(module="vector_search", purpose="embeddings")
    if not allowed:
        logging.warning("Azure embeddings blocked by policy: %s", reason)
        return _get_local_embedding(text, cfg)

    payload = {"input": _normalize_text(text)}
    headers = {"api-key": cfg.embedding_key, "Content-Type": "application/json"}
    url = _embedding_url(cfg)
    max_retries = int(os.environ.get("AZURE_OPENAI_EMBEDDING_MAX_RETRIES", "3") or 3)
    base_delay = float(os.environ.get("AZURE_OPENAI_EMBEDDING_RETRY_BASE_SECONDS", "1.0") or 1.0)
    min_delay = float(os.environ.get("AZURE_OPENAI_EMBEDDING_MIN_DELAY_SECONDS", "0") or 0)

    for attempt in range(max_retries + 1):
        start = time.time()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=cfg.embedding_timeout)
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("data", [{}])[0].get("embedding")
            latency_ms = int((time.time() - start) * 1000)
            log_openai_usage(
                source="azure_embeddings",
                metadata={"deployment": cfg.embedding_deployment},
                usage=data.get("usage"),
                latency_ms=latency_ms,
            )
            if isinstance(embedding, list):
                if min_delay > 0:
                    time.sleep(min_delay)
                return embedding
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 429 and attempt < max_retries:
                retry_after = None
                if exc.response is not None:
                    retry_after = exc.response.headers.get("retry-after")
                if retry_after:
                    try:
                        time.sleep(float(retry_after))
                    except ValueError:
                        time.sleep(base_delay * (2**attempt))
                else:
                    time.sleep(base_delay * (2**attempt))
                continue
            logging.warning("Embedding request failed: %s", exc)
            return None
        except Exception as exc:
            logging.warning("Embedding request failed: %s", exc)
            return None
    return None


def build_vector_queries(text: str, cfg: Optional[VectorSearchConfig] = None) -> Optional[list]:
    cfg = cfg or load_vector_config()
    if not cfg.enabled or VectorizedQuery is None:
        return None

    embedding = get_embedding(text, cfg)
    if not embedding:
        return None

    return [
        VectorizedQuery(
            vector=embedding,
            k_nearest_neighbors=cfg.k,
            fields=cfg.vector_field,
        )
    ]


def resolve_vector_filter_mode(cfg: Optional[VectorSearchConfig] = None):
    cfg = cfg or load_vector_config()
    if VectorFilterMode is None:
        return None
    mode = (cfg.filter_mode or "").lower()
    if mode == "prefilter":
        return VectorFilterMode.PRE_FILTER
    if mode == "postfilter":
        return VectorFilterMode.POST_FILTER
    return VectorFilterMode.PRE_FILTER
