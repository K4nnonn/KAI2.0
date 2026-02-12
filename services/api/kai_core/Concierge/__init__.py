"""
Kai Concierge - AI-enabled implementation.

Features:
- Azure OpenAI responses with lightweight prompt.
- Conversation memory persisted to Azure Table Storage.
- Session metadata tracking (counts, timestamps).
- Graceful fallbacks when storage/OpenAI unavailable.
"""

import json
import logging
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import azure.functions as func  # type: ignore
except Exception:
    # Local/container fallback when azure.functions is not installed
    class _HttpResponse:
        def __init__(self, body, status_code=200, mimetype="application/json", headers=None):
            self.body = body
            self.status_code = status_code
            self.mimetype = mimetype
            self.headers = headers or {}

    class _HttpRequest:
        def __init__(self, body=None, headers=None, method="POST"):
            self._body = body or {}
            self.headers = headers or {}
            self.method = method

        def get_json(self):
            return self._body

    class _func:
        HttpResponse = _HttpResponse
        HttpRequest = _HttpRequest

    func = _func()
import requests
from azure.core.credentials import AzureKeyCredential
from azure.data.tables import TableServiceClient
from azure.search.documents import SearchClient

from kai_core.account_utils import normalize_account_name
from kai_core.shared.core_logic.protected_prompts import get_system_prompt
from kai_core.shared.vector_search import (
    build_vector_queries,
    load_vector_config,
    resolve_vector_filter_mode,
)
from kai_core.config import (
    get_deployment_mode,
    get_system_prompt_override,
    get_tenant_override,
    is_maintenance_mode,
    is_azure_openai_enabled,
)
from kai_core.telemetry import log_openai_usage
from kai_core.plugins.creative import CreativeFactory
from kai_core.shared.adapters import CreativeContext
from kai_core.plugins.serp import check_url_health
from kai_core.plugins.pmax import PMaxAnalyzer
from kai_core.plugins.sqr import SqrAnalyzer
from kai_core.shared.azure_budget import allow_azure_usage

ALLOWED_WEB_DOMAINS = [
    "support.google.com",
    "help.ads.microsoft.com",
    "searchengineland.com",
    "ppchero.com",
    "wordstream.com",
]

def _utc_iso() -> str:
    return datetime.utcnow().isoformat()


def _tenant_id_from_request(req: func.HttpRequest, body: Optional[Dict] = None) -> str:
    """
    Derive a tenant identifier from headers or payload.
    Defaults to 'public' to remain backward compatible when absent.
    """
    default_tenant = (
        get_tenant_override()
        or (os.environ.get("DEFAULT_TENANT_ID") or "public").strip()
        or "public"
    )
    headers = req.headers or {}
    header_tenant = headers.get("x-tenant-id") or headers.get("X-Tenant-Id")
    if header_tenant:
        return header_tenant.strip() or default_tenant
    if body and isinstance(body, dict):
        tenant = body.get("tenantId") or body.get("tenant_id")
        if tenant:
            return str(tenant).strip() or default_tenant
    return default_tenant


def _partition_key(session_id: str, tenant_id: str) -> str:
    tenant = tenant_id or "public"
    sid = session_id or "unknown-session"
    return f"{tenant}|{sid}"


def _run_audit_and_extract(account_name: str, extractor, overrides: Optional[Dict] = None) -> Dict:
    """
    Run the offline audit using local CSV/template and apply a view extractor.
    """
    from kai_core.GenerateAudit import _build_account_context, _resolve_data_directory  # local import to avoid heavy import during tests
    try:
        from kai_core.core_logic.engine import UnifiedAuditEngine  # type: ignore
    except Exception:
        from kai_core_engine import UnifiedAuditEngine  # type: ignore

    template_path = Path(__file__).resolve().parents[1] / "GenerateAudit" / "template.xlsx"
    data_dir = _resolve_data_directory(account_name)
    business_context = _build_account_context(account_name, overrides or {})

    engine = UnifiedAuditEngine(
        template_path=template_path,
        data_directory=data_dir,
        business_unit=account_name,
        business_context=business_context.model_dump(),
    )
    output_file = Path(tempfile.gettempdir()) / f"Kai_{account_name}_{int(time.time())}.xlsx"
    try:
        result = engine.generate_audit(account_name=account_name, output_path=output_file)
    finally:
        output_file.unlink(missing_ok=True)

    result["business_context"] = business_context.model_dump()
    return {"result": result, "business_context": business_context.model_dump(), "extract": extractor(result) if extractor else None}


def _run_audit(account_name: str, overrides: Optional[Dict] = None) -> Dict:
    """
    Run offline audit and return full result with business_context.
    """
    payload = _run_audit_and_extract(account_name, extractor=None, overrides=overrides)
    return payload


def _load_local_docs() -> List[Dict]:
    """
    Graceful loader for any docs that exist in the release. Non-fatal if empty.
    Prefers COMPLIANCE_ARCH_MAP.md; will index other .md/.txt files present.
    """
    docs_dir = Path(__file__).resolve().parents[2] / "docs"
    if not docs_dir.exists():
        return []
    chunks: List[Dict] = []
    for idx, path in enumerate(sorted(docs_dir.glob("*"))):
        if path.name.lower() == "compliance_arch_map.md" or path.suffix.lower() in {".md", ".txt"}:
            try:
                text = path.read_text(encoding="utf-8")
                snippet = text[:1500]
                marker = f"[DOC{idx + 1}]"
                chunks.append(
                    {
                        "marker": marker,
                        "id": path.stem,
                        "title": path.stem.replace("_", " "),
                        "section": "Documentation",
                        "content": snippet,
                        "templateVersion": "docs-fallback",
                    }
                )
            except Exception:
                continue
    return chunks


def _get_table_client(table_name: str):
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING not configured")
    service = TableServiceClient.from_connection_string(connection_string)
    return service.get_table_client(table_name)


def _ensure_session_entity(session_id: str, tenant_id: str, account_name: Optional[str] = None) -> Dict:
    table_name = os.environ.get("CONCIERGE_SESSION_TABLE", "conciergeSessions")
    client = _get_table_client(table_name)
    partition = _partition_key(session_id, tenant_id)
    canonical_account = normalize_account_name(account_name) if account_name else None
    try:
        entity = client.get_entity(partition_key=partition, row_key="session")
    except Exception:
        entity = {
            "PartitionKey": partition,
            "RowKey": "session",
            "createdAt": _utc_iso(),
            "lastMessageAt": _utc_iso(),
            "messageCount": 0,
            "currentIntent": "",
        }
        if canonical_account:
            entity["accountName"] = canonical_account
        client.upsert_entity(entity)
    else:
        if canonical_account and not entity.get("accountName"):
            entity["accountName"] = canonical_account
            client.update_entity(entity)
    return client.get_entity(partition_key=partition, row_key="session")


def _update_session_metadata(session_id: str, tenant_id: str, intent: Optional[str] = None) -> None:
    table_name = os.environ.get("CONCIERGE_SESSION_TABLE", "conciergeSessions")
    client = _get_table_client(table_name)
    try:
        partition = _partition_key(session_id, tenant_id)
        entity = client.get_entity(partition_key=partition, row_key="session")
        entity["lastMessageAt"] = _utc_iso()
        entity["messageCount"] = int(entity.get("messageCount", 0)) + 1
        if intent:
            entity["currentIntent"] = intent
        client.update_entity(entity)
    except Exception as exc:
        logging.warning("Failed to update session metadata: %s", exc)


def save_message(session_id: str, tenant_id: str, role: str, content: str, intent: Optional[str] = None) -> None:
    table_name = os.environ.get("CONCIERGE_MESSAGE_TABLE", "conciergeMessages")
    try:
        client = _get_table_client(table_name)
        entity = {
            "PartitionKey": _partition_key(session_id, tenant_id),
            "RowKey": f"{_utc_iso()}_{role}_{os.urandom(4).hex()}",
            "role": role,
            "content": content,
            "intent": intent or "",
            "timestamp": _utc_iso(),
        }
        client.create_entity(entity)
    except Exception as exc:
        logging.warning("Failed to save message: %s", exc)


def record_retrieval(session_id: str, tenant_id: str, account_name: str, query: str, doc_ids: List[str], intent: str, count: int):
    table_name = os.environ.get("CONCIERGE_RETRIEVAL_TABLE", "conciergeRetrievals")
    try:
        client = _get_table_client(table_name)
        entity = {
            "PartitionKey": _partition_key(session_id, tenant_id),
            "RowKey": f"{_utc_iso()}_{os.urandom(4).hex()}",
            "timestamp": _utc_iso(),
            "accountName": account_name,
            "queryText": query,
            "retrievedDocIds": json.dumps(doc_ids),
            "citationCount": count,
            "intent": intent,
        }
        client.create_entity(entity)
    except Exception as exc:
        logging.warning("Failed to record retrieval: %s", exc)


def get_history(session_id: str, tenant_id: str, limit: int = 10) -> List[Dict[str, str]]:
    table_name = os.environ.get("CONCIERGE_MESSAGE_TABLE", "conciergeMessages")
    try:
        client = _get_table_client(table_name)
        partition = _partition_key(session_id, tenant_id)
        entities = list(
            client.query_entities(f"PartitionKey eq '{partition}'", results_per_page=limit * 2)
        )
        entities.sort(key=lambda e: e.get("timestamp", ""))
        history = []
        for e in entities[-limit:]:
            history.append({"role": e.get("role", "assistant"), "content": e.get("content", "")})
        return history
    except Exception as exc:
        logging.warning("Failed to read history: %s", exc)
        return []


def get_search_client() -> Optional[SearchClient]:
    endpoint = os.environ.get("CONCIERGE_SEARCH_ENDPOINT")
    key = os.environ.get("CONCIERGE_SEARCH_KEY")
    index_name = os.environ.get("CONCIERGE_SEARCH_INDEX")
    if not endpoint or not key or not index_name:
        return None
    credential = AzureKeyCredential(key)
    return SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)


def retrieve_audit_context(query: str, account_name: str) -> List[Dict]:
    search_client = get_search_client()
    canonical_account = normalize_account_name(account_name)
    if not search_client or not canonical_account:
        return []
    try:
        escaped_account = canonical_account.replace("'", "''")
        filter_expr = f"accountName eq '{escaped_account}'"
        cfg = load_vector_config()
        vector_queries = build_vector_queries(query, cfg) if cfg.enabled else None
        vector_filter_mode = resolve_vector_filter_mode(cfg) if vector_queries else None

        if vector_queries:
            try:
                results = search_client.search(
                    search_text=query if cfg.hybrid else None,
                    filter=filter_expr,
                    query_type="semantic" if cfg.hybrid else None,
                    semantic_configuration_name="kai-semantic" if cfg.hybrid else None,
                    vector_queries=vector_queries,
                    vector_filter_mode=vector_filter_mode,
                    top=cfg.k or 5,
                )
            except Exception as exc:
                logging.warning("Vector search failed; falling back to semantic: %s", exc)
                results = None
        else:
            results = None

        if results is None:
            results = search_client.search(
                search_text=query,
                filter=filter_expr,
                query_type="semantic",
                semantic_configuration_name="kai-semantic",
                top=5,
            )
        chunks = []
        for idx, doc in enumerate(results):
            chunk = {
                "marker": f"[{idx + 1}]",
                "id": doc["id"],
                "title": doc.get("section") or "Audit Insight",
                "content": doc.get("content", ""),
                "account": doc.get("accountName"),
                "auditDate": doc.get("auditDate"),
                "section": doc.get("section"),
            }

            # Smart Bidding upgrade: include signals and qaNotes if available
            if "signals" in doc and doc["signals"]:
                chunk["signals"] = doc["signals"]
            if "qaNotes" in doc and doc["qaNotes"]:
                chunk["qaNotes"] = doc["qaNotes"]
            if "templateVersion" in doc and doc["templateVersion"]:
                chunk["templateVersion"] = doc["templateVersion"]

            chunks.append(chunk)
        return chunks
    except Exception as exc:
        logging.warning("Audit retrieval failed: %s", exc)
        return []


def search_best_practices(query: str) -> List[Dict]:
    """
    Use Google Custom Search API to find best-practice articles.
    Returns list of {marker, title, url, snippet, source}.
    """
    api_key = os.environ.get("GOOGLE_SEARCH_API_KEY")
    search_engine_id = os.environ.get("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key or not search_engine_id:
        logging.warning("Google Custom Search not configured (missing API key or Search Engine ID)")
        return []

    endpoint = "https://www.googleapis.com/customsearch/v1"

    # Google Custom Search doesn't need site: filters in query if CSE is configured with domains
    # But we can add them for extra filtering
    domain_filter = " OR ".join([f"site:{domain}" for domain in ALLOWED_WEB_DOMAINS])
    full_query = f"{query} ({domain_filter})"

    params = {
        "key": api_key,
        "cx": search_engine_id,
        "q": full_query,
        "num": 5  # Request up to 5 results, we'll return top 3
    }

    try:
        resp = requests.get(endpoint, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("items", []):
            url = item.get("link", "")
            # Extract domain from URL for validation
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                host = parsed.netloc
            except Exception:
                host = url

            # Verify result is from allowed domains
            if not any(domain in host for domain in ALLOWED_WEB_DOMAINS):
                continue

            marker = f"[W{len(results) + 1}]"
            results.append({
                "marker": marker,
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("snippet", ""),
                "source": host,
            })

            if len(results) == 3:
                break

        return results
    except requests.HTTPError as exc:
        logging.warning("Google Custom Search failed: %s", exc.response.text if exc.response else exc)
    except Exception as exc:
        logging.warning("Google Custom Search error: %s", exc)
    return []


def call_azure_openai(
    messages: List[Dict[str, str]],
    session_id: Optional[str] = None,
    intent: Optional[str] = None,
    tenant_id: Optional[str] = None,
    use_json_mode: bool = False,
    max_tokens: int = 500,
    temperature: float = 0.6,
    purpose: Optional[str] = None,
) -> str:
    if not is_azure_openai_enabled():
        raise RuntimeError("Azure OpenAI disabled by configuration")

    allowed, reason = allow_azure_usage(
        module="concierge",
        purpose=purpose or intent or "chat",
        intent=intent,
    )
    if not allowed:
        raise RuntimeError(f"Azure OpenAI blocked by policy: {reason}")

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4-turbo")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

    if not endpoint or not api_key:
        raise RuntimeError("Azure OpenAI credentials missing")

    url = f"{endpoint}openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if use_json_mode:
        payload["response_format"] = {"type": "json_object"}
        payload["temperature"] = 0
    headers = {"Content-Type": "application/json", "api-key": api_key}

    start = time.perf_counter()
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    latency_ms = (time.perf_counter() - start) * 1000
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage")
    log_openai_usage(
        source="concierge",
        metadata={
            "sessionId": session_id or "",
            "intent": intent or "",
            "deployment": deployment,
            "tenantId": tenant_id or "public",
        },
        usage=usage,
        latency_ms=latency_ms,
    )
    return data["choices"][0]["message"]["content"]


def classify_intent(message: str) -> str:
    text = message.lower()
    if any(word in text for word in ["creative", "ad copy", "headline"]):
        return "creative_tool"
    if any(word in text for word in ["check url", "landing page", "404"]):
        return "serp_tool"
    if "pmax" in text or "performance max" in text:
        return "pmax_tool"
    if any(word in text for word in ["search term", "sqr", "match type"]):
        return "sqr_tool"
    if any(word in text for word in ["best practice", "how should", "what's the right way", "optimize"]):
        return "best_practice_query"
    if any(word in text for word in ["audit", "report", "generate", "analyze"]):
        return "audit_query"
    if any(word in text for word in ["status", "progress", "done", "complete"]):
        return "status_query"
    if any(word in text for word in ["upload", "file", "csv", "data"]):
        return "file_query"
    if any(word in text for word in ["help", "how", "what", "guide", "strategy", "architecture", "negative keyword"]):
        return "knowledge_query"
    return "general_query"


def build_context_message(audit_chunks: List[Dict], web_chunks: List[Dict]) -> str:
    lines = []
    for chunk in audit_chunks:
        # Basic audit content
        line = f"{chunk['marker']} Account: {chunk.get('account')} | Section: {chunk.get('section')} | Content: {chunk.get('content')}"

        # Smart Bidding upgrade: append signals and qaNotes if present
        if chunk.get("signals"):
            signals_json = json.dumps(chunk["signals"])
            line += f" | Machine Metrics: {signals_json}"

        if chunk.get("qaNotes"):
            line += f" | QA Notes: {chunk['qaNotes']}"

        if chunk.get("templateVersion"):
            line += f" | Template: {chunk['templateVersion']}"

        lines.append(line)

    for chunk in web_chunks:
        lines.append(f"{chunk['marker']} {chunk.get('title')} ({chunk.get('url')}): {chunk.get('snippet')}")

    return "Relevant references:\n" + "\n".join(lines) if lines else "No external references available."


def compose_response_payload(ai_reply: str, intent: str, session_id: str, audit_chunks: List[Dict], web_chunks: List[Dict]):
    audit_sources = [
        {
            "marker": chunk["marker"],
            "title": chunk.get("title"),
            "section": chunk.get("section"),
            "auditDate": chunk.get("auditDate"),
            "content": chunk.get("content"),
        }
        for chunk in audit_chunks
    ]
    web_sources = [
        {
            "marker": chunk["marker"],
            "title": chunk.get("title"),
            "url": chunk.get("url"),
            "snippet": chunk.get("snippet"),
        }
        for chunk in web_chunks
    ]
    return {
        "reply": ai_reply,
        "intent": intent,
        "sessionId": session_id,
        "citations": {
            "audit": audit_sources,
            "web": web_sources,
        }
    }


def handle_request_simulated(message: str, account_name: str = "public") -> str:
    """
    Lightweight, offline-safe handler for testing knowledge routing without external calls.
    """
    intent = classify_intent(message)
    if intent in {"creative_tool", "serp_tool", "pmax_tool", "sqr_tool"}:
        return f"Routed to {intent}"
    docs = _load_local_docs()
    prompt = get_system_prompt()
    doc_snippet = docs[0]["content"] if docs else ""
    return f"{prompt}\n\n{doc_snippet}"


def classify_business_context(account_name: str, payload: Dict[str, object]) -> Any:
    from kai_core.shared.business_physics import AccountContext  # type: ignore

    vertical_hint = payload.get("vertical") or payload.get("businessModel")
    overrides: Dict[str, object] = {}
    if vertical_hint:
        overrides["vertical"] = vertical_hint
    if "margin" in payload:
        overrides["margin_input"] = payload.get("margin")
    if "grossMargin" in payload:
        overrides["margin_input"] = payload.get("grossMargin")
    if "targetROAS" in payload:
        overrides["target_roas"] = payload.get("targetROAS")
    if "target_roas" in payload:
        overrides["target_roas"] = payload.get("target_roas")
    if "biddingStrategy" in payload:
        overrides["bidding_strategy"] = payload.get("biddingStrategy")
    if "conversions30d" in payload:
        overrides["conversions_30d"] = payload.get("conversions30d")
    if "currentROAS" in payload:
        overrides["current_roas"] = payload.get("currentROAS")
    if "verificationStatus" in payload:
        overrides["verification_status"] = payload.get("verificationStatus")
    if "targetCountry" in payload:
        overrides["target_country"] = payload.get("targetCountry")

    detected_vertical = _infer_vertical(account_name, overrides.get("vertical"))
    return AccountContext(
        account_name=account_name,
        detected_vertical=detected_vertical,
        margin_input=_to_float(overrides.get("margin_input")),
        target_roas=_to_float(overrides.get("target_roas")),
        bidding_strategy=str(overrides.get("bidding_strategy")).upper() if overrides.get("bidding_strategy") else None,
        conversions_30d=_to_int(overrides.get("conversions_30d")),
        verification_status=bool(overrides.get("verification_status") or False),
        target_country=str(overrides.get("target_country")).lower() if overrides.get("target_country") else None,
    )


def _infer_vertical(account_name: str, override: Optional[object]) -> str:
    if override:
        label = str(override).upper()
        mapping = {
            "ECOMMERCE": "ECOMMERCE",
            "E-COMMERCE": "ECOMMERCE",
            "RETAIL": "ECOMMERCE",
            "D2C": "ECOMMERCE",
            "SAAS": "SAAS",
            "SOFTWARE": "SAAS",
            "FINANCE": "FINANCE",
            "BANKING": "FINANCE",
            "INSURANCE": "FINANCE",
            "LOCAL": "LOCAL",
            "LEAD_GEN": "LEAD_GEN",
            "LEADGEN": "LEAD_GEN",
        }
        return mapping.get(label, "LEAD_GEN")
    lowered = (account_name or "").lower()
    finance_keywords = ["retire", "wealth", "broker", "asset", "crypto", "fili", "invest", "medicare"]
    if any(keyword in lowered for keyword in finance_keywords):
        return "FINANCE"
    return "LEAD_GEN"


def _to_float(value: Optional[object]) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Optional[object]) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def handle_options() -> func.HttpResponse:
    return func.HttpResponse(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )


def error_response(message: str, status: int) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps({"error": message}),
        status_code=status,
        mimetype="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )


def main(req: func.HttpRequest) -> func.HttpResponse:
    mode = get_deployment_mode()
    logging.info("Concierge AI function invoked (mode=%s)", mode)

    if req.method == "OPTIONS":
        return handle_options()

    try:
        payload = req.get_json()
    except ValueError:
        return error_response("Invalid JSON payload", 400)

    tenant_id = _tenant_id_from_request(req, payload)
    message = (payload.get("message") or "").strip()
    session_id = (payload.get("sessionId") or "").strip()
    audience = (payload.get("audience") or payload.get("tone") or "client").strip().lower()
    requested_account = normalize_account_name(payload.get("accountName"))
    if not message:
        return error_response("Message is required", 400)
    if not session_id:
        return error_response("sessionId is required", 400)

    intent = classify_intent(message)
    # Map tool aliases to legacy routing keys
    alias_map = {
        "creative_tool": "creative_query",
        "serp_tool": "serp_query",
        "pmax_tool": "pmax_query",
        "sqr_tool": "sqr_query",
        "knowledge_query": "general_query",  # handled via fallback docs below
    }
    intent = alias_map.get(intent, intent)

    # Route early for plugin intents
    if intent in {"creative_query", "serp_query", "pmax_query", "sqr_query"}:
        if intent == "creative_query":
            product_ctx = (payload.get("productContext") or payload.get("product") or message).strip()
            tone = (payload.get("tone") or "neutral").strip()
            context = CreativeContext(
                final_url="",
                keywords=[product_ctx],
                business_name="",
                usp_list=[product_ctx],
            )
            creative = CreativeFactory.generate_ad_copy(
                context=context,
                tone=tone,
                tenant_id=tenant_id,
            )
            return func.HttpResponse(
                json.dumps({"intent": intent, "creative": creative}),
                status_code=200,
                mimetype="application/json",
                headers={"Access-Control-Allow-Origin": "*"},
            )
        if intent == "serp_query":
            urls = re.findall(r"https?://\S+", message)
            if not urls:
                return error_response("No URLs found to check.", 400)
            health = check_url_health(urls)
            return func.HttpResponse(
                json.dumps({"intent": intent, "results": health}),
                status_code=200,
                mimetype="application/json",
                headers={"Access-Control-Allow-Origin": "*"},
            )
        if intent in {"pmax_query", "sqr_query"}:
            if not requested_account:
                return error_response("accountName is required for PMax/SQR views.", 400)
            try:
                audit_payload = _run_audit(
                    account_name=requested_account,
                    overrides=payload.get("context") or {},
                )
                audit_result = audit_payload.get("result", {})
                if intent == "pmax_query":
                    analyzer = PMaxAnalyzer()
                    # Attempt to pull asset groups / placements if present in result.
                    asset_groups = audit_result.get("asset_groups") or audit_result.get("pmax_asset_groups") or []
                    placements = audit_result.get("placements") or audit_result.get("pmax_placements") or []
                    total_cost = audit_result.get("total_cost", 0.0)
                    shopping_cost = audit_result.get("shopping_cost", 0.0)
                    video_cost = audit_result.get("video_cost", 0.0)
                    view = analyzer.analyze(
                        asset_groups=asset_groups,
                        placements=placements,
                        total_cost=total_cost,
                        shopping_cost=shopping_cost,
                        video_cost=video_cost,
                    )
                else:
                    analyzer = SqrAnalyzer()
                    search_terms = audit_result.get("search_terms") or audit_result.get("sqr_terms") or []
                    view = analyzer.analyze(search_terms, target_cpa=payload.get("targetCPA") or 50.0)
                return func.HttpResponse(
                    json.dumps(
                        {
                            "intent": intent,
                            "view": view,
                            "businessContext": audit_payload.get("business_context", {}),
                        }
                    ),
                    status_code=200,
                    mimetype="application/json",
                    headers={"Access-Control-Allow-Origin": "*"},
                )
            except Exception as exc:  # noqa: BLE001
                logging.exception("Plugin audit extraction failed")
                return error_response(f"Plugin view failed: {exc}", 500)

    try:
        session_entity = _ensure_session_entity(session_id, tenant_id, requested_account)
        existing_account = session_entity.get("accountName")
        if requested_account and existing_account and requested_account != existing_account:
            return error_response("accountName mismatch for this session", 403)
        account_name = existing_account or requested_account
        if not account_name:
            return error_response("accountName is required for concierge access", 400)

        save_message(session_id, tenant_id, "user", message)

        history = get_history(session_id, tenant_id, limit=10)
        context_payload = payload.get("context") or {}
        account_context = classify_business_context(account_name, context_payload)
        from kai_core.shared.business_physics import evaluate_guardrails  # type: ignore

        guardrail_findings = evaluate_guardrails(account_context, context_payload.get("currentROAS"))
        for finding in guardrail_findings:
            logging.warning(
                "[Concierge] Guardrail %s/%s — %s",
                finding.level,
                finding.section,
                finding.detail,
            )

        audit_chunks: List[Dict] = []
        web_chunks: List[Dict] = []
        if account_name:
            audit_chunks = retrieve_audit_context(message, account_name)
        if intent == "best_practice_query":
            web_chunks = search_best_practices(message)
        # Fallback: if no audit context and general/help queries, use compliance doc
        if not audit_chunks and intent in {"general_query", "help_query", "best_practice_query"}:
            audit_chunks = _load_local_docs()

        context_message = build_context_message(audit_chunks, web_chunks)

        # Audience-aware persona: client-friendly vs internal/analyst detail
        if audience in ("internal", "analyst", "owner"):
            style_hint = (
                "Audience: INTERNAL ANALYST.\n"
                "- You are speaking to channel experts and system owners.\n"
                "- It is appropriate to reference specific levers such as Smart Bidding coverage, "
                "impression share lost to budget, match-type mix, PMAX adoption, and telemetry.\n"
                "- Still, avoid saying that you are an AI or referencing models; respond as a senior human consultant.\n"
                "- Organise explanations by the seven themes where relevant:\n"
                "  Transparency & Campaign Settings; Strategic Alignment; Media & Channel Planning; "
                "Data, Targeting, and Technology; Optimization & Automation; "
                "Creative, Messaging, and Formats; Measurement.\n"
                "- For each theme you touch, call out: the key data points, why they matter, and the practical next steps.\n"
            )
        else:
            style_hint = (
                "Audience: CLIENT / BUSINESS STAKEHOLDER.\n"
                "- Use plain business language and minimise search jargon.\n"
                "- Lead with business impact, then briefly explain the mechanics only when needed.\n"
                "- Do not mention Smart Bidding, PMAX, match types, or other platform features unless "
                "they are essential—and when you do, translate them into everyday terms.\n"
                "- Frame answers through the seven themes where helpful:\n"
                "  Transparency & Campaign Settings; Strategic Alignment; Media & Channel Planning; "
                "Data, Targeting, and Technology; Optimization & Automation; "
                "Creative, Messaging, and Formats; Measurement.\n"
                "- Keep responses concise, executive-friendly, and focused on what to do next.\n"
            )

        base_prompt = get_system_prompt()
        persona_prompt = (
            base_prompt
            + "\n\n"
            "When answering questions, you draw from three types of evidence when available:\n"
            "1) HUMAN ANALYSIS: Auditor-written assessments, scores, and narrative insights\n"
            "2) MACHINE METRICS (signals): Calculated indicators such as automation coverage, budget concentration, and query breadth\n"
            "3) QA NOTES: Checks that compare human scores to machine metrics and flag contradictions\n\n"
            "Citations and data use:\n"
            "- When you reference audit content, keep inline markers like [1], [2] for audit references and [W1], [W2] for web sources when they are provided.\n"
            "- If Machine Metrics are present, include them alongside human insights to strengthen your case.\n"
            "- If QA Notes flag contradictions (for example, strong scores but weak metrics), acknowledge both and explain likely reasons.\n"
            "- When data is missing, state that limitation clearly and avoid guessing numbers.\n\n"
            f"{style_hint}"
        )

        guardrail_brief = ""
        if guardrail_findings:
            guardrail_lines = [
                f"{finding.level}: {finding.detail} Action: {finding.action}"
                for finding in guardrail_findings
            ]
            guardrail_brief = "Guardrail findings:\n" + "\n".join(guardrail_lines)

        chat_messages = [
            {
                "role": "system",
                "content": persona_prompt,
            },
            {"role": "system", "content": context_message},
        ]
        if guardrail_brief:
            chat_messages.append({"role": "system", "content": guardrail_brief})
        system_override = get_system_prompt_override()
        if system_override:
            chat_messages.insert(0, {"role": "system", "content": system_override})
        if is_maintenance_mode():
            chat_messages.insert(0, {"role": "system", "content": "Maintenance mode is enabled for this session."})
        chat_messages.extend(history)
        chat_messages.append({"role": "user", "content": message})

        ai_reply = call_azure_openai(chat_messages, session_id=session_id, intent=intent, tenant_id=tenant_id)

        save_message(session_id, tenant_id, "assistant", ai_reply, intent)
        _update_session_metadata(session_id, tenant_id, intent)
        if audit_chunks or web_chunks:
            record_retrieval(
                session_id,
                tenant_id,
                account_name or "unknown",
                message,
                [chunk["id"] for chunk in audit_chunks],
                intent,
                len(audit_chunks) + len(web_chunks),
            )

        response_payload = compose_response_payload(ai_reply, intent, session_id, audit_chunks, web_chunks)
        response_payload["context"] = {
            "businessContext": account_context.model_dump(),
            "guardrailFindings": [finding.model_dump() for finding in guardrail_findings],
        }
        return func.HttpResponse(
            json.dumps(response_payload),
            status_code=200,
            mimetype="application/json",
            headers={"Access-Control-Allow-Origin": "*"},
        )

    except requests.HTTPError as http_err:
        logging.error("Azure OpenAI error: %s", http_err.response.text if http_err.response else http_err)
        return error_response("AI service temporarily unavailable", 502)
    except Exception as exc:
        logging.exception("Concierge failure")
        return error_response(str(exc), 500)
