"""
TeamsConcierge - modular bridge for future Microsoft Teams integration.

This function is intentionally thin and optional:
- It does NOT change existing Kai or GenerateAudit behavior.
- It simply translates an incoming chat-style payload into a Concierge call.
- It remains dormant until CONCIERGE_FUNCTION_URL is configured and Teams is wired.

Expected payload (can be adapted by a Bot/Logic App later):
{
  "text": "question or instruction",
  "sessionId": "teams-session-or-conversation-id",
  "accountName": "Brand",
  "audience": "client" | "internal"  # optional, defaults to client
}
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import azure.functions as func
import requests


def _build_concierge_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a generic chat payload into the Concierge request shape.

    This keeps the bridge modular: when a real Teams bot is added, it can
    transform the native Teams event into this simple schema and post here.
    """
    text = (body.get("text") or body.get("message") or "").strip()
    if not text:
        raise ValueError("Missing 'text' in request payload")

    # Session anchoring: prefer explicit sessionId, otherwise fall back
    # to composite identifiers if provided.
    session_id = (body.get("sessionId") or "").strip()
    if not session_id:
        tenant = (body.get("tenantId") or "").strip()
        channel = (body.get("channelId") or "").strip()
        user = (body.get("userId") or "").strip()
        parts = [p for p in (tenant, channel or user) if p]
        if parts:
            session_id = "|".join(parts)
        else:
            raise ValueError("Missing 'sessionId' and no fallback identifiers provided")

    account_name = (body.get("accountName") or "").strip()
    audience = (body.get("audience") or body.get("tone") or "client").strip().lower()
    tenant_id = (body.get("tenantId") or body.get("tenant_id") or body.get("tenant") or "").strip()

    payload: Dict[str, Any] = {
        "message": text,
        "sessionId": session_id,
        "accountName": account_name,
        "audience": audience,
    }
    context_fields = {
        "vertical": body.get("vertical"),
        "margin": body.get("margin"),
        "grossMargin": body.get("grossMargin"),
        "targetROAS": body.get("targetROAS"),
        "conversions30d": body.get("conversions30d"),
        "currentROAS": body.get("currentROAS"),
        "targetCountry": body.get("targetCountry"),
        "verificationStatus": body.get("verificationStatus"),
    }
    context_payload = body.get("context") or {}
    for key, value in context_fields.items():
        if value is not None:
            context_payload.setdefault(key, value)
    if context_payload:
        payload["context"] = context_payload
    if tenant_id:
        payload["tenantId"] = tenant_id
    return payload


def format_adaptive_card(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal Adaptive Card formatter for Teams responses (JSON schema).
    """
    title = result.get("title") or "Kai Response"
    body_text = result.get("text") or result.get("reply") or "Completed"
    sections = []
    if result.get("summary"):
        sections.append({"type": "TextBlock", "text": f"Summary: {result['summary']}", "wrap": True})
    if result.get("details"):
        sections.append({"type": "TextBlock", "text": f"Details: {result['details']}", "wrap": True})
    card = {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": [
            {"type": "TextBlock", "text": title, "weight": "Bolder", "size": "Medium"},
            {"type": "TextBlock", "text": body_text, "wrap": True},
            *sections,
        ],
        "actions": [],
    }
    return card


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("TeamsConcierge bridge invoked")

    if req.method == "OPTIONS":
        return func.HttpResponse(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )

    try:
        body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON payload"}),
            status_code=400,
            mimetype="application/json",
            headers={"Access-Control-Allow-Origin": "*"},
        )

    try:
        concierge_payload = _build_concierge_payload(body)
    except ValueError as exc:
        return func.HttpResponse(
            json.dumps({"error": str(exc)}),
            status_code=400,
            mimetype="application/json",
            headers={"Access-Control-Allow-Origin": "*"},
        )

    concierge_url = os.environ.get("CONCIERGE_FUNCTION_URL")
    if not concierge_url:
        # Bridge is intentionally dormant until configured.
        logging.warning("CONCIERGE_FUNCTION_URL is not set; TeamsConcierge is not configured.")
        return func.HttpResponse(
            json.dumps(
                {
                    "error": "Teams bridge is not yet configured. "
                    "Set CONCIERGE_FUNCTION_URL to enable this endpoint."
                }
            ),
            status_code=503,
            mimetype="application/json",
            headers={"Access-Control-Allow-Origin": "*"},
        )

    try:
        resp = requests.post(concierge_url, json=concierge_payload, timeout=30)
        resp.raise_for_status()
    except requests.HTTPError as http_err:
        logging.error("TeamsConcierge -> Concierge HTTP error: %s", http_err)
        return func.HttpResponse(
            json.dumps({"error": "Upstream concierge call failed", "details": str(http_err)}),
            status_code=502,
            mimetype="application/json",
            headers={"Access-Control-Allow-Origin": "*"},
        )
    except Exception as exc:
        logging.exception("TeamsConcierge unexpected failure")
        return func.HttpResponse(
            json.dumps({"error": "Unexpected bridge error", "details": str(exc)}),
            status_code=500,
            mimetype="application/json",
            headers={"Access-Control-Allow-Origin": "*"},
        )

    # For now we just relay the Concierge JSON response. A future Teams
    # adapter (Bot Framework, Logic App, etc.) can transform this into
    # Adaptive Cards or Teams message formats.
    return func.HttpResponse(
        resp.text,
        status_code=resp.status_code,
        mimetype="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )
