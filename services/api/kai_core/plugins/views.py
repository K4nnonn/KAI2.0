"""
Audit view extractors for PMax and SQR summaries.
"""
from __future__ import annotations

from typing import Any, Dict, List


def _matches(term: str, needles: List[str]) -> bool:
    lowered = term.lower()
    return any(n.lower() in lowered for n in needles)


def extract_pmax_view(audit_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Filter guardrail findings for PMax/Performance Max related items.
    """
    findings = audit_result.get("guardrail_findings") or []
    needles = ["pmax", "performance max"]
    return [f for f in findings if _matches(str(f), needles)]


def extract_sqr_view(audit_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Filter guardrail findings for search term / keyword / negative / match type.
    """
    findings = audit_result.get("guardrail_findings") or []
    needles = ["search term", "keyword", "negative", "match type", "query"]
    return [f for f in findings if _matches(str(f), needles)]
