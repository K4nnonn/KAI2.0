"""
Insights Injector - AI-Generated Executive Summary for Audit Reports

Fills the blank 'Insights' sheet in audit Excel files with GPT-4 strategic analysis.

Process:
1. Receives workbook + scoring results from generate_audit()
2. Analyzes scores to identify opportunities and risks
3. Calls Azure OpenAI to generate 3-bullet executive summary
4. Writes summary to 'Insights' sheet cell B5
"""

import json
import logging
import os
import time
from typing import Dict, Optional, Any
from openpyxl import Workbook
import requests

from kai_core.telemetry import log_openai_usage
from kai_core.shared.ai_sync import audit_persona_prefix
from kai_core.config import is_azure_openai_enabled
from kai_core.shared.azure_budget import allow_azure_usage


def inject_insights(
    workbook: Workbook,
    scoring_results: Dict[str, Any],
    account_name: str = "Account",
    enable_ai: bool = True
) -> None:
    """
    Generate and inject AI insights into the Insights sheet.

    Args:
        workbook: openpyxl Workbook object (already populated with scoring data)
        scoring_results: Dictionary returned from generate_audit() containing:
            - scored: int (number of criteria scored)
            - not_applicable: int
            - needs_data: int
            - overall_score: float (weighted average)
            - scope_summary: dict (data coverage info)
        account_name: Name of the account being audited
        enable_ai: Whether to use AI for insights generation (default: True)

    Side Effects:
        - Locates 'Insights' sheet in workbook
        - Writes AI-generated summary to cell B5
        - Logs OpenAI API usage
    """

    # Check if AI is enabled by user toggle
    if not enable_ai:
        logging.info("[Insights Injector] AI disabled by user - using fallback insights")
        _write_fallback_insights(workbook, scoring_results, account_name)
        return

    # Check if AI is enabled and configured
    if not _is_ai_configured():
        logging.warning("[Insights Injector] Azure OpenAI not configured - skipping insights generation")
        _write_fallback_insights(workbook, scoring_results, account_name)
        return

    # Generate insights via LLM
    logging.info(f"[Insights Injector] Generating executive summary for {account_name}...")
    insights_text = _generate_insights_via_llm(scoring_results, account_name)

    if not insights_text:
        logging.warning("[Insights Injector] LLM call failed - using fallback insights")
        _write_fallback_insights(workbook, scoring_results, account_name)
        return

    # Write to Insights sheet
    _write_to_insights_sheet(workbook, insights_text)
    logging.info("[Insights Injector] Successfully injected AI insights into Excel")


def _is_ai_configured() -> bool:
    """Check if Azure OpenAI credentials are available."""
    if not is_azure_openai_enabled():
        return False
    if os.environ.get("INSIGHT_INJECTOR_ALLOW_AZURE", "true").lower() != "true":
        return False
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY")
    deployment = (
        os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        or os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
    )
    return all([endpoint, key, deployment])


def _generate_insights_via_llm(
    scoring_results: Dict[str, Any],
    account_name: str
) -> Optional[str]:
    """
    Call Azure OpenAI to generate strategic insights.

    Returns:
        str: 3-bullet executive summary, or None if call fails
    """

    # Build context for LLM
    scored = scoring_results.get('scored', 0)
    total_evaluated = scored + scoring_results.get('not_applicable', 0)
    overall_score = scoring_results.get('overall_score', 0.0)
    needs_data = scoring_results.get('needs_data', 0)
    scope = scoring_results.get('scope_summary', {})

    # Create prompt
    prompt = f"""Analyze this PPC audit for {account_name}:

AUDIT RESULTS:
- Overall Score: {overall_score:.2f}/5.0
- Criteria Scored: {scored}
- Not Applicable: {scoring_results.get('not_applicable', 0)}
- Data Needed: {needs_data}
- Coverage: {scored}/{total_evaluated} criteria evaluated

DATA SCOPE:
{json.dumps(scope, indent=2) if scope else 'No scope data available'}

TASK: Write a 3-bullet executive summary. Voice: elite human strategist—confident, crisp, and clear. Be concrete, avoid boilerplate, and vary wording. Each bullet must be unique and grounded in the data above.

• Biggest Opportunity: single sentence on the highest-impact improvement area + a short “because…” rationale tied to the metrics.
• Biggest Risk: single sentence on the most critical gap + a short “because…” rationale tied to the metrics.
• Key Recommendation: single sentence on the most actionable next step + a short “because…” rationale tied to the metrics.

Constraints:
- Ground every bullet in the audit metrics (no invented KPIs).
- Concise, exec-ready tone; no filler.
- Do not repeat phrases across bullets; avoid generic wording unless tied directly to the data scope.
- No placeholders—each bullet must be bespoke to this account’s data.
- Do not use em dashes; prefer clear connectors like “because” or “which”."""

    allowed, reason = allow_azure_usage(module="insight_injector", purpose="insights")
    if not allowed:
        logging.warning("[Insights Injector] Azure blocked by policy: %s", reason)
        return None

    # Call Azure OpenAI
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY")
    deployment = (
        os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        or os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
        or "gpt-4-turbo"
    )
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "api-key": key
    }
    system_prompt = (
        audit_persona_prefix()
        + " You are producing an executive summary for the Insights sheet. "
        "Keep it client-ready, concrete, and grounded in the provided audit data."
    )
    body = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 400
    }

    # Make request with retries
    max_retries = 3
    retry_backoff = 2.0

    for attempt in range(max_retries):
        try:
            start_time = time.perf_counter()
            response = requests.post(
                url,
                params={"api-version": api_version},
                headers=headers,
                json=body,
                timeout=30
            )
            latency_ms = (time.perf_counter() - start_time) * 1000

            response.raise_for_status()
            data = response.json()

            # Extract content
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            # Log telemetry
            usage = data.get("usage", {})
            log_openai_usage(
                source="insight_injector",
                metadata={
                    "account": account_name,
                    "overall_score": overall_score,
                    "criteria_scored": scored
                },
                usage=usage,
                latency_ms=latency_ms
            )

            return content

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                # Rate limit - wait and retry
                wait_time = retry_backoff ** attempt
                logging.warning(f"[Insights Injector] Rate limited, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                logging.error(f"[Insights Injector] HTTP error: {e}")
                return None

        except Exception as e:
            logging.error(f"[Insights Injector] Error calling Azure OpenAI: {e}")
            return None

    logging.error("[Insights Injector] Max retries exceeded")
    return None


def _write_to_insights_sheet(workbook: Workbook, insights_text: str) -> None:
    """Write insights to the 'Insights' sheet at cell B5."""

    # Look for Insights sheet (case-insensitive); create if missing to avoid silent skips.
    insights_sheet = None
    for sheet_name in workbook.sheetnames:
        if sheet_name.lower() == 'insights':
            insights_sheet = workbook[sheet_name]
            break

    if insights_sheet is None:
        insights_sheet = workbook.create_sheet("Insights")
        # Light structure for readability
        insights_sheet.column_dimensions['B'].width = 80
        insights_sheet['B3'] = "Executive Summary"
        logging.info("[Insights Injector] Created missing 'Insights' sheet")

    # Write to cell B5 (standard location for executive summary)
    insights_sheet['B5'] = insights_text

    # Optional: Add metadata
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    insights_sheet['B4'] = f"AI-Generated Executive Summary ({timestamp})"


def _write_fallback_insights(
    workbook: Workbook,
    scoring_results: Dict[str, Any],
    account_name: str
) -> None:
    """Write basic fallback insights when AI is unavailable."""

    overall_score = scoring_results.get('overall_score', 0.0)
    scored = scoring_results.get('scored', 0)
    needs_data = scoring_results.get('needs_data', 0)

    # Generate simple text-based summary
    if overall_score >= 4.0:
        opportunity = "Account shows strong optimization across most criteria"
        risk = "Maintain current performance and monitor for emerging gaps"
    elif overall_score >= 3.0:
        opportunity = "Several high-impact optimization opportunities identified in the audit"
        risk = "Address data gaps to ensure comprehensive coverage"
    else:
        opportunity = "Significant optimization potential across multiple categories"
        risk = "Low scores indicate urgent need for structural improvements"

    fallback_text = f"""• **Overall Assessment:** Account scored {overall_score:.2f}/5.0 across {scored} evaluated criteria

• **Opportunity:** {opportunity}

• **Risk:** {risk}

• **Data Coverage:** {needs_data} criteria require additional data for complete evaluation

[Note: AI insights unavailable - Azure OpenAI not configured]"""

    _write_to_insights_sheet(workbook, fallback_text)
