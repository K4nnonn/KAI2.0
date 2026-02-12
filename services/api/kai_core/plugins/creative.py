"""
AI Creative Plugin (satellite) - generates ad copy and validates against Google Ads limits.
"""
from __future__ import annotations

import json
import importlib
import os
from typing import Dict, List
import requests
from kai_core.shared.adapters import AdCopyValidator, CreativeContext, calculate_google_ads_width


class CreativeFactory:
    """
    Generates RSA-style copy using the shared OpenAI wrapper and validates outputs.
    """

    SYSTEM_PROMPT = (
        "You are a PPC Copywriter. Create responsive search ad copy that follows Google Ads limits.\n"
        "- Provide 3 Headlines (max 30 char display width) and 2 Descriptions (max 90 char display width).\n"
        "- Keep punctuation compliant (no headline exclamation marks, no gimmicky symbols).\n"
        "- Return JSON with keys: headlines (list), descriptions (list).\n"
    )

    @classmethod
    def generate_ad_copy(cls, context: CreativeContext, tone: str = "neutral", tenant_id: str | None = None) -> Dict[str, List[str] | str]:
        """
        Generate copy for the given creative context; validate and truncate if needed.

        Returns a dict: {headlines, descriptions, raw}
        """
        concierge_mod = importlib.import_module("kai_core.Concierge")
        call_azure_openai = getattr(concierge_mod, "call_azure_openai")
        user_prompt = (
            f"Final URL: {context.final_url}\n"
            f"Business Name: {context.business_name}\n"
            f"Keywords: {', '.join(context.keywords)}\n"
            f"USPs: {', '.join(context.usp_list)}\n"
            f"Tone: {tone}"
        )
        messages = [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        raw = None
        try:
            raw = call_azure_openai(messages, tenant_id=tenant_id)
        except Exception:
            raw = None

        if not raw:
            raw = cls._call_local_ollama(messages)

        if not raw:
            raw = cls._fallback_copy(context, tone)
        headlines: List[str] = []
        descriptions: List[str] = []

        # Try to parse JSON; fallback to line parsing
        try:
            parsed = json.loads(raw)
            headlines = parsed.get("headlines", []) if isinstance(parsed, dict) else []
            descriptions = parsed.get("descriptions", []) if isinstance(parsed, dict) else []
        except Exception:
            lines = [l.strip("-• ").strip() for l in raw.splitlines() if l.strip()]
            headlines = [l for l in lines if len(headlines) < 3]
            descriptions = [l for l in lines[len(headlines):] if len(descriptions) < 2]

        # Sanitize/validate; truncate if needed
        def _truncate_to_limit(text: str, limit: int) -> str:
            acc = ""
            for ch in text:
                next_w = calculate_google_ads_width(acc + ch)
                if next_w > limit:
                    break
                acc += ch
            return acc

        cleaned_headlines = []
        for h in headlines[:3]:
            sanitized = AdCopyValidator.sanitize_headline(h)
            if not AdCopyValidator.is_valid_headline(sanitized):
                sanitized = _truncate_to_limit(sanitized, AdCopyValidator.HEADLINE_LIMIT)
            cleaned_headlines.append(sanitized)

        cleaned_descriptions = []
        for d in descriptions[:2]:
            sanitized = AdCopyValidator.sanitize_description(d)
            if not AdCopyValidator.is_valid_description(sanitized):
                sanitized = _truncate_to_limit(sanitized, AdCopyValidator.DESC_LIMIT)
            cleaned_descriptions.append(sanitized)

        return {
            "headlines": cleaned_headlines,
            "descriptions": cleaned_descriptions,
            "raw": raw,
        }

    @staticmethod
    def _call_local_ollama(messages: list[dict]) -> str | None:
        endpoint = (os.environ.get("LOCAL_LLM_ENDPOINT") or "").strip()
        if not endpoint:
            return None
        model = (os.environ.get("LOCAL_LLM_MODEL") or "llama3").strip()
        timeout_env = os.environ.get("LOCAL_LLM_TIMEOUT_SECONDS")
        timeout = float(timeout_env or "25")

        base = endpoint.rstrip("/")
        if base.endswith("/api/chat"):
            url = base
        elif base.endswith("/api"):
            url = f"{base}/chat"
        else:
            url = f"{base}/api/chat"

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2, "num_predict": 320},
        }
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                content = (
                    data.get("message", {}).get("content")
                    or data.get("response")
                    or data.get("text")
                )
                if isinstance(content, str) and content.strip():
                    return content.strip()
        except Exception:
            return None
        return None

    @staticmethod
    def _fallback_copy(context: CreativeContext, tone: str) -> str:
        business = context.business_name or "Your Business"
        keyword = (context.keywords or ["Your Service"])[0]
        headlines = [
            f"{business} Official Site",
            f"{keyword} Solutions",
            f"Trusted {business} Team",
        ]
        descriptions = [
            f"Explore {business} options tailored to {keyword.lower()}.",
            f"Get reliable service and fast support from {business}.",
        ]
        payload = {"headlines": headlines, "descriptions": descriptions, "tone": tone, "fallback": True}
        return json.dumps(payload)
