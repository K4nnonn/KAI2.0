import ast
import json
import hashlib
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import requests

from kai_core.shared.ai_sync import BASE_PERSONA, audit_persona_prefix
from kai_core.config import get_deployment_mode, is_azure_openai_enabled
from kai_core.shared.azure_budget import allow_azure_usage


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def _normalize_sentence(sentence: str) -> str:
    s = sentence.lower()
    s = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _contradicts_signal(source: str, candidate: str) -> bool:
    if not source or not candidate:
        return False
    src = source.lower()
    cand = candidate.lower()
    negatives = ("very low", "low", "below", "under target", "missing", "0.0%")
    positives = (
        "excellent",
        "strong",
        "high",
        "healthy",
        "above",
        "good",
        "meets",
        "compliant",
        "sufficient",
        "adequate",
        "effective",
    )
    src_neg = any(tok in src for tok in negatives)
    src_pos = any(tok in src for tok in positives)
    cand_neg = any(tok in cand for tok in negatives)
    cand_pos = any(tok in cand for tok in positives)
    return (src_neg and cand_pos) or (src_pos and cand_neg)


def _contains_numeric_fact(source: str, candidate: str) -> bool:
    if not source or not candidate:
        return True
    numbers = re.findall(r"\d+(?:\.\d+)?%?", source)
    if not numbers:
        return True
    return any(num in candidate for num in numbers)


def _try_load_json(text: str) -> Optional[Dict[str, str]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(text)
        except Exception:
            return None
        return data if isinstance(data, dict) else None


def _extract_balanced_json(text: str) -> Optional[str]:
    start = None
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = idx
            depth += 1
        elif ch == "}" and start is not None:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _extract_json(text: str) -> Optional[Dict[str, str]]:
    if not text:
        return None
    cleaned = text.strip()
    for block in _CODE_FENCE_RE.findall(cleaned):
        data = _try_load_json(block.strip())
        if data:
            return data
    data = _try_load_json(cleaned)
    if data:
        return data
    candidate = _extract_balanced_json(cleaned)
    if not candidate:
        return None
    return _try_load_json(candidate.strip())


def _extract_labeled_sections(text: str) -> Optional[Dict[str, str]]:
    if not text:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    match = _CODE_FENCE_RE.search(cleaned)
    if match:
        cleaned = match.group(1).strip()
    pattern = re.compile(
        r"(?im)^\s*(?:\d+[\.\)]\s*)?(?:[-•]\s*)?(?:[*_`#]+\s*)?(details|actions|rationale)(?:\s*[*_`#]+)?(?:\s*[:\-\u2014]\s*|\s*$)"
    )
    matches = list(pattern.finditer(cleaned))
    if not matches:
        return None
    sections: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        label = match.group(1).lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
        sections[label] = cleaned[start:end].strip()
    return sections or None


def _extract_numbered_sections(text: str) -> Optional[Dict[str, str]]:
    if not text:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    lines = []
    for match in re.finditer(r"(?m)^\s*(?:\d+[\.\)]|[-*])\s+(.*)$", cleaned):
        line = match.group(1).strip()
        if line:
            lines.append(line)
    if len(lines) < 3:
        return None
    return {
        "details": lines[0],
        "actions": lines[1],
        "rationale": lines[2],
    }


def _looks_like_json_blob(text: str) -> bool:
    if not text:
        return False
    cleaned = text.strip()
    if cleaned.startswith("{"):
        return True
    if re.search(r'\"(details|actions|rationale)\"\s*:', cleaned):
        return True
    return False


def _trim_text(text: Optional[str], limit: int) -> Optional[str]:
    if not text:
        return text
    cleaned = str(text).strip()
    if limit <= 0 or len(cleaned) <= limit:
        return cleaned
    clipped = cleaned[: max(0, limit - 3)].rstrip()
    return f"{clipped}..."


def _normalize_for_overlap(text: str) -> str:
    if not text:
        return ""
    cleaned = text.lower()
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _shared_numbers(a: str, b: str) -> bool:
    if not a or not b:
        return False
    nums_a = set(re.findall(r"\d+(?:\.\d+)?%?", a))
    nums_b = set(re.findall(r"\d+(?:\.\d+)?%?", b))
    if not nums_a or not nums_b:
        return False
    return bool(nums_a & nums_b)


class AuditNarrativeHumanizer:
    def __init__(self) -> None:
        self.enabled = os.environ.get("ENABLE_AUDIT_VERBALIZER", "true").lower() == "true"
        self.mode = os.environ.get("AUDIT_VERBALIZER_MODE", "all").lower()
        self.max_retries = int(os.environ.get("AUDIT_VERBALIZER_RETRIES", "0") or "0")
        self.max_tokens = int(os.environ.get("AUDIT_VERBALIZER_MAX_TOKENS", "200") or "200")
        self.local_max_tokens = int(
            os.environ.get("AUDIT_VERBALIZER_LOCAL_MAX_TOKENS", "120") or "120"
        )
        self.label_first = (
            os.environ.get("AUDIT_VERBALIZER_LABEL_FIRST", "true").lower() == "true"
        )
        default_local_timeout = 20.0
        if os.environ.get("REQUIRE_LOCAL_LLM", "false").lower() == "true" and "AUDIT_VERBALIZER_LOCAL_TIMEOUT" not in os.environ:
            default_local_timeout = float(os.environ.get("LOCAL_LLM_REQUIRED_TIMEOUT_SECONDS", "45") or "45")
        self.local_timeout = float(os.environ.get("AUDIT_VERBALIZER_LOCAL_TIMEOUT", str(default_local_timeout)) or default_local_timeout)
        self.local_connect_timeout = float(
            os.environ.get("AUDIT_VERBALIZER_LOCAL_CONNECT_TIMEOUT", "5") or "5"
        )
        self.azure_timeout = float(os.environ.get("AUDIT_VERBALIZER_AZURE_TIMEOUT", "10") or "10")
        self.local_health_timeout = float(os.environ.get("LOCAL_LLM_HEALTH_TIMEOUT_SECONDS", "5") or "5")
        self.local_health_cache_seconds = float(os.environ.get("LOCAL_LLM_HEALTH_CACHE_SECONDS", "60") or "60")
        self.local_endpoint = os.environ.get("LOCAL_LLM_ENDPOINT")
        self.local_model = os.environ.get("LOCAL_LLM_MODEL", "llama3")
        self.local_enabled = os.environ.get("AUDIT_VERBALIZER_LOCAL_ENABLED", "true").lower() == "true"
        self.local_fail_fast = os.environ.get("AUDIT_VERBALIZER_LOCAL_FAIL_FAST", "true").lower() == "true"
        self.local_max_calls = int(os.environ.get("AUDIT_VERBALIZER_LOCAL_MAX_CALLS", "0") or "0")
        self.local_backoff_seconds = float(
            os.environ.get("AUDIT_VERBALIZER_LOCAL_BACKOFF_SECONDS", "0") or "0"
        )
        self.local_time_budget_seconds = float(
            os.environ.get("AUDIT_VERBALIZER_LOCAL_TIME_BUDGET_SECONDS", "0") or "0"
        )
        self._local_budget_start = None
        self._local_budget_exhausted = False
        self._local_backoff_until = 0.0
        require_local_env = os.environ.get("AUDIT_VERBALIZER_REQUIRE_LOCAL")
        if require_local_env is None:
            self.require_local = get_deployment_mode() == "LOCAL" and bool(self.local_endpoint)
        else:
            self.require_local = require_local_env.lower() == "true"
        if (
            self.require_local
            and "AUDIT_VERBALIZER_MODE" not in os.environ
            and self.mode == "all"
        ):
            self.mode = "templated"
        self.enforce_distinct = (
            os.environ.get("AUDIT_VERBALIZER_ENFORCE_DISTINCT", "true").lower() == "true"
        )
        if "LOCAL_LLM_NUM_CTX" not in os.environ:
            self.local_num_ctx = 1024 if self.require_local else 2048
        else:
            self.local_num_ctx = int(os.environ.get("LOCAL_LLM_NUM_CTX", "2048") or "2048")
        if self.require_local and "AUDIT_VERBALIZER_LOCAL_MAX_TOKENS" not in os.environ:
            self.local_max_tokens = min(self.local_max_tokens, 80)
        if self.require_local and "AUDIT_VERBALIZER_LABEL_FIRST" not in os.environ:
            self.label_first = False
        if (
            self.require_local
            and "AUDIT_VERBALIZER_RETRIES" not in os.environ
            and self.max_retries < 1
        ):
            self.max_retries = 1

        self.azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.azure_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY")
        self.azure_deployment = (
            os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            or os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
            or os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        )
        self.azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.azure_enabled = os.environ.get("AUDIT_VERBALIZER_ALLOW_AZURE", "true").lower() == "true"
        rpm_env = os.environ.get("AUDIT_VERBALIZER_RPM") or os.environ.get("AZURE_OPENAI_REQUESTS_PER_MINUTE")
        try:
            rpm_value = float(rpm_env) if rpm_env else 20.0
        except ValueError:
            rpm_value = 20.0
        rpm_value = max(1.0, rpm_value)
        self.azure_request_interval = 60.0 / rpm_value
        self._last_azure_ts = 0.0

        self.seen_sentences: set[str] = set()
        self._local_health_ok_until = 0.0
        self._local_health_err_until = 0.0
        self._local_health_err: Optional[str] = None
        self._require_local_checked = False
        self._require_local_error: Optional[str] = None
        self.stats = {
            "rows_total": 0,
            "rows_rewritten": 0,
            "local_calls": 0,
            "azure_calls": 0,
            "failures": 0,
            "fallback_used": 0,
            "repeats_detected": 0,
            "templated_detected": 0,
        }
        self.last_local_response: Optional[str] = None
        self.last_local_raw: Optional[Dict] = None

        self.templated_phrases = [
            "where to look:",
            "data needed:",
            "score context:",
            "needs immediate build or fix",
            "linear, time decay, position-based, data-driven",
            "excellent.",
            "non-existent.",
            "poor execution.",
            "requires significant improvement",
            "ok but lots of room for improvement",
            "needs improvement",
            "underperforming because",
            "fragile.",
            "needs immediate reinforcement",
            "kpi. and noiseaware comparison",
            "noiseaware comparison",
            "kpi.",
            "needs individualized definition based on business objective",
            "use: define baseline/target",
            "rerun the audit once the export is available",
            "ensure campaign export includes",
            "ensure ad export includes",
            "ensure keyword export includes",
            "automation manages dayparting",
            "maintain automation; smart bidding",
            "check creative tags",
            "check for valuetrack",
            "check tools & settings > measurement > conversions",
            "check which attribution model is active",
            "enable data-driven attribution",
            "ensure all key conversion actions are tracked",
            "export campaign details with negative keyword list information",
            "export search query report from sa360/google ads",
            "ad group report with active keyword counts",
            "campaigns randomly mix match types",
            "campaigns mix match types frequently",
            "implement valuetrack and utm parameters",
            "install remarketing tags",
            "maintain current best practice",
            "required for rlsa targeting",
            "review ad trafficking documentation",
            "sa360 typically doesn't include network targeting",
            "verify search partner inclusion",
        ]
        self.reject_log_path = os.environ.get("AUDIT_VERBALIZER_REJECT_LOG", "").strip()

        if not self.enabled:
            logging.info("[audit_verbalizer] disabled by flag")
        if self.enabled and not self.local_enabled:
            logging.info("[audit_verbalizer] local disabled by AUDIT_VERBALIZER_LOCAL_ENABLED")
        if self.enabled and not self.azure_enabled:
            logging.info("[audit_verbalizer] azure disabled by AUDIT_VERBALIZER_ALLOW_AZURE")

    def _append_reject_log(
        self,
        stage: str,
        reasons: List[str],
        context: Dict[str, Optional[str]],
        candidates: Optional[Dict[str, Optional[str]]] = None,
    ) -> None:
        if not self.reject_log_path:
            return
        payload = {
            "stage": stage,
            "reasons": reasons,
            "context": context,
        }
        if candidates:
            payload["candidates"] = {
                k: (v[:200] + "...") if isinstance(v, str) and len(v) > 200 else v
                for k, v in candidates.items()
            }
        try:
            with open(self.reject_log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except OSError:
            logging.warning("[audit_verbalizer] failed to write reject log")

    def _local_health_check(self, endpoint: str) -> Tuple[bool, Optional[str]]:
        """Probe local LLM health using /api/tags (Ollama-compatible). Cache result."""
        now = time.time()
        if now < self._local_health_ok_until:
            return True, None
        if now < self._local_health_err_until:
            return False, self._local_health_err
        try:
            _, tags_url = self._local_endpoints(endpoint)
            url = tags_url
            resp = requests.get(url, timeout=self.local_health_timeout)
            allow_404 = os.environ.get("LOCAL_LLM_HEALTH_ALLOW_404", "false").lower() == "true"
            if allow_404 and resp.status_code == 404:
                self._local_health_ok_until = now + self.local_health_cache_seconds
                self._local_health_err = None
                return True, None
            resp.raise_for_status()
            self._local_health_ok_until = now + self.local_health_cache_seconds
            self._local_health_err = None
            return True, None
        except Exception as exc:
            self._local_health_err = str(exc)
            self._local_health_err_until = now + self.local_health_cache_seconds
            logging.warning("[audit_verbalizer] local health check failed: %s", self._local_health_err)
            return False, self._local_health_err

    def _ensure_local_ready(self) -> Optional[str]:
        """Ensure local LLM is reachable when required; returns error string if not ready."""
        if not self.require_local:
            return None
        if not self.local_enabled:
            return "Local LLM required but AUDIT_VERBALIZER_LOCAL_ENABLED is false"
        if not self.local_endpoint:
            return "Local LLM required but LOCAL_LLM_ENDPOINT is not set"
        ok, err = self._local_health_check(self.local_endpoint.rstrip("/"))
        if not ok:
            return f"Local LLM required but health check failed: {err}"
        return None

    @staticmethod
    def _local_endpoints(endpoint: str) -> Tuple[str, str]:
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

    def reset(self) -> None:
        self.seen_sentences = set()

    def get_stats(self) -> Dict[str, int]:
        return dict(self.stats)

    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s.strip()]

    def _update_seen(self, texts: List[str]) -> None:
        for text in texts:
            for sentence in self._split_sentences(text):
                self.seen_sentences.add(_normalize_sentence(sentence))

    def _contains_templated_phrase(self, text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        return any(phrase in lowered for phrase in self.templated_phrases)

    def _has_templated(self, texts: List[str]) -> bool:
        return any(self._contains_templated_phrase(text) for text in texts if text)

    def _has_templated_for_mode(self, texts: List[str]) -> bool:
        if self.mode != "data_needed":
            return self._has_templated(texts)
        scrubbed: List[str] = []
        for text in texts:
            if not text:
                scrubbed.append(text)
                continue
            cleaned = re.sub(r"(?i)\\bdata needed\\b[:\\s]*", "", text)
            cleaned = re.sub(r"(?i)\\bwhere to look\\b[:\\s]*", "", cleaned)
            scrubbed.append(cleaned)
        return self._has_templated(scrubbed)

    @staticmethod
    def _strip_literal_prefixes(text: str) -> str:
        if not text:
            return text
        cleaned = text.strip().strip('"')
        # Remove embedded label phrases even when wrapped in quotes.
        cleaned = re.sub(r"(?i)\"?\bwhere[_\\s]?to[_\\s]?look\b\"?\s*(?:is|should be|:)?\s*", "", cleaned)
        cleaned = re.sub(r"(?i)\"?\bdata[_\\s]?needed\b\"?\s*(?:is|are|:)?\s*", "", cleaned)
        cleaned = re.sub(
            r"(?i)(?:^|\n)\s*(?:\"\s*)?(?:\d+[\.\)]\s*)?(?:[*_`#]+\s*)?(where to look|data needed|details|actions|rationale)(?:\s*[*_`#]+)?\s*[:\-\u2014]+\s*",
            "",
            cleaned,
        )
        return cleaned.strip()

    @staticmethod
    def _sanitize_llm_text(text: str) -> str:
        if not text:
            return text
        cleaned = text.strip()
        cleaned = cleaned.replace("\\n", " ").replace("\\$", "$").replace("—", "-")
        cleaned = cleaned.replace("\\\\", "\\")
        cleaned = re.sub(r"\\text\\{.*?\\}", "", cleaned)
        cleaned = re.sub(r"^\\s*\\d+\\.\\s*", "", cleaned)
        cleaned = re.sub(r"[*`#]+", "", cleaned)
        cleaned = re.sub(r"(?i)^(details|actions|rationale)\\s*[:\\-]\\s*", "", cleaned)
        cleaned = re.sub(r"\\s+", " ", cleaned).strip()
        cleaned = cleaned.encode("ascii", "ignore").decode()
        return cleaned

    def _de_template(self, text: str) -> str:
        if not text:
            return text
        cleaned = text
        for phrase in self.templated_phrases:
            cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\(\s*\)", "", cleaned)
        cleaned = re.sub(r"\.{2,}", ".", cleaned)
        cleaned = re.sub(r"\s+\.", ".", cleaned)
        cleaned = re.sub(r"(?i)^\s*and\s+", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" .;:-")
        return cleaned

    def _has_repeat(self, texts: List[str]) -> bool:
        for text in texts:
            for sentence in self._split_sentences(text):
                if _normalize_sentence(sentence) in self.seen_sentences:
                    return True
        return False

    @staticmethod
    def _has_internal_overlap(texts: List[str]) -> bool:
        if len(texts) < 2:
            return False
        normed = [_normalize_for_overlap(t or "") for t in texts]
        for i in range(len(normed)):
            for j in range(i + 1, len(normed)):
                a, b = normed[i], normed[j]
                if not a or not b:
                    continue
                if a == b:
                    return True
                if len(a) > 20 and a in b:
                    return True
                if len(b) > 20 and b in a:
                    return True
                a_tokens = set(a.split())
                b_tokens = set(b.split())
                if not a_tokens or not b_tokens:
                    continue
                overlap = len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))
                if overlap >= 0.8:
                    return True
        return False

    def _build_action_text(
        self,
        base_label: str,
        score: Optional[float],
        data_needed: Optional[str],
        where_to_look: Optional[str],
        detail_text: Optional[str],
    ) -> str:
        detail_text = detail_text or ""
        if where_to_look:
            short_ref = "the 'Where to look' references in this audit"
            if len(where_to_look) > 60 or where_to_look.lower() in detail_text.lower():
                return f"{base_label}: validate using {short_ref}."
            return f"{base_label}: review {where_to_look.strip()} and confirm the setting matches policy."
        if data_needed:
            short_ref = "the missing report listed in 'Data needed'"
            if len(data_needed) > 60 or data_needed.lower() in detail_text.lower():
                return f"{base_label}: provide {short_ref} to validate this check."
            return f"{base_label}: provide {data_needed.strip()} to validate this check."
        if score is None:
            return f"{base_label}: confirm the setting in the source system to validate this check."
        if isinstance(score, (int, float)) and score >= 4:
            return f"{base_label}: keep the current approach and monitor in the next export."
        return f"{base_label}: prioritize a targeted fix and re-audit after updates."

    def _build_rationale_text(
        self,
        base_label: str,
        score: Optional[float],
        data_needed: Optional[str],
        where_to_look: Optional[str],
        detail_text: Optional[str],
    ) -> str:
        detail_text = detail_text or ""
        if data_needed:
            short_ref = "the missing report listed in 'Data needed'"
            if len(data_needed) > 60 or data_needed.lower() in detail_text.lower():
                return f"{base_label}: without {short_ref}, this check cannot be verified accurately."
            return f"{base_label}: without {data_needed.strip()}, this check cannot be verified accurately."
        if where_to_look:
            short_ref = "the 'Where to look' references in this audit"
            if len(where_to_look) > 60 or where_to_look.lower() in detail_text.lower():
                return f"{base_label}: validating against {short_ref} prevents false positives."
            return f"{base_label}: confirming the setting in {where_to_look.strip()} prevents false positives."
        if score is None:
            return f"{base_label}: verification protects output accuracy for this criterion."
        if isinstance(score, (int, float)) and score >= 4:
            return f"{base_label}: strong performance here supports efficiency; keep it stable to avoid regression."
        return f"{base_label}: this area materially affects efficiency and scale; improving it typically unlocks gains."

    def _diversify_triplet(
        self,
        category: str,
        criterion: str,
        detail_text: str,
        action_text: str,
        rationale_text: str,
        score: Optional[float],
        data_needed: Optional[str],
        where_to_look: Optional[str],
    ) -> Tuple[str, str, str]:
        category = "" if category is None else str(category)
        criterion = "" if criterion is None else str(criterion)
        base_label = (criterion or "").strip()
        if category:
            category = category.strip()
            if base_label and category.lower() not in base_label.lower():
                base_label = f"{category} - {base_label}"
            elif not base_label:
                base_label = category
        if not base_label:
            base_label = "This check"
        detail_text = self._sanitize_llm_text(detail_text or "")
        action_text = self._sanitize_llm_text(action_text or "")
        rationale_text = self._sanitize_llm_text(rationale_text or "")

        if self._has_internal_overlap([detail_text, action_text]):
            action_text = self._build_action_text(base_label, score, data_needed, where_to_look, detail_text)
        if self._has_internal_overlap([detail_text, rationale_text]) or self._has_internal_overlap([action_text, rationale_text]):
            rationale_text = self._build_rationale_text(base_label, score, data_needed, where_to_look, detail_text)
        if self._has_internal_overlap([action_text, rationale_text]):
            rationale_text = (
                "Why it matters: verification protects audit accuracy and prevents mis-prioritized fixes."
            )

        return detail_text, action_text, rationale_text

    def _finalize_triplet(
        self,
        category: str,
        criterion: str,
        detail_text: str,
        action_text: str,
        rationale_text: str,
        score: Optional[float],
        data_needed: Optional[str],
        where_to_look: Optional[str],
    ) -> Tuple[str, str, str]:
        numeric_overlap = (
            _shared_numbers(detail_text, action_text)
            or _shared_numbers(detail_text, rationale_text)
            or _shared_numbers(action_text, rationale_text)
        )
        if self.enforce_distinct and (self._has_internal_overlap([detail_text, action_text, rationale_text]) or numeric_overlap):
            detail_text, action_text, rationale_text = self._diversify_triplet(
                category=category,
                criterion=criterion,
                detail_text=detail_text,
                action_text=action_text,
                rationale_text=rationale_text,
                score=score,
                data_needed=data_needed,
                where_to_look=where_to_look,
            )
        self._update_seen([detail_text or "", action_text or "", rationale_text or ""])
        return detail_text, action_text, rationale_text

    @staticmethod
    def _is_low_content(text: str, min_words: int = 6) -> bool:
        if not text:
            return True
        tokens = re.findall(r"[a-z0-9]{2,}", text.lower())
        return len(tokens) < min_words

    @staticmethod
    def _mentions_criterion(texts: List[str], criterion: str) -> bool:
        if not criterion:
            return True
        needle = criterion.lower()
        combined = " ".join(texts).lower()
        if needle in combined:
            return True
        tokens = re.findall(r"[a-z0-9]{3,}", needle)
        if not tokens:
            return True
        hits = sum(1 for token in tokens if token in combined)
        return hits >= min(2, len(tokens))

    @staticmethod
    def _mentions_data_needed(texts: List[str], data_needed: Optional[str]) -> bool:
        if not data_needed:
            return True
        tokens = re.findall(r"[a-z0-9]{4,}", data_needed.lower())
        if not tokens:
            return True
        combined = " ".join(texts).lower()
        generic = {"report", "campaign", "ad", "ads", "group", "keyword", "keywords", "data"}
        specific_tokens = [token for token in tokens if token not in generic]
        if specific_tokens:
            return any(token in combined for token in specific_tokens)
        return any(token in combined for token in tokens)

    @staticmethod
    def _mentions_where_to_look(texts: List[str], where_to_look: Optional[str]) -> bool:
        if not where_to_look:
            return True
        tokens = re.findall(r"[a-z0-9]{4,}", where_to_look.lower())
        if not tokens:
            return True
        combined = " ".join(texts).lower()
        digit_tokens = [token for token in tokens if any(ch.isdigit() for ch in token)]
        if digit_tokens:
            return any(token in combined for token in digit_tokens)
        return any(token in combined for token in tokens)

    def _needs_rewrite(self, texts: List[str]) -> bool:
        if not texts:
            return False
        if self.mode == "all":
            return True
        for text in texts:
            lowered = (text or "").lower()
            if any(phrase in lowered for phrase in self.templated_phrases):
                return True
            for sentence in self._split_sentences(text):
                if _normalize_sentence(sentence) in self.seen_sentences:
                    return True
        return False

    def _trim_payload_for_local(self, payload: Dict[str, str]) -> Dict[str, str]:
        require_local = os.environ.get("REQUIRE_LOCAL_LLM", "false").lower() == "true"
        max_chars = int(
            os.environ.get(
                "AUDIT_VERBALIZER_LOCAL_PAYLOAD_MAX_CHARS",
                "1800" if require_local else "2600",
            )
            or ("1800" if require_local else "2600")
        )
        max_field_chars = int(
            os.environ.get(
                "AUDIT_VERBALIZER_LOCAL_FIELD_MAX_CHARS",
                "300" if require_local else "420",
            )
            or ("300" if require_local else "420")
        )
        max_context_chars = int(
            os.environ.get(
                "AUDIT_VERBALIZER_LOCAL_CONTEXT_MAX_CHARS",
                "320" if require_local else "600",
            )
            or ("320" if require_local else "600")
        )
        max_avoid = int(
            os.environ.get(
                "AUDIT_VERBALIZER_LOCAL_AVOID_MAX",
                "12" if require_local else "24",
            )
            or ("12" if require_local else "24")
        )
        max_avoid_chars = int(
            os.environ.get(
                "AUDIT_VERBALIZER_LOCAL_AVOID_CHARS",
                "90" if require_local else "120",
            )
            or ("90" if require_local else "120")
        )

        trimmed = dict(payload)
        for key in ("details", "actions", "rationale", "where_to_look", "data_needed", "calculation"):
            if key in trimmed:
                trimmed[key] = _trim_text(trimmed.get(key), max_field_chars)

        context = trimmed.get("context")
        if context:
            context_json = json.dumps(context, ensure_ascii=False)
            if len(context_json) > max_context_chars:
                trimmed["context"] = {}

        avoid = trimmed.get("avoid_sentences") or []
        if max_avoid > 0 and avoid:
            avoid = [s for s in (_trim_text(s, max_avoid_chars) for s in avoid) if s]
            trimmed["avoid_sentences"] = avoid[:max_avoid]
        else:
            trimmed.pop("avoid_sentences", None)

        def _size() -> int:
            return len(json.dumps(trimmed, ensure_ascii=False))

        if _size() > max_chars:
            trimmed["context"] = {}
        if _size() > max_chars:
            trimmed["avoid_sentences"] = (trimmed.get("avoid_sentences") or [])[:10]
        if _size() > max_chars:
            for key in ("details", "actions", "rationale"):
                if key in trimmed:
                    trimmed[key] = _trim_text(trimmed.get(key), max(120, max_field_chars // 2))

        return trimmed

    def _call_local(
        self,
        messages: List[Dict[str, str]],
        strict: bool = False,
        force_json: Optional[bool] = None,
    ) -> Optional[str]:
        if not self.local_enabled:
            return None
        if not self.local_endpoint:
            return None
        if self._local_budget_exhausted:
            return None
        if self.local_time_budget_seconds > 0:
            if self._local_budget_start is None:
                self._local_budget_start = time.monotonic()
            elif (time.monotonic() - self._local_budget_start) > self.local_time_budget_seconds:
                self._local_budget_exhausted = True
                self._append_reject_log(
                    "local_budget",
                    ["budget_exhausted"],
                    {"mode": self.mode, "endpoint": self.local_endpoint},
                )
                return None
        if self.local_max_calls and self.stats["local_calls"] >= self.local_max_calls:
            return None
        if self.local_backoff_seconds > 0 and time.monotonic() < self._local_backoff_until:
            return None
        base_endpoint = self.local_endpoint.rstrip("/")
        ok, err = self._local_health_check(base_endpoint)
        if not ok:
            self._append_reject_log(
                "local_health",
                [err or "unhealthy"],
                {"mode": self.mode, "endpoint": base_endpoint},
            )
            return None
        endpoint, _ = self._local_endpoints(base_endpoint)
        payload = {
            "model": self.local_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": self.local_max_tokens,
                "num_ctx": self.local_num_ctx,
            },
        }
        if force_json is None:
            force_json = os.environ.get("LOCAL_LLM_FORCE_JSON", "true").lower() == "true"
        if strict or force_json:
            payload["format"] = "json"
        try:
            start = time.perf_counter()
            resp = requests.post(
                endpoint,
                json=payload,
                timeout=(self.local_connect_timeout, self.local_timeout),
            )
            resp.raise_for_status()
            data = resp.json()
            self.last_local_raw = data
            self.stats["local_calls"] += 1
            logging.info("[audit_verbalizer] local latency_ms=%.1f", (time.perf_counter() - start) * 1000)
            content = (
                data.get("message", {}).get("content")
                or data.get("content")
                or data.get("choices", [{}])[0].get("message", {}).get("content")
                or data.get("choices", [{}])[0].get("text")
                or data.get("response")
                or data.get("text")
            )
            self.last_local_response = content
            return content
        except Exception as exc:
            logging.warning("[audit_verbalizer] local failed: %s", exc)
            self._append_reject_log(
                "local_error",
                [str(exc)],
                {"mode": self.mode, "endpoint": endpoint},
            )
            if self.local_backoff_seconds > 0:
                self._local_backoff_until = time.monotonic() + self.local_backoff_seconds
            return None

    def _call_azure(self, messages: List[Dict[str, str]]) -> Optional[str]:
        if not self.azure_enabled or not is_azure_openai_enabled():
            return None
        allowed, reason = allow_azure_usage(module="audit_verbalizer", purpose="verbalizer")
        if not allowed:
            logging.info("[audit_verbalizer] azure blocked by policy: %s", reason)
            return None
        if not (self.azure_endpoint and self.azure_key and self.azure_deployment):
            return None
        if self.azure_request_interval > 0:
            now = time.monotonic()
            sleep_for = self.azure_request_interval - (now - self._last_azure_ts)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_azure_ts = time.monotonic()
        url = f"{self.azure_endpoint.rstrip('/')}/openai/deployments/{self.azure_deployment}/chat/completions"
        headers = {"Content-Type": "application/json", "api-key": self.azure_key}
        payload = {
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": self.max_tokens,
        }
        try:
            start = time.perf_counter()
            resp = requests.post(url, headers=headers, params={"api-version": self.azure_api_version}, json=payload, timeout=self.azure_timeout)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                try:
                    sleep_for = float(retry_after) if retry_after else 5.0
                except ValueError:
                    sleep_for = 5.0
                logging.warning("[audit_verbalizer] azure rate limited (429). Sleeping %.1fs", sleep_for)
                time.sleep(sleep_for)
                return None
            resp.raise_for_status()
            data = resp.json()
            self.stats["azure_calls"] += 1
            logging.info("[audit_verbalizer] azure latency_ms=%.1f", (time.perf_counter() - start) * 1000)
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            logging.warning("[audit_verbalizer] azure failed: %s", exc)
            return None

    def _build_messages(self, payload: Dict[str, str], strict: bool, compact: bool = False) -> List[Dict[str, str]]:
        use_compact = compact and os.environ.get("AUDIT_VERBALIZER_COMPACT_PROMPT", "true").lower() == "true"
        persona = audit_persona_prefix()
        if use_compact:
            system_prompt = (
                f"{persona} Return a single JSON object with keys: details, actions, rationale. "
                "No extra keys. No markdown. Keep each field 1-2 sentences and include the criterion name. "
                "If where_to_look is provided, mention at least one specific item from it."
            )
            user_prompt = json.dumps(payload)
        elif strict:
            system_prompt = (
                f"{persona} Rewrite audit text into concise, professional, human language. "
                "Use only facts from the payload. Do not add new metrics or assumptions. "
                "If where_to_look is present, incorporate its facts naturally without repeating the exact phrasing. "
                "If data_needed is present, explain the missing data and impact in plain language without boilerplate. "
                "Do not use the literal prefixes 'Where to look:' or 'Data needed:'. "
                "Include the criterion name in at least one field so each row is specific and unique. "
                "Return only a single JSON object with keys: details, actions, rationale. "
                "No markdown, no code fences, no extra text."
            )
            user_prompt = (
                "Return a single JSON object with double-quoted keys/values only.\n"
                "If unsure, keep meaning but rephrase to avoid repeated sentences.\n"
                f"{json.dumps(payload)}"
            )
        else:
            system_prompt = (
                f"{persona} You rewrite audit text into a concise, professional, human response. "
                "Only use facts in the payload. Do not add new metrics or assumptions. "
                "If where_to_look is present, incorporate its facts naturally without repeating the exact phrasing. "
                "If data_needed is present, explain the missing data and impact in plain language without boilerplate. "
                "Do not use the literal prefixes 'Where to look:' or 'Data needed:'. "
                "Include the criterion name in at least one field so each row is specific and unique. "
                "Avoid repeating prior sentences. Return JSON only and do not wrap in code fences."
            )
            user_prompt = (
                "Rewrite these fields as a human professional. Keep each field 1-2 sentences.\n"
                "Return JSON with keys: details, actions, rationale.\n"
                f"{json.dumps(payload)}"
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_label_messages(self, payload: Dict[str, str]) -> List[Dict[str, str]]:
        persona = audit_persona_prefix()
        payload = dict(payload)
        payload.pop("avoid_sentences", None)
        avoid_note = " Avoid boilerplate and do not echo the input phrasing."
        if self.mode == "data_needed":
            system_prompt = (
                f"{persona} Return exactly three lines:\n"
                "1) <details>\n"
                "2) <actions>\n"
                "3) <rationale>\n"
                "Use only facts from the payload. Do not add new metrics. "
                "Do not explain the payload or its fields. "
                "Include the criterion name in at least one line. "
                "If data_needed or where_to_look is present, explain it plainly without boilerplate. "
                f"No JSON.{avoid_note}"
            )
        else:
            system_prompt = (
                f"{persona} Rewrite into three labeled lines:\n"
                "Details: <1-2 sentences>\n"
                "Actions: <1-2 sentences>\n"
                "Rationale: <1-2 sentences>\n"
                "Use only facts from the payload. Do not add new metrics. "
                "Include the criterion name in at least one field. "
                "If data_needed or where_to_look is present, explain it plainly without boilerplate. "
                f"No JSON.{avoid_note}"
            )
        user_prompt = json.dumps(payload)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def rewrite_if_needed(
        self,
        account_name: str,
        category: str,
        criterion: str,
        detail_text: str,
        action_text: str,
        rationale_text: str,
        score: Optional[float],
        calculation: str,
        data_needed: Optional[str],
        context_summary: Optional[Dict[str, Dict]] = None,
        where_to_look: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        self.stats["rows_total"] += 1
        category = "" if category is None else str(category)
        criterion = "" if criterion is None else str(criterion)
        texts = [detail_text or "", action_text or "", rationale_text or ""]

        if not self.enabled:
            self._update_seen(texts)
            return detail_text, action_text, rationale_text

        if self.mode == "data_needed" and not data_needed and not where_to_look:
            self._update_seen(texts)
            return detail_text, action_text, rationale_text

        repeat_found = self._has_repeat(texts)
        templated_found = self._has_templated(texts)
        needs_rewrite = self._needs_rewrite(texts)
        rewrite_data_needed = os.environ.get("AUDIT_VERBALIZER_REWRITE_DATA_NEEDED", "true").lower() == "true"
        if not needs_rewrite and rewrite_data_needed and (data_needed or where_to_look):
            needs_rewrite = True
        # When rewrite_data_needed is enabled, data_needed/where_to_look rows are eligible for rewrites.
        # Only rewrite when we detect templated phrasing or repeats.
        if not needs_rewrite:
            if self.enforce_distinct and self._has_internal_overlap(texts):
                detail_text, action_text, rationale_text = self._diversify_triplet(
                    category=category,
                    criterion=criterion,
                    detail_text=detail_text or "",
                    action_text=action_text or "",
                    rationale_text=rationale_text or "",
                    score=score,
                    data_needed=data_needed,
                    where_to_look=where_to_look,
                )
            return self._finalize_triplet(
                category=category,
                criterion=criterion,
                detail_text=detail_text or "",
                action_text=action_text or "",
                rationale_text=rationale_text or "",
                score=score,
                data_needed=data_needed,
                where_to_look=where_to_look,
            )

        if self.local_enabled:
            local_ready_error = self._ensure_local_ready()
            if local_ready_error:
                self._append_reject_log(
                    "local_ready",
                    [local_ready_error],
                    {
                        "criterion": criterion,
                        "category": category,
                        "mode": self.mode,
                        "data_needed": data_needed,
                        "where_to_look": where_to_look,
                    },
                )
                self.stats["failures"] += 1
                raise RuntimeError(local_ready_error)

        # Light cleanup path: remove boilerplate before calling the LLM.
        # Applies to templated rows (including data_needed/where_to_look) to avoid slow LLM calls.
        if templated_found and self.mode in ("templated", "data_needed"):
            cleaned_details = self._de_template(detail_text or "")
            cleaned_actions = self._de_template(action_text or "")
            cleaned_rationale = self._de_template(rationale_text or "")
            cleaned_details = self._sanitize_llm_text(cleaned_details)
            cleaned_actions = self._sanitize_llm_text(cleaned_actions)
            cleaned_rationale = self._sanitize_llm_text(cleaned_rationale)
            base_label = (criterion or "").strip()
            if category:
                category = category.strip()
                if base_label and category.lower() not in base_label.lower():
                    base_label = f"{category} - {base_label}"
                elif not base_label:
                    base_label = category
            if not base_label:
                base_label = "This check"
            if not cleaned_details:
                cleaned_details = f"{base_label} needs review based on the current export."
            if not cleaned_actions:
                if where_to_look:
                    cleaned_actions = f"{base_label}: review {where_to_look.strip()} and confirm the setting matches policy."
                elif data_needed:
                    cleaned_actions = f"{base_label}: provide {data_needed.strip()} to validate this check."
                else:
                    cleaned_actions = f"{base_label}: confirm the setting in the source system."
            if not cleaned_rationale:
                if data_needed:
                    cleaned_rationale = f"{base_label}: missing data - {data_needed.strip()}."
                elif where_to_look:
                    cleaned_rationale = f"{base_label}: reference required - {where_to_look.strip()}."
                else:
                    cleaned_rationale = f"{base_label}: current exports do not include enough detail to verify this item."
            if data_needed and base_label.lower() not in cleaned_actions.lower():
                cleaned_actions = f"{base_label}: {cleaned_actions}"
            if data_needed and base_label.lower() not in cleaned_rationale.lower():
                cleaned_rationale = f"{base_label}: {cleaned_rationale}"
            if self._is_low_content(cleaned_details, min_words=4):
                cleaned_details = f"{base_label} needs review based on the current export."
            if self._is_low_content(cleaned_actions, min_words=4):
                cleaned_actions = f"{base_label}: confirm the setting in the source system."
            if self._is_low_content(cleaned_rationale, min_words=4):
                cleaned_rationale = f"{base_label}: current exports do not include enough detail to verify this item."
            if not self._has_templated([cleaned_details, cleaned_actions, cleaned_rationale]):
                self.stats["rows_rewritten"] += 1
                return self._finalize_triplet(
                    category=category,
                    criterion=criterion,
                    detail_text=cleaned_details or "",
                    action_text=cleaned_actions or "",
                    rationale_text=cleaned_rationale or "",
                    score=score,
                    data_needed=data_needed,
                    where_to_look=where_to_look,
                )

        if repeat_found:
            self.stats["repeats_detected"] += 1
        if templated_found:
            self.stats["templated_detected"] += 1
        if repeat_found or templated_found:
            reasons = []
            if repeat_found:
                reasons.append("repeat")
            if templated_found:
                reasons.append("templated")
            self._append_reject_log(
                "precheck",
                reasons,
                {
                    "criterion": criterion,
                    "category": category,
                    "mode": self.mode,
                    "data_needed": data_needed,
                    "where_to_look": where_to_look,
                },
            )

        avoid_sentences = sorted(self.seen_sentences)
        avoid_sentences.extend(self.templated_phrases)
        avoid_sentences = sorted(set(avoid_sentences))
        payload = {
            "account": account_name,
            "category": category,
            "criterion": criterion,
            "score": score,
            "details": detail_text,
            "actions": action_text,
            "rationale": rationale_text,
            "calculation": calculation,
            "data_needed": data_needed,
            "context": context_summary or {},
            "avoid_sentences": avoid_sentences[:80],
        }
        if where_to_look:
            payload["where_to_look"] = where_to_look

        if self.local_enabled:
            local_payload = {
                "account": account_name,
                "category": category,
                "criterion": criterion,
                "details": detail_text,
                "actions": action_text,
                "rationale": rationale_text,
                "avoid_sentences": avoid_sentences[:80],
            }
            if where_to_look:
                local_payload["where_to_look"] = where_to_look
            if data_needed:
                local_payload["data_needed"] = data_needed
            if self.mode == "data_needed":
                # Keep only the fields needed for minimal, fast rewrites.
                local_payload = {
                    "account": account_name,
                    "category": category,
                    "criterion": criterion,
                    "details": detail_text,
                    "actions": action_text,
                    "rationale": rationale_text,
                    "where_to_look": where_to_look,
                    "data_needed": data_needed,
                }
                local_payload = {k: v for k, v in local_payload.items() if v}
            local_payload = self._trim_payload_for_local(local_payload)
        else:
            local_payload = payload

        data = None
        last_response_text = None
        force_json = os.environ.get("LOCAL_LLM_FORCE_JSON", "true").lower() == "true"
        label_first = self.label_first
        compact_local = self.mode == "data_needed" or (
            self.mode == "templated" and (data_needed or where_to_look)
        )
        use_local_json = self.mode != "data_needed"
        local_failed = False
        azure_attempted = False

        def _ensure_criterion(text: str | None) -> str | None:
            if not text:
                return text
            if criterion and criterion.lower() not in text.lower():
                return f"{criterion}: {text}"
            return text

        if self.local_enabled and label_first:
            label_messages = self._build_label_messages(local_payload)
            label_text = self._call_local(label_messages, strict=False, force_json=False)
            if label_text is None:
                local_failed = True
            if label_text:
                extracted = _extract_labeled_sections(label_text) or _extract_numbered_sections(label_text)
                if extracted:
                    candidate_details = self._de_template(self._sanitize_llm_text(
                        self._strip_literal_prefixes(extracted.get("details") or "")
                    )) or detail_text
                    candidate_actions = self._de_template(self._sanitize_llm_text(
                        self._strip_literal_prefixes(extracted.get("actions") or "")
                    )) or action_text
                    candidate_rationale = self._de_template(self._sanitize_llm_text(
                        self._strip_literal_prefixes(extracted.get("rationale") or "")
                    )) or rationale_text
                    if where_to_look and not self._mentions_where_to_look([candidate_actions], where_to_look):
                        mention = self._sanitize_llm_text(self._strip_literal_prefixes(where_to_look))
                        if mention:
                            base = (candidate_actions or "").rstrip().rstrip(".")
                            candidate_actions = f"{base}. Focus on {mention}." if base else f"Focus on {mention}."
                    if data_needed and not self._mentions_data_needed([candidate_actions], data_needed):
                        data_note = self._sanitize_llm_text(self._strip_literal_prefixes(data_needed))
                        if data_note:
                            base = (candidate_actions or "").rstrip().rstrip(".")
                            candidate_actions = f"{base}. Needs {data_note}." if base else data_note
                    candidate_details = _ensure_criterion(candidate_details)
                    candidate_actions = _ensure_criterion(candidate_actions)
                    candidate_rationale = _ensure_criterion(candidate_rationale)
                    numeric_ok = True
                    if not data_needed:
                        numeric_ok = _contains_numeric_fact(detail_text or "", candidate_details or "")
                    reasons = []
                    if self._has_repeat([candidate_details, candidate_actions, candidate_rationale]):
                        reasons.append("repeat")
                    if self._has_templated_for_mode(
                        [candidate_details, candidate_actions, candidate_rationale]
                    ):
                        reasons.append("templated")
                    if not self._mentions_criterion(
                        [candidate_details, candidate_actions, candidate_rationale], criterion
                    ):
                        reasons.append("missing_criterion")
                    if not numeric_ok:
                        reasons.append("missing_numeric")
                    if _contradicts_signal(detail_text or "", candidate_details or ""):
                        reasons.append("contradict_signal")
                    if self.enforce_distinct and self._has_internal_overlap(
                        [candidate_details, candidate_actions, candidate_rationale]
                    ):
                        reasons.append("overlap")
                    invalid = len(reasons) > 0
                    if not invalid:
                        self.stats["rows_rewritten"] += 1
                        return self._finalize_triplet(
                            category=category,
                            criterion=criterion,
                            detail_text=candidate_details or "",
                            action_text=candidate_actions or "",
                            rationale_text=candidate_rationale or "",
                            score=score,
                            data_needed=data_needed,
                            where_to_look=where_to_look,
                        )
                    self._append_reject_log(
                        "label_first",
                        reasons,
                        {
                            "criterion": criterion,
                            "category": category,
                            "mode": self.mode,
                            "data_needed": data_needed,
                            "where_to_look": where_to_look,
                        },
                        {
                            "details": candidate_details,
                            "actions": candidate_actions,
                            "rationale": candidate_rationale,
                        },
                    )
                    if invalid and not azure_attempted:
                        azure_attempted = True
                        azure_label_messages = self._build_label_messages(payload)
                        azure_label_text = self._call_azure(azure_label_messages)
                        if azure_label_text:
                            extracted = _extract_labeled_sections(azure_label_text) or _extract_numbered_sections(azure_label_text)
                            if extracted:
                                az_details = self._de_template(self._sanitize_llm_text(
                                    self._strip_literal_prefixes(extracted.get("details") or "")
                                )) or detail_text
                                az_actions = self._de_template(self._sanitize_llm_text(
                                    self._strip_literal_prefixes(extracted.get("actions") or "")
                                )) or action_text
                                az_rationale = self._de_template(self._sanitize_llm_text(
                                    self._strip_literal_prefixes(extracted.get("rationale") or "")
                                )) or rationale_text
                                az_details = _ensure_criterion(az_details)
                                az_actions = _ensure_criterion(az_actions)
                                az_rationale = _ensure_criterion(az_rationale)
                                az_numeric_ok = True
                                if not data_needed:
                                    az_numeric_ok = _contains_numeric_fact(detail_text or "", az_details or "")
                                az_reasons = []
                                if self._has_repeat([az_details, az_actions, az_rationale]):
                                    az_reasons.append("repeat")
                                if self._has_templated_for_mode([az_details, az_actions, az_rationale]):
                                    az_reasons.append("templated")
                                if not self._mentions_criterion([az_details, az_actions, az_rationale], criterion):
                                    az_reasons.append("missing_criterion")
                                if not az_numeric_ok:
                                    az_reasons.append("missing_numeric")
                                if _contradicts_signal(detail_text or "", az_details or ""):
                                    az_reasons.append("contradict_signal")
                                if self.enforce_distinct and self._has_internal_overlap(
                                    [az_details, az_actions, az_rationale]
                                ):
                                    az_reasons.append("overlap")
                                if not az_reasons:
                                    self.stats["rows_rewritten"] += 1
                                    return self._finalize_triplet(
                                        category=category,
                                        criterion=criterion,
                                        detail_text=az_details or "",
                                        action_text=az_actions or "",
                                        rationale_text=az_rationale or "",
                                        score=score,
                                        data_needed=data_needed,
                                        where_to_look=where_to_look,
                                    )
        for attempt in range(self.max_retries + 1):
            strict = attempt > 0 or force_json
            response_text = None
            if self.local_enabled and use_local_json and not (self.local_fail_fast and local_failed):
                messages = self._build_messages(local_payload, strict=strict, compact=compact_local)
                response_text = self._call_local(messages, strict=strict)
                if response_text is None:
                    local_failed = True
                if response_text:
                    last_response_text = response_text
                data = _extract_json(response_text)
                if not data and response_text:
                    logging.info(
                        "[audit_verbalizer] local response not JSON (strict=%s, len=%d)",
                        strict,
                        len(response_text),
                    )
            if not data:
                azure_messages = self._build_messages(payload, strict=attempt > 0)
                response_text = self._call_azure(azure_messages)
                if response_text:
                    last_response_text = response_text
                data = _extract_json(response_text)
            if data:
                candidate_details = self._de_template(self._sanitize_llm_text(
                    self._strip_literal_prefixes(str(data.get("details", "")).strip())
                )) or detail_text
                candidate_actions = self._de_template(self._sanitize_llm_text(
                    self._strip_literal_prefixes(str(data.get("actions", "")).strip())
                )) or action_text
                candidate_rationale = self._de_template(self._sanitize_llm_text(
                    self._strip_literal_prefixes(str(data.get("rationale", "")).strip())
                )) or rationale_text
                if _contradicts_signal(detail_text or "", candidate_details or ""):
                    candidate_details = (
                        self._sanitize_llm_text(self._strip_literal_prefixes(detail_text or ""))
                        or detail_text
                    )
                if not candidate_actions:
                    candidate_actions = (
                        self._sanitize_llm_text(self._strip_literal_prefixes(action_text or ""))
                        or action_text
                    )
                if where_to_look and not self._mentions_where_to_look([candidate_actions], where_to_look):
                    mention = self._sanitize_llm_text(self._strip_literal_prefixes(where_to_look))
                    if mention:
                        base = candidate_actions.rstrip().rstrip(".")
                        candidate_actions = f"{base}. Focus on {mention}."
                if data_needed and not self._mentions_data_needed([candidate_actions], data_needed):
                    data_note = self._sanitize_llm_text(self._strip_literal_prefixes(data_needed))
                    if data_note:
                        base = candidate_actions.rstrip().rstrip(".")
                        candidate_actions = f"{base}. Needs {data_note}."
                if not data_needed:
                    if self._is_low_content(candidate_actions):
                        candidate_actions = (
                            self._sanitize_llm_text(self._strip_literal_prefixes(action_text or ""))
                            or action_text
                        )
                    if self._is_low_content(candidate_rationale):
                        candidate_rationale = (
                            self._sanitize_llm_text(self._strip_literal_prefixes(rationale_text or ""))
                            or rationale_text
                        )
                if not _contains_numeric_fact(detail_text or "", candidate_details or ""):
                    candidate_details = (
                        self._sanitize_llm_text(self._strip_literal_prefixes(detail_text or ""))
                        or detail_text
                    )
                combined = " ".join([candidate_details, candidate_actions, candidate_rationale]).lower()
                reasons = []
                if _looks_like_json_blob(candidate_details) or _looks_like_json_blob(candidate_actions) or _looks_like_json_blob(candidate_rationale):
                    reasons.append("json_blob")
                if self._has_repeat([candidate_details, candidate_actions, candidate_rationale]):
                    reasons.append("repeat")
                if self._has_templated_for_mode(
                    [candidate_details, candidate_actions, candidate_rationale]
                ):
                    reasons.append("templated")
                if not self._mentions_criterion(
                    [candidate_details, candidate_actions, candidate_rationale], criterion
                ):
                    reasons.append("missing_criterion")
                if not _contains_numeric_fact(detail_text or "", candidate_details or ""):
                    reasons.append("missing_numeric")
                if data_needed and not self._mentions_data_needed(
                    [candidate_details, candidate_actions, candidate_rationale], data_needed
                ):
                    reasons.append("missing_data_needed")
                if self.enforce_distinct and self._has_internal_overlap(
                    [candidate_details, candidate_actions, candidate_rationale]
                ):
                    reasons.append("overlap")
                invalid = len(reasons) > 0
                if invalid and attempt < self.max_retries:
                    self._append_reject_log(
                        "json_retry",
                        reasons,
                        {
                            "criterion": criterion,
                            "category": category,
                            "mode": self.mode,
                            "data_needed": data_needed,
                            "where_to_look": where_to_look,
                        },
                        {
                            "details": candidate_details,
                            "actions": candidate_actions,
                            "rationale": candidate_rationale,
                        },
                    )
                    data = None
                    continue
                if invalid and not azure_attempted:
                    azure_attempted = True
                    azure_messages = self._build_messages(payload, strict=True)
                    azure_text = self._call_azure(azure_messages)
                    if azure_text:
                        last_response_text = azure_text
                        data = _extract_json(azure_text)
                        if data:
                            candidate_details = self._de_template(self._sanitize_llm_text(
                                self._strip_literal_prefixes(str(data.get("details", "")).strip())
                            )) or detail_text
                            candidate_actions = self._de_template(self._sanitize_llm_text(
                                self._strip_literal_prefixes(str(data.get("actions", "")).strip())
                            )) or action_text
                            candidate_rationale = self._de_template(self._sanitize_llm_text(
                                self._strip_literal_prefixes(str(data.get("rationale", "")).strip())
                            )) or rationale_text
                            candidate_details = _ensure_criterion(candidate_details)
                            candidate_actions = _ensure_criterion(candidate_actions)
                            candidate_rationale = _ensure_criterion(candidate_rationale)
                            retry_reasons = []
                            if self._has_repeat([candidate_details, candidate_actions, candidate_rationale]):
                                retry_reasons.append("repeat")
                            if self._has_templated_for_mode([candidate_details, candidate_actions, candidate_rationale]):
                                retry_reasons.append("templated")
                            if not self._mentions_criterion([candidate_details, candidate_actions, candidate_rationale], criterion):
                                retry_reasons.append("missing_criterion")
                            if not _contains_numeric_fact(detail_text or "", candidate_details or ""):
                                retry_reasons.append("missing_numeric")
                            if data_needed and not self._mentions_data_needed(
                                [candidate_details, candidate_actions, candidate_rationale], data_needed
                            ):
                                retry_reasons.append("missing_data_needed")
                            if self.enforce_distinct and self._has_internal_overlap(
                                [candidate_details, candidate_actions, candidate_rationale]
                            ):
                                retry_reasons.append("overlap")
                            if not retry_reasons:
                                self.stats["rows_rewritten"] += 1
                                return self._finalize_triplet(
                                    category=category,
                                    criterion=criterion,
                                    detail_text=candidate_details or "",
                                    action_text=candidate_actions or "",
                                    rationale_text=candidate_rationale or "",
                                    score=score,
                                    data_needed=data_needed,
                                    where_to_look=where_to_look,
                                )
                if invalid:
                    self._append_reject_log(
                        "json_final",
                        reasons,
                        {
                            "criterion": criterion,
                            "category": category,
                            "mode": self.mode,
                            "data_needed": data_needed,
                            "where_to_look": where_to_look,
                        },
                        {
                            "details": candidate_details,
                            "actions": candidate_actions,
                            "rationale": candidate_rationale,
                        },
                    )
                    data = None
                break

        if not data:
            extracted = None
            if last_response_text:
                extracted = _extract_labeled_sections(last_response_text)
                if not extracted:
                    extracted = _extract_numbered_sections(last_response_text)
                if extracted:
                    new_details = self._de_template(self._sanitize_llm_text(self._strip_literal_prefixes(extracted.get("details") or "")))
                    new_actions = self._de_template(self._sanitize_llm_text(self._strip_literal_prefixes(extracted.get("actions") or "")))
                    new_rationale = self._de_template(self._sanitize_llm_text(self._strip_literal_prefixes(extracted.get("rationale") or "")))
                    if any(
                        _looks_like_json_blob(text)
                        for text in (new_details, new_actions, new_rationale, last_response_text)
                    ):
                        extracted = None
            if not extracted and not (self.local_fail_fast and local_failed):
                label_messages = self._build_label_messages(local_payload)
                label_text = self._call_local(label_messages, strict=False, force_json=False)
                if label_text is None:
                    local_failed = True
                if label_text:
                    extracted = _extract_labeled_sections(label_text)
                    if not extracted:
                        extracted = _extract_numbered_sections(label_text)
                    if extracted:
                        new_details = self._sanitize_llm_text(
                            self._strip_literal_prefixes(extracted.get("details") or "")
                        )
                        new_actions = self._sanitize_llm_text(
                            self._strip_literal_prefixes(extracted.get("actions") or "")
                        )
                        new_rationale = self._sanitize_llm_text(
                            self._strip_literal_prefixes(extracted.get("rationale") or "")
                        )
                        if any(
                            _looks_like_json_blob(text)
                            for text in (new_details, new_actions, new_rationale, label_text)
                        ):
                            extracted = None
                if extracted:
                    if not new_details:
                        new_details = (
                            self._sanitize_llm_text(self._strip_literal_prefixes(detail_text or ""))
                            or detail_text
                        )
                    if not new_actions:
                        new_actions = (
                            self._sanitize_llm_text(self._strip_literal_prefixes(action_text or ""))
                            or action_text
                        )
                    if not new_rationale:
                        new_rationale = (
                            self._sanitize_llm_text(self._strip_literal_prefixes(rationale_text or ""))
                            or rationale_text
                        )
                    if data_needed and not self._mentions_data_needed([new_actions], data_needed):
                        data_note = self._sanitize_llm_text(self._strip_literal_prefixes(data_needed))
                        if data_note:
                            base = (new_actions or "").rstrip().rstrip(".")
                            if base:
                                new_actions = f"{base}. Needs {data_note}."
                            else:
                                new_actions = data_note
                    if not _contains_numeric_fact(detail_text or "", new_details or ""):
                        new_details = (
                            self._sanitize_llm_text(self._strip_literal_prefixes(detail_text or ""))
                            or detail_text
                        )
                    if _contradicts_signal(detail_text or "", new_details or ""):
                        new_details = (
                            self._sanitize_llm_text(self._strip_literal_prefixes(detail_text or ""))
                            or detail_text
                        )
                    if where_to_look and not self._mentions_where_to_look([new_actions], where_to_look):
                        mention = self._sanitize_llm_text(self._strip_literal_prefixes(where_to_look))
                        if mention:
                            base = (new_actions or "").rstrip().rstrip(".")
                            if base:
                                new_actions = f"{base}. Focus on {mention}."
                            else:
                                new_actions = f"Focus on {mention}."
                    if not data_needed:
                        if self._is_low_content(new_actions):
                            new_actions = (
                                self._sanitize_llm_text(self._strip_literal_prefixes(action_text or ""))
                                or action_text
                            )
                        if self._is_low_content(new_rationale):
                            new_rationale = (
                                self._sanitize_llm_text(self._strip_literal_prefixes(rationale_text or ""))
                                or rationale_text
                            )
                    if self._has_templated_for_mode([new_details, new_actions, new_rationale]):
                        extracted = None
                    else:
                        self.stats["rows_rewritten"] += 1
                        return self._finalize_triplet(
                            category=category,
                            criterion=criterion,
                            detail_text=new_details or "",
                            action_text=new_actions or "",
                            rationale_text=new_rationale or "",
                            score=score,
                            data_needed=data_needed,
                            where_to_look=where_to_look,
                        )
            # Last-chance deterministic cleanup to avoid templated repetition when LLM output is unusable.
            base_label = (criterion or "").strip() or (category or "").strip() or "This check"
            fallback_details = (
                self._sanitize_llm_text(self._strip_literal_prefixes(detail_text or ""))
                or detail_text
            )
            fallback_actions = (
                self._sanitize_llm_text(self._strip_literal_prefixes(action_text or ""))
                or action_text
            )
            fallback_rationale = (
                self._sanitize_llm_text(self._strip_literal_prefixes(rationale_text or ""))
                or rationale_text
            )
            if data_needed and not self._mentions_data_needed([fallback_actions], data_needed):
                data_note = self._sanitize_llm_text(self._strip_literal_prefixes(data_needed))
                if data_note:
                    fallback_actions = f"{base_label}: {data_note} required to validate this check."
            if where_to_look and not self._mentions_where_to_look([fallback_actions], where_to_look):
                mention = self._sanitize_llm_text(self._strip_literal_prefixes(where_to_look))
                if mention:
                    fallback_actions = f"{base_label}: review {mention} and confirm the setting."
            fallback_details = _ensure_criterion(fallback_details)
            fallback_actions = _ensure_criterion(fallback_actions)
            fallback_rationale = _ensure_criterion(fallback_rationale)
            if self._is_low_content(fallback_details, min_words=4):
                fallback_details = f"{base_label} needs review based on the current export."
            if self._is_low_content(fallback_actions, min_words=4):
                fallback_actions = f"{base_label}: confirm the setting in the source system."
            if self._is_low_content(fallback_rationale, min_words=4):
                fallback_rationale = f"{base_label}: current exports do not include enough detail to verify this item."
            if not self._has_templated_for_mode([fallback_details, fallback_actions, fallback_rationale]):
                self.stats["rows_rewritten"] += 1
                return self._finalize_triplet(
                    category=category,
                    criterion=criterion,
                    detail_text=fallback_details or "",
                    action_text=fallback_actions or "",
                    rationale_text=fallback_rationale or "",
                    score=score,
                    data_needed=data_needed,
                    where_to_look=where_to_look,
                )
            self.stats["fallback_used"] += 1
            self._append_reject_log(
                "fallback",
                ["no_valid_llm_output"],
                {
                    "criterion": criterion,
                    "category": category,
                    "mode": self.mode,
                    "data_needed": data_needed,
                    "where_to_look": where_to_look,
                },
                {
                    "details": detail_text,
                    "actions": action_text,
                    "rationale": rationale_text,
                },
            )
            fallback_details = self._de_template(
                self._sanitize_llm_text(self._strip_literal_prefixes(detail_text or ""))
            ) or self._sanitize_llm_text(self._strip_literal_prefixes(detail_text or ""))
            fallback_actions = self._de_template(
                self._sanitize_llm_text(self._strip_literal_prefixes(action_text or ""))
            ) or self._sanitize_llm_text(self._strip_literal_prefixes(action_text or ""))
            fallback_rationale = self._de_template(
                self._sanitize_llm_text(self._strip_literal_prefixes(rationale_text or ""))
            ) or self._sanitize_llm_text(self._strip_literal_prefixes(rationale_text or ""))
            if data_needed or where_to_look:
                missing = data_needed or where_to_look or ""
                missing = self._sanitize_llm_text(self._strip_literal_prefixes(missing))
                base_label = (criterion or "").strip()
                if category:
                    category = category.strip()
                    if base_label and category.lower() not in base_label.lower():
                        base_label = f"{category} - {base_label}"
                    elif not base_label:
                        base_label = category
                if base_label:
                    base_label = base_label.rstrip(" ?!.")
                if not base_label:
                    base_label = "This check"
                if missing:
                    key = f"{base_label}|{missing}"
                    key_hash = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16)
                    detail_templates = [
                        "{label} cannot be verified without {missing}.",
                        "Verification for {label} depends on {missing}.",
                        "{label} needs {missing} before it can be confirmed.",
                    ]
                    action_templates = [
                        "Provide {missing} to validate {label}.",
                        "Share {missing} so {label} can be verified.",
                        "Supply {missing}; without it, {label} cannot be confirmed.",
                    ]
                    rationale_templates = [
                        "Missing {missing} limits confidence in {label}.",
                        "Without {missing}, evidence for {label} is incomplete.",
                        "{label} remains unverified until {missing} is provided.",
                    ]
                    fallback_details = detail_templates[key_hash % len(detail_templates)].format(
                        label=base_label, missing=missing
                    )
                    fallback_actions = action_templates[key_hash % len(action_templates)].format(
                        label=base_label, missing=missing
                    )
                    fallback_rationale = rationale_templates[key_hash % len(rationale_templates)].format(
                        label=base_label, missing=missing
                    )
            fallback_details = _ensure_criterion(fallback_details)
            fallback_actions = _ensure_criterion(fallback_actions)
            fallback_rationale = _ensure_criterion(fallback_rationale)
            return self._finalize_triplet(
                category=category,
                criterion=criterion,
                detail_text=fallback_details or detail_text or "",
                action_text=fallback_actions or action_text or "",
                rationale_text=fallback_rationale or rationale_text or "",
                score=score,
                data_needed=data_needed,
                where_to_look=where_to_look,
            )

        new_details = self._de_template(self._sanitize_llm_text(
            self._strip_literal_prefixes(str(data.get("details", "")).strip())
        )) or detail_text
        new_actions = self._de_template(self._sanitize_llm_text(
            self._strip_literal_prefixes(str(data.get("actions", "")).strip())
        )) or action_text
        new_rationale = self._de_template(self._sanitize_llm_text(
            self._strip_literal_prefixes(str(data.get("rationale", "")).strip())
        )) or rationale_text
        if not data_needed:
            if self._is_low_content(new_actions):
                new_actions = (
                    self._sanitize_llm_text(self._strip_literal_prefixes(action_text or ""))
                    or action_text
                )
            if self._is_low_content(new_rationale):
                new_rationale = (
                    self._sanitize_llm_text(self._strip_literal_prefixes(rationale_text or ""))
                    or rationale_text
                )
        if not _contains_numeric_fact(detail_text or "", new_details or ""):
            new_details = (
                self._sanitize_llm_text(self._strip_literal_prefixes(detail_text or ""))
                or detail_text
            )

        if _NON_ASCII_RE.search(new_details + new_actions + new_rationale):
            pass

        self.stats["rows_rewritten"] += 1
        return self._finalize_triplet(
            category=category,
            criterion=criterion,
            detail_text=new_details or "",
            action_text=new_actions or "",
            rationale_text=new_rationale or "",
            score=score,
            data_needed=data_needed,
            where_to_look=where_to_look,
        )
