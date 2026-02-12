"""
Adapter layer for creative-related domain objects and validation utilities.

This module is intentionally standalone to avoid touching existing engine/connector code.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CreativeContext:
    """
    Normalized creative payload regardless of source (CSV now, API later).
    """

    final_url: str
    keywords: List[str]
    business_name: str
    usp_list: List[str]
    source_id: Optional[str] = None

    def add_keyword(self, keyword: str) -> None:
        if keyword and keyword not in self.keywords:
            self.keywords.append(keyword)

    def add_usp(self, usp: str) -> None:
        if usp and usp not in self.usp_list:
            self.usp_list.append(usp)


def calculate_google_ads_width(text: str) -> int:
    """
    Google Ads display width counting: wide/full-width chars count as 2.
    """
    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


class AdCopyValidator:
    HEADLINE_LIMIT = 30
    DESC_LIMIT = 90

    @staticmethod
    def sanitize_headline(text: str) -> str:
        cleaned = text.replace("!", "")
        cleaned = re.sub(r"([?.])\1+", r"\1", cleaned)
        cleaned = re.sub(r"[•→★]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def sanitize_description(text: str) -> str:
        cleaned = re.sub(r"[•→★]", "", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def is_valid_headline(cls, text: str) -> bool:
        sanitized = cls.sanitize_headline(text)
        length_ok = calculate_google_ads_width(sanitized) <= cls.HEADLINE_LIMIT
        caps_ok = not (sanitized.isupper() and len(sanitized) > 4)
        return length_ok and caps_ok

    @classmethod
    def is_valid_description(cls, text: str) -> bool:
        sanitized = cls.sanitize_description(text)
        length_ok = calculate_google_ads_width(sanitized) <= cls.DESC_LIMIT
        return length_ok

    @classmethod
    def validate(cls, headlines: List[str], descriptions: List[str]) -> dict:
        """
        Lightweight batch validation summary.
        """
        results = {
            "headlines": [{"text": h, "valid": cls.is_valid_headline(h)} for h in headlines],
            "descriptions": [{"text": d, "valid": cls.is_valid_description(d)} for d in descriptions],
        }
        return results
