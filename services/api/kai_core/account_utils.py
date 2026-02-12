"""
Shared helpers for working with account names across Azure Functions.
"""

from __future__ import annotations

import re
from typing import Optional


ACCOUNT_ALIASES = {
    "retirement": "Retirement",
    "brand": "Brand",
    "brokerage": "Brokerage",
    "wealth": "Wealth Management",
    "wealth management": "Wealth Management",
    "wealth_management": "Wealth Management",
    "wealthmanagement": "Wealth Management",
    "youth": "Youth",
    "hsa": "HSA",
    "crypto": "Crypto",
    "asset": "Asset Management",
    "asset management": "Asset Management",
    "asset_management": "Asset Management",
    "asset mgmt": "Asset Management",
    "fili term life": "FILI Term Life",
    "fili_term_life": "FILI Term Life",
    "fili": "FILI",
    "medicare": "Medicare",
    "wi": "WI",
    "fcm": "FCM",
    "family office": "Family Office",
    "family_office": "Family Office",
    "iicg": "IICG",
    "icg": "IICG",
    "iwms": "IWMS",
    "plynk": "Plynk",
    "sps": "SPS",
}


def normalize_account_name(value: Optional[str], *, allow_passthrough: bool = True) -> Optional[str]:
    """
    Normalize user-provided account names so that downstream filters match.
    """
    if value is None:
        return None

    cleaned = re.sub(r"[\s_\-]+", " ", value).strip().lower()
    if not cleaned:
        return None

    if cleaned in ACCOUNT_ALIASES:
        return ACCOUNT_ALIASES[cleaned]

    for token in cleaned.split():
        if token in ACCOUNT_ALIASES:
            return ACCOUNT_ALIASES[token]

    if allow_passthrough:
        return value.strip() or None
    return None


def slugify_account_name(value: str) -> str:
    """
    Produce a filesystem-friendly slug for locating data folders in storage.
    """
    normalized = re.sub(r"[\s\-]+", "_", value.strip())
    sanitized = re.sub(r"[^\w_]", "", normalized)
    return (sanitized or "General").lower()
