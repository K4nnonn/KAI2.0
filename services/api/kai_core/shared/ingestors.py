"""
Dual-mode ingestion scaffolding.

Phase 1: CSV ingestion with currency normalization.
Phase 2: API ingestors can implement the same interface without refactoring consumers.
"""
from __future__ import annotations

import csv
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)


class DataIngestionStrategy(ABC):
    """
    Target interface for data ingestion (CSV now, API later).
    """

    @abstractmethod
    def ingest(self, source: Any) -> List[Dict[str, Any]]:
        """
        Convert a data source into a list of normalized records (dicts).
        """
        raise NotImplementedError


def _normalize_currency(value: str | float | int | None) -> float:
    """
    Normalize currency strings like "$5.00" or "€1,234.56" to float.
    If already numeric, pass through as float.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    # Remove currency symbols and commas
    text = re.sub(r"[^\d.\-]", "", text)
    try:
        return float(text) if text else 0.0
    except ValueError:
        logger.warning("Currency normalization failed for value=%s", value)
        return 0.0


class CsvIngestor(DataIngestionStrategy):
    """
    CSV ingestor with basic header passthrough and currency normalization.
    """

    def ingest(self, source: str | Path | Iterable[str]) -> List[Dict[str, Any]]:
        path = Path(source)
        rows: List[Dict[str, Any]] = []
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                row: Dict[str, Any] = {}
                for key, val in raw.items():
                    if key is None:
                        continue
                    normalized_key = key.strip()
                    if "cost" in normalized_key.lower() or "value" in normalized_key.lower():
                        row[normalized_key] = _normalize_currency(val)
                    else:
                        row[normalized_key] = val
                rows.append(row)
        return rows
