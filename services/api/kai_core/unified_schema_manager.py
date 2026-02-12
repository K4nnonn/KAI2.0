import logging
from typing import Iterable, List, Optional

import pandas as pd


class UnifiedSchemaManager:
    """
    Rosetta Stone that aligns Execution (PMax), Creative, and Market data onto a common key.
    This follows the "topic-centric schema" pattern from the treatise.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def _find_key(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
        lowered = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in lowered:
                return lowered[cand.lower()]
        return None

    def merge_intelligence(
        self,
        pmax_df: pd.DataFrame,
        creative_df: pd.DataFrame,
        market_df: pd.DataFrame,
        key_priority: Iterable[str] = ("product_sku", "landing_page_url", "url", "page", "final_url"),
    ) -> pd.DataFrame:
        """
        Joins three domains into a unified table.
        - Detects the best available key across frames using key_priority.
        - Renames the resolved key to 'entity_key' for consistency.
        """
        frames = {"pmax": pmax_df.copy() if pmax_df is not None else pd.DataFrame(),
                  "creative": creative_df.copy() if creative_df is not None else pd.DataFrame(),
                  "market": market_df.copy() if market_df is not None else pd.DataFrame()}

        key_map: List[tuple[str, str]] = []
        for name, frame in frames.items():
            if frame.empty:
                continue
            key_col = self._find_key(frame, key_priority)
            if key_col:
                frames[name] = frame.rename(columns={key_col: "entity_key"})
                key_map.append((name, key_col))

        if not key_map:
            self.logger.warning("No common key found across frames; returning empty merge.")
            return pd.DataFrame()

        # Start merge with the richest frame (choose pmax > creative > market).
        base = frames.get("pmax")
        if base is None or base.empty:
            base = next((f for f in frames.values() if not f.empty), pd.DataFrame())

        merged = base
        for name in ("creative", "market"):
            frame = frames.get(name)
            if frame is None or frame.empty:
                continue
            merged = pd.merge(merged, frame, on="entity_key", how="left", suffixes=("", f"_{name}"))

        self.logger.info("Unified table shape=%s (keys resolved: %s)", merged.shape, key_map)
        return merged

