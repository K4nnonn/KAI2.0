import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import pandas as pd


@dataclass
class MarketVolatilitySummary:
    """Lightweight signal used by the agent to reason about market shifts."""

    sensor_score: float
    high_volatility: bool
    keyword_count: int
    top_movers: List[Dict[str, Any]]


class MarketIntelligence:
    """
    Context layer for external market data (e.g., Semrush-style exports).
    Implements the "volatility logic" from the treatise: measure how turbulent the SERP is
    so the agent can attribute performance changes to market demand vs. execution issues.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def normalize_market_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required columns exist; tolerate multiple naming conventions."""
        rename_map = {
            "rank_current": "rank",
            "current_rank": "rank",
            "rank_previous": "prev_rank",
            "previous_rank": "prev_rank",
        }
        normalized = df.rename(columns=rename_map).copy()
        required = ["keyword", "domain", "rank"]
        for col in required:
            if col not in normalized.columns:
                normalized[col] = pd.NA
        if "prev_rank" not in normalized.columns:
            normalized["prev_rank"] = pd.NA
        if "scraped_at" not in normalized.columns:
            normalized["scraped_at"] = pd.NaT
        return normalized

    def compute_volatility(self, market_df: pd.DataFrame, rolling_window: int = 7) -> MarketVolatilitySummary:
        """
        Volatility Logic (Sensor Score):
        - Rank turbulence: rolling std of rank movements.
        - Shock factor: latest rank delta vs. previous.
        Sensor score is scaled 0-1 and capped.
        """
        df = self.normalize_market_frame(market_df)
        if df.empty:
            return MarketVolatilitySummary(sensor_score=0.0, high_volatility=False, keyword_count=0, top_movers=[])

        df = df.copy()
        # Use per-keyword volatility; fall back gracefully when prev_rank is missing.
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df["prev_rank"] = pd.to_numeric(df["prev_rank"], errors="coerce")
        df["delta"] = (df["prev_rank"] - df["rank"]).fillna(0)

        keyword_vol = (
            df.groupby("keyword")
            .agg(
                mean_rank=("rank", "mean"),
                rank_std=("rank", "std"),
                last_delta=("delta", "last"),
            )
            .reset_index()
        )
        keyword_vol["rank_std"] = keyword_vol["rank_std"].fillna(0)
        keyword_vol["last_delta"] = keyword_vol["last_delta"].fillna(0)

        # Sensor score: combine normalized volatility and the latest shock.
        keyword_vol["sensor"] = (
            (keyword_vol["rank_std"] / (keyword_vol["mean_rank"].abs() + 1e-6)).clip(0, 1)
            * 0.6
            + (keyword_vol["last_delta"].abs() / 10.0).clip(0, 1) * 0.4
        )
        sensor_score = float(keyword_vol["sensor"].mean())
        high_volatility = sensor_score >= 0.4

        top_movers = (
            keyword_vol.sort_values(by="sensor", ascending=False)
            .head(5)
            .to_dict(orient="records")
        )

        summary = MarketVolatilitySummary(
            sensor_score=sensor_score,
            high_volatility=high_volatility,
            keyword_count=keyword_vol.shape[0],
            top_movers=top_movers,
        )
        self.logger.info("Market volatility sensor=%.2f (keywords=%s)", sensor_score, keyword_vol.shape[0])
        return summary

