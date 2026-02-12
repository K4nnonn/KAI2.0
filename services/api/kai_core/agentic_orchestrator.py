import logging
from typing import Dict, List, Optional, Any

import pandas as pd

from kai_core.market_intelligence import MarketIntelligence, MarketVolatilitySummary
from kai_core.pmax_channel_split import PMaxChannelSplitter, PMaxChannelBreakout
from kai_core.creative_cv_scoring import CreativeCVScorer


class MarketingReasoningAgent:
    """
    Master Agent that stitches Context (Market), Execution (PMax), and Creative signals.
    Implements a simple ReAct-style chain: reason over each pillar, then synthesize.
    """

    def __init__(
        self,
        market_intel: Optional[MarketIntelligence] = None,
        pmax_splitter_factory: Optional[type[PMaxChannelSplitter]] = None,
        creative_scorer: Optional[CreativeCVScorer] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.market_intel = market_intel or MarketIntelligence(self.logger)
        self.pmax_splitter_factory = pmax_splitter_factory or PMaxChannelSplitter
        self.creative_scorer = creative_scorer or CreativeCVScorer()

    def analyze(
        self,
        query: str,
        pmax_df: Optional[pd.DataFrame] = None,
        creative_df: Optional[pd.DataFrame] = None,
        market_df: Optional[pd.DataFrame] = None,
        brand_terms: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Returns a root-cause style narrative plus the evidence from each pillar."""
        checks: Dict[str, Any] = {}

        # Check 1: Market volatility
        market_signal: Optional[MarketVolatilitySummary] = None
        if market_df is not None and not market_df.empty:
            market_signal = self.market_intel.compute_volatility(market_df)
            checks["market"] = {
                "sensor_score": market_signal.sensor_score,
                "high_volatility": market_signal.high_volatility,
                "top_movers": market_signal.top_movers,
            }

        # Check 2: Execution / PMax cannibalization
        pmax_signal: Optional[PMaxChannelBreakout] = None
        brand_pressure: Optional[float] = None
        if pmax_df is not None and not pmax_df.empty:
            try:
                splitter = self.pmax_splitter_factory(pmax_df)
                pmax_signal = splitter.infer()
            except Exception as exc:  # pragma: no cover
                self.logger.warning("PMax splitter failed: %s", exc)
            brand_pressure = self._detect_brand_pressure(pmax_df, brand_terms or [])
            if pmax_signal:
                checks["execution"] = {
                    "channel_breakout": pmax_signal.__dict__,
                    "brand_pressure": brand_pressure,
                }

        # Check 3: Creative quality
        creative_signal = self._summarize_creative(creative_df)
        if creative_signal is not None:
            checks["creative"] = creative_signal

        root_cause = self._synthesize(checks, market_signal, pmax_signal, brand_pressure, creative_signal)

        return {
            "query": query,
            "root_cause": root_cause,
            "checks": checks,
        }

    def _detect_brand_pressure(self, pmax_df: pd.DataFrame, brand_terms: List[str]) -> Optional[float]:
        """Heuristic: estimate % of cost on brand search terms to flag cannibalization risk."""
        if not brand_terms:
            return None
        cost_col = next((c for c in pmax_df.columns if c.lower() in ("cost", "cost_micros")), None)
        term_col = next((c for c in pmax_df.columns if "search_term" in c.lower()), None)
        if not cost_col or not term_col:
            return None
        df = pmax_df[[cost_col, term_col]].copy()
        df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce").fillna(0)
        df["is_brand"] = df[term_col].str.lower().apply(
            lambda t: any(bt.lower() in str(t) for bt in brand_terms)
        )
        total = df[cost_col].sum()
        brand_cost = df.loc[df["is_brand"], cost_col].sum()
        return float(brand_cost / total) if total > 0 else None

    def _summarize_creative(self, creative_df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        if creative_df is None or creative_df.empty:
            return None
        score_col = next((c for c in creative_df.columns if "quality" in c.lower() and "score" in c.lower()), None)
        if not score_col:
            return None
        series = pd.to_numeric(creative_df[score_col], errors="coerce").dropna()
        if series.empty:
            return None
        return {
            "avg_quality_score": float(series.mean()),
            "p25": float(series.quantile(0.25)),
            "p75": float(series.quantile(0.75)),
            "count": int(series.shape[0]),
        }

    def _synthesize(
        self,
        checks: Dict[str, Any],
        market_signal: Optional[MarketVolatilitySummary],
        pmax_signal: Optional[PMaxChannelBreakout],
        brand_pressure: Optional[float],
        creative_signal: Optional[Dict[str, Any]],
    ) -> str:
        """Generate a concise root-cause narrative with priority: Market > Execution > Creative."""
        if market_signal and market_signal.high_volatility:
            return "Market shift detected: high SERP volatility; performance may follow demand softness."

        if pmax_signal:
            search_share = (pmax_signal.search_cost / pmax_signal.total_cost) if pmax_signal.total_cost else 0
            if brand_pressure and brand_pressure > 0.35 and search_share > 0.5:
                return "PMax is leaning on branded search; rebalance toward Shopping/Video to avoid cannibalization."
            if search_share > 0.65:
                return "PMax spend is concentrated in Search; explore Shopping/Video coverage to regain reach."

        if creative_signal and creative_signal.get("avg_quality_score", 100) < 70:
            return "Creative depth/quality is dragging results; refresh assets and tighten messaging hierarchy."

        return "No dominant root-cause detected; signals are stable. Investigate funnel, tracking, or budgets."

