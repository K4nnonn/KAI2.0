"""
Infer channel splits for PMax using the subtraction method (Search vs Shopping vs Video vs Display).
This is additive and can be called from existing audit flows without refactoring core logic.
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class PMaxChannelBreakout:
    search_cost: float
    shopping_cost: float
    video_cost: float
    display_cost: float
    remainder_cost: float
    total_cost: float


class PMaxChannelSplitter:
    def __init__(self, df_campaign: pd.DataFrame, df_shopping: Optional[pd.DataFrame] = None):
        """
        df_campaign: campaign-level table with columns:
          - cost (numeric)
          - ad_network_type (optional): YOUTUBE, DISPLAY, SEARCH, etc.
        df_shopping: optional shopping_performance_view-like table with cost column
        """
        self.df_campaign = df_campaign
        self.df_shopping = df_shopping

    def infer(self) -> PMaxChannelBreakout:
        total_cost = float(self.df_campaign.get("cost", 0).sum() or 0)

        video_cost = float(self._sum_cost_by_network("YOUTUBE"))
        display_cost = float(self._sum_cost_by_network("DISPLAY"))
        search_cost = float(self._sum_cost_by_network("SEARCH"))

        shopping_cost = 0.0
        if self.df_shopping is not None and "cost" in self.df_shopping.columns:
            shopping_cost = float(self.df_shopping["cost"].sum())

        inferred_search = total_cost - (video_cost + display_cost + shopping_cost)
        remainder = inferred_search - search_cost if search_cost else inferred_search

        final_search = search_cost if search_cost > 0 else max(inferred_search, 0.0)
        remainder = max(remainder, 0.0)

        return PMaxChannelBreakout(
            search_cost=final_search,
            shopping_cost=max(shopping_cost, 0.0),
            video_cost=max(video_cost, 0.0),
            display_cost=max(display_cost, 0.0),
            remainder_cost=remainder,
            total_cost=total_cost,
        )

    def _sum_cost_by_network(self, network_name: str) -> float:
        if "ad_network_type" not in self.df_campaign.columns:
            return 0.0
        subset = self.df_campaign[self.df_campaign["ad_network_type"].str.upper() == network_name]
        if subset.empty or "cost" not in subset.columns:
            return 0.0
        return subset["cost"].sum()
