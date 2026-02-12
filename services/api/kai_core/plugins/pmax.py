"""
PMax Insights Engine (satellite) for spend splits, efficiency, and mobile app waste detection.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

MOBILE_APP_PATTERN = r"^mobileapp::[1-2]-"


class PMaxAnalyzer:
    def __init__(self, cost_col: str = "Cost", conversions_col: str = "Conversions", value_col: str = "Conversion Value"):
        self.cost_col = cost_col
        self.conversions_col = conversions_col
        self.value_col = value_col

    @staticmethod
    def spend_split(total_cost: float, shopping_cost: float = 0.0, video_cost: float = 0.0) -> Dict[str, float]:
        other = max(total_cost - (shopping_cost + video_cost), 0.0)
        return {"total_cost": total_cost, "shopping_cost": shopping_cost, "video_cost": video_cost, "other_cost": other}

    @staticmethod
    def efficiency_score(cost: float, conversions: float, value: float) -> float:
        if cost <= 0:
            return 0.0
        roas = value / cost
        return (roas * 0.7) + (conversions * 0.3)

    def identify_mobile_app_waste(self, df: pd.DataFrame, cost_threshold: float = 50.0) -> pd.DataFrame:
        if df.empty:
            return df
        if "Placement" not in df.columns or self.cost_col not in df.columns or self.conversions_col not in df.columns:
            return df.iloc[0:0]
        mobile_apps = df[df["Placement"].astype(str).str.contains(MOBILE_APP_PATTERN, regex=True)]
        waste = mobile_apps[(mobile_apps[self.cost_col] > cost_threshold) & (mobile_apps[self.conversions_col] == 0)]
        return waste

    def analyze(self, asset_groups: Optional[Iterable[Dict[str, Any]]] = None, placements: Optional[Iterable[Dict[str, Any]]] = None, total_cost: float = 0.0, shopping_cost: float = 0.0, video_cost: float = 0.0) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        summary["spend_split"] = self.spend_split(total_cost, shopping_cost, video_cost)

        df_assets = pd.DataFrame(asset_groups or [])
        if not df_assets.empty and self.cost_col in df_assets.columns and self.value_col in df_assets.columns and self.conversions_col in df_assets.columns:
            df_assets = df_assets.copy()
            df_assets["efficiency_score"] = df_assets.apply(
                lambda row: self.efficiency_score(float(row.get(self.cost_col, 0) or 0), float(row.get(self.conversions_col, 0) or 0), float(row.get(self.value_col, 0) or 0)),
                axis=1,
            )
            summary["asset_group_efficiency"] = df_assets.to_dict(orient="records")
        else:
            summary["asset_group_efficiency"] = []

        df_places = pd.DataFrame(placements or [])
        summary["mobile_app_waste"] = self.identify_mobile_app_waste(df_places).to_dict(orient="records") if not df_places.empty else []

        return summary
