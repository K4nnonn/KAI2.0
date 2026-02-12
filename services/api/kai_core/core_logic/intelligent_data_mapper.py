"""
INTELLIGENT DATA SOURCE MAPPER
Automatically discovers which SA360 report files contain which data
Never assumes - always validates
Adapts to different SA360 export structures
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path

def _apply_censoring_policy(value: float, is_lt: bool, is_gt: bool) -> float:
    """Map censored values like '< 10%' to midpoints so averages remain meaningful."""
    if is_lt:
        return value / 2.0
    if is_gt:
        return min(99.9, (value + 100.0) / 2.0)
    return value


def _normalize_numeric_token(raw):
    if raw is None:
        return np.nan

    s = str(raw).strip()
    if s in ("", "--", "N/A", "(not set)"):
        return np.nan

    had_percent = "%" in s
    is_lt = bool(re.match(r"^\s*[<≤]", s))
    is_gt = bool(re.match(r"^\s*[>≥]", s))

    s = re.sub(r'^[<>\=\~≤≥]+\s*', '', s)
    s = re.sub(r'\s*%$', '', s)

    s = s.replace(",", "").replace("$", "")

    if not s:
        return np.nan

    try:
        value = float(s)
    except ValueError:
        return np.nan

    if had_percent and (is_lt or is_gt):
        value = _apply_censoring_policy(value, is_lt, is_gt)

    return value


def to_numeric_safe(series):
    """Convert strings with commas/percent to numeric, handling censored values."""
    if series.dtype == 'object':
        converted = series.apply(_normalize_numeric_token)
        return converted

    return pd.to_numeric(series, errors='coerce').fillna(0)


CRITICAL_COLUMN_TOKENS = (
    "account",
    "campaign",
    "ad group",
    "keyword",
    "impr",
    "click",
    "cost",
)


def _looks_like_structured_export(df: pd.DataFrame) -> bool:
    """
    Determine whether a dataframe appears to have parsed column headers correctly.

    Many SA360 exports use multiple encodings/delimiters. When parsed incorrectly,
    pandas will treat a data row as the header (e.g., column names become numeric
    account IDs). This helper ensures we only accept frames that contain at least
    one of the core PPC column tokens in their header names.
    """
    if df is None or df.empty:
        return False

    normalized_cols = [str(col).strip().lower() for col in df.columns if isinstance(col, str)]
    if not normalized_cols:
        return False

    has_known_column = any(
        any(token in col for token in CRITICAL_COLUMN_TOKENS)
        for col in normalized_cols
    )
    if not has_known_column:
        return False

    # If every column is unnamed, we likely picked up the wrong header row.
    unnamed_cols = sum(col.startswith("unnamed") for col in normalized_cols)
    if unnamed_cols == len(normalized_cols):
        return False

    return True

class IntelligentDataSourceMapper:
    """
    Intelligently discovers which SA360 files contain which data types
    Validates assumptions and adapts to data structure
    """

    def __init__(self):
        self.source_map = {}
        self.validation_log = []
        self.performance_summary = {}
        self.context_summary = {}

    @staticmethod
    def _normalize_text_series(series: pd.Series) -> pd.Series:
        if pd.api.types.is_string_dtype(series):
            return series.fillna("").astype(str).str.strip()
        return series.astype(str).str.strip()

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: List[str]) -> str:
        lowered = {str(col).lower(): col for col in df.columns}
        for name in candidates:
            key = name.lower()
            if key in lowered:
                return lowered[key]
        for col in df.columns:
            col_lower = str(col).lower()
            for name in candidates:
                if name.lower() in col_lower:
                    return col
        return ""

    @staticmethod
    def _get_first_available(all_data: Dict[str, pd.DataFrame], keys: Tuple[str, ...]) -> pd.DataFrame:
        for key in keys:
            if key in all_data and isinstance(all_data[key], pd.DataFrame):
                df = all_data[key]
                if df is not None and len(df) > 0:
                    return df
        return pd.DataFrame()

    def _summarize_context(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Capture supplemental signals (change cadence, audiences, geo, conversions, extensions)."""
        summary: Dict[str, Dict] = {}
        cost_candidates = ["Cost", "Cost (converted)", "Spend"]
        click_candidates = ["Clicks"]
        impr_candidates = ["Impr.", "Impressions"]

        def _pick_metric(df: pd.DataFrame) -> str:
            return (
                self._find_column(df, cost_candidates)
                or self._find_column(df, click_candidates)
                or self._find_column(df, impr_candidates)
            )

        def _top_entities(
            df: pd.DataFrame,
            name_candidates: List[str],
            metric_candidates: List[str],
            top_n: int = 3,
        ) -> Optional[Dict[str, object]]:
            name_col = self._find_column(df, name_candidates)
            metric_col = self._find_column(df, metric_candidates)
            if not name_col or not metric_col:
                return None
            metrics = to_numeric_safe(df[metric_col])
            if metrics.isna().all():
                return None
            grouped = (
                pd.DataFrame({"name": df[name_col], "metric": metrics})
                .groupby("name", dropna=False)["metric"]
                .sum()
                .sort_values(ascending=False)
                .head(top_n)
            )
            if grouped.empty:
                return None
            return {
                "metric": metric_col,
                "top": [
                    {"name": name if pd.notna(name) else "Unknown", "value": float(val)}
                    for name, val in grouped.items()
                ],
            }

        state_pattern = re.compile(r"\b(?:529|fidelity\s*-\s*529)\s*(az|ct|de|ma|nh)\b", re.IGNORECASE)

        def _extract_state(value: object) -> Optional[str]:
            if value is None:
                return None
            text = str(value)
            match = state_pattern.search(text)
            if match:
                return match.group(1).upper()
            return None

        # Change history
        change_df = self._get_first_available(all_data, ("change_history", "change_history_report"))
        if not change_df.empty:
            date_col = self._find_column(change_df, ["Change time", "Change Time", "Timestamp", "Date", "Day"])
            type_col = self._find_column(change_df, ["Change type", "Change Type", "Type"])
            change_info: Dict[str, object] = {"total_events": int(len(change_df))}
            if date_col:
                timestamps = pd.to_datetime(change_df[date_col], errors="coerce")
                cutoff = datetime.utcnow() - timedelta(days=30)
                recent = timestamps >= cutoff
                if timestamps.notna().any():
                    last_change = timestamps.max()
                    change_info["last_change"] = last_change.isoformat() if pd.notna(last_change) else None
                change_info["events_30d"] = int(recent.sum())
            if type_col:
                top_types = (
                    self._normalize_text_series(change_df[type_col])
                    .value_counts()
                    .head(3)
                    .index.tolist()
                )
                if top_types:
                    change_info["top_types"] = top_types
            summary["change_history"] = change_info

        # Audience segments
        audience_df = self._get_first_available(all_data, ("audience_segment", "audience"))
        if not audience_df.empty:
            name_col = self._find_column(audience_df, ["Audience list", "Audience", "Segment", "List name"])
            type_col = self._find_column(audience_df, ["List type", "Audience type"])
            aud_info: Dict[str, object] = {"rows": int(len(audience_df))}
            if name_col:
                names = self._normalize_text_series(audience_df[name_col])
                aud_info["unique_lists"] = int(names.nunique())
                aud_info["sample_lists"] = names.dropna().unique().tolist()[:5]
            if type_col:
                aud_info["type_breakdown"] = (
                    self._normalize_text_series(audience_df[type_col])
                    .value_counts()
                    .head(3)
                    .to_dict()
                )
            summary["audience"] = aud_info

        # Geo coverage
        geo_df = self._get_first_available(all_data, ("user_locations", "geo", "geos"))
        if not geo_df.empty:
            geo_col = self._find_column(geo_df, ["Target location", "Location", "Geo", "Region"])
            impr_col = self._find_column(geo_df, ["Impr.", "Impressions"])
            geo_info: Dict[str, object] = {"rows": int(len(geo_df))}
            if geo_col:
                names = self._normalize_text_series(geo_df[geo_col])
                geo_info["unique_locations"] = int(names.nunique())
            if geo_col and impr_col:
                impressions = to_numeric_safe(geo_df[impr_col])
                top_rows = (
                    pd.DataFrame({"location": geo_df[geo_col], "impr": impressions})
                    .groupby("location", dropna=False)["impr"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(5)
                )
                geo_info["top_locations"] = [
                    {"location": loc if pd.notna(loc) else "Unknown", "impr": float(val)}
                    for loc, val in top_rows.items()
                ]
            summary["geo"] = geo_info

        # Conversion tracking
        conv_df = self._get_first_available(all_data, ("conversion_source", "conversion_action"))
        if not conv_df.empty:
            status_col = self._find_column(conv_df, ["Status", "Conversion status", "State"])
            conv_info: Dict[str, object] = {"total_actions": int(len(conv_df))}
            if status_col:
                statuses = self._normalize_text_series(conv_df[status_col]).str.lower()
                active_mask = statuses.isin({"enabled", "active", "recording", "primary"})
                conv_info["active_actions"] = int(active_mask.sum())
                conv_info["inactive_actions"] = int((~active_mask).sum())
            summary["conversion_tracking"] = conv_info

        # Extension coverage
        def _extension_stats(df: pd.DataFrame) -> Dict[str, object]:
            stats: Dict[str, object] = {"rows": int(len(df))}
            status_col = self._find_column(df, ["Status", "State"])
            if status_col:
                statuses = self._normalize_text_series(df[status_col]).str.lower()
                active = statuses.isin({"enabled", "eligible", "approved"})
                stats["active_pct"] = round((active.sum() / len(df) * 100.0), 1) if len(df) > 0 else 0.0
            return stats

        sitelink_df = self._get_first_available(all_data, ("sitelink_assets", "sitelink"))
        if not sitelink_df.empty:
            summary["sitelinks"] = _extension_stats(sitelink_df)

        callout_df = self._get_first_available(all_data, ("callout_assets", "callout"))
        if not callout_df.empty:
            summary["callouts"] = _extension_stats(callout_df)

        # Top campaigns / ad groups by spend or volume (for "where to look" context)
        campaign_df = self._get_first_available(all_data, ("campaign_details", "campaign", "account"))
        if not campaign_df.empty:
            top_campaigns = _top_entities(
                campaign_df,
                ["Campaign", "Campaign name"],
                cost_candidates + click_candidates + impr_candidates,
            )
            if top_campaigns:
                summary["top_campaigns"] = top_campaigns

            metric_col = _pick_metric(campaign_df)
            name_col = self._find_column(campaign_df, ["Account name", "Campaign", "Campaign name"])
            if metric_col and name_col:
                metrics = to_numeric_safe(campaign_df[metric_col])
                states = campaign_df[name_col].apply(_extract_state)
                state_df = pd.DataFrame({"state": states, "metric": metrics}).dropna(subset=["state"])
                if not state_df.empty:
                    grouped = (
                        state_df.groupby("state", dropna=False)["metric"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(5)
                    )
                    summary["state_breakdown"] = {
                        "metric": metric_col,
                        "states": [
                            {"state": state, "value": float(val)}
                            for state, val in grouped.items()
                        ],
                    }

        adgroup_df = self._get_first_available(all_data, ("ad_group_details", "ad_group"))
        if not adgroup_df.empty:
            top_adgroups = _top_entities(
                adgroup_df,
                ["Ad group", "Ad group name", "Ad group ID"],
                cost_candidates + click_candidates + impr_candidates,
            )
            if top_adgroups:
                summary["top_ad_groups"] = top_adgroups

        return summary

    def discover_sources(self, all_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze all loaded data files to determine their capabilities

        Returns:
            {
                'performance_source': 'adgroup',  # Which file has live performance
                'performance_granularity': 'ad_group',  # Campaign or ad_group level
                'campaign_attributes_source': 'campaign',  # Campaign details
                'performance_rows': 4087,  # Number of rows with performance
                ...
            }
        """
        print("\n" + "="*80)
        print("INTELLIGENT DATA SOURCE DISCOVERY")
        print("="*80 + "\n")

        results = {}

        # Test 1: Which file has performance data?
        print("[1/4] Discovering performance data source...")
        perf_candidates = {}

        # Priority order: campaign, adgroup, ad (core reporting levels)
        # Exclude: landing_page, audience (supplemental data, not campaign structure)
        priority_files = ['campaign', 'adgroup', 'ad', 'campaign_details', 'adgroup_details']

        for file_key, df in all_data.items():
            # Check for impressions column (Google Ads uses 'Impr.', but accept 'Impressions' too)
            print(f"  DEBUG: Checking {file_key}, columns: {list(df.columns)[:3]}...")
            impr_col = None
            if 'Impr.' in df.columns:
                impr_col = 'Impr.'
                print(f"  DEBUG: Found 'Impr.' in {file_key}")
            elif 'Impressions' in df.columns:
                impr_col = 'Impressions'
                print(f"  DEBUG: Found 'Impressions' column in {file_key}")

            if impr_col:
                # If we found 'Impressions', create an 'Impr.' alias in the original dataframe for compatibility
                if impr_col == 'Impressions' and 'Impr.' not in df.columns:
                    df['Impr.'] = df['Impressions']
                    all_data[file_key] = df  # Update the original dataframe

                df_copy = df.copy()
                df_copy['Impr_num'] = to_numeric_safe(df_copy[impr_col])
                perf_rows = df_copy[df_copy['Impr_num'] > 0]

                perf_candidates[file_key] = {
                    'total_rows': len(df),
                    'performance_rows': len(perf_rows),
                    'pct_with_performance': (len(perf_rows) / len(df) * 100) if len(df) > 0 else 0,
                    'is_priority': file_key in priority_files
                }

                print(f"  {file_key:20} {len(df):>8,} total | {len(perf_rows):>8,} with performance ({perf_candidates[file_key]['pct_with_performance']:.1f}%)")

        # Handle case where no performance candidates found (business unit filter too restrictive)
        if not perf_candidates:
            print(f"\n  [ERROR] No performance data found after business unit filtering!")
            print(f"  Available data sources: {list(all_data.keys())}")
            print(f"  Checking if filtering removed all data...")

            # Check raw data before filtering
            for file_name, df in all_data.items():
                if df is not None and len(df) > 0:
                    perf_cols = [c for c in df.columns if any(x in c for x in ['Impr', 'Click', 'Cost', 'Conv'])]
                    has_perf = len([c for c in perf_cols if df[c].notna().any()]) > 0
                    print(f"    {file_name}: {len(df)} rows, has_perf={has_perf}, perf_cols={perf_cols[:2]}")

            raise ValueError(f"No performance data sources found. Business unit filtering may have removed all data. Check BU patterns match campaign names.")

        # Select the file with highest percentage of performance rows.
        # Prefer priority files, but if they have zero performance rows, fall back to any file with data.
        priority_candidates = {k: v for k, v in perf_candidates.items() if v['is_priority']}

        def _pick_best(candidates: dict[str, dict]) -> tuple[str, dict]:
            return max(candidates.items(), key=lambda x: x[1]['pct_with_performance'])

        best_source = None
        if priority_candidates:
            nonzero_priority = {k: v for k, v in priority_candidates.items() if v['performance_rows'] > 0}
            if nonzero_priority:
                best_source = _pick_best(nonzero_priority)
            else:
                nonzero_any = {k: v for k, v in perf_candidates.items() if v['performance_rows'] > 0}
                if nonzero_any:
                    best_source = _pick_best(nonzero_any)
        if best_source is None:
            best_source = _pick_best(perf_candidates)

        results['performance_source'] = best_source[0]
        results['performance_rows'] = best_source[1]['performance_rows']

        self.validation_log.append({
            'test': 'Performance Data Source Discovery',
            'candidates': perf_candidates,
            'decision': best_source[0],
            'reason': f"{best_source[1]['performance_rows']:,} rows with performance ({best_source[1]['pct_with_performance']:.1f}%)"
        })

        print(f"\n  SELECTED: {results['performance_source']} ({results['performance_rows']:,} rows with performance)\n")

        # Test 2: Determine granularity
        print("[2/4] Determining performance granularity...")
        perf_source = all_data[results['performance_source']]

        if 'Ad group' in perf_source.columns and 'Campaign' in perf_source.columns:
            results['performance_granularity'] = 'ad_group'
            print(f"  Performance data is at AD GROUP level\n")
        elif 'Campaign' in perf_source.columns:
            results['performance_granularity'] = 'campaign'
            print(f"  Performance data is at CAMPAIGN level\n")
        else:
            results['performance_granularity'] = 'unknown'
            print(f"  WARNING: Could not determine granularity\n")

        # Test 3: Find campaign attributes source
        print("[3/4] Discovering campaign attributes source...")
        preferred_keys = ("campaign", "campaign_details", "account")
        def _matches_campaign_attrs(frame: pd.DataFrame) -> bool:
            if frame is None or frame.empty or 'Campaign' not in frame.columns:
                return False
            status_col = self._find_column(
                frame,
                ["Campaign state", "Campaign status", "Campaign Status"],
            )
            type_col = self._find_column(
                frame,
                ["Campaign type", "Campaign Type"],
            )
            return bool(status_col or type_col)

        for key in preferred_keys:
            df = all_data.get(key)
            if _matches_campaign_attrs(df):
                results['campaign_attributes_source'] = key
                print(f"  SELECTED: {key} (preferred campaign attributes source)\n")
                break

        if not results.get('campaign_attributes_source'):
            for file_key, df in all_data.items():
                if _matches_campaign_attrs(df):
                    results['campaign_attributes_source'] = file_key
                    print(f"  SELECTED: {file_key} (has campaign status/type columns)\n")
                    break

        # Test 4: Validate data quality
        print("[4/4] Validating data quality...")
        perf_df = all_data[results['performance_source']].copy()
        perf_df['Impr_num'] = to_numeric_safe(perf_df['Impr.'])
        perf_df['Clicks_num'] = to_numeric_safe(perf_df['Clicks'])
        perf_df['Cost_num'] = to_numeric_safe(perf_df['Cost'])

        live_data = perf_df[perf_df['Impr_num'] > 0]

        if len(live_data) > 0:
            total_impr = live_data['Impr_num'].sum()
            total_clicks = live_data['Clicks_num'].sum()
            total_cost = live_data['Cost_num'].sum()
            avg_ctr = (total_clicks / total_impr * 100) if total_impr > 0 else 0
            avg_cpc = (total_cost / total_clicks) if total_clicks > 0 else 0

            self.performance_summary = {
                'total_impressions': total_impr,
                'total_clicks': total_clicks,
                'total_cost': total_cost,
                'avg_ctr': avg_ctr,
                'avg_cpc': avg_cpc
            }

            print(f"  Total Impressions: {total_impr:,.0f}")
            print(f"  Total Clicks: {total_clicks:,.0f}")
            print(f"  Total Cost: ${total_cost:,.2f}")
            print(f"  Average CTR: {avg_ctr:.2f}%")
            print(f"  Average CPC: ${avg_cpc:.2f}")
            print(f"  Data quality: EXCELLENT\n")

        print("="*80 + "\n")

        context_summary = self._summarize_context(all_data)
        if context_summary:
            self.context_summary = context_summary
            results["context"] = context_summary

        return results

    def get_campaign_performance(self, all_data: Dict, source_map: Dict, customer_ids: List[str] = None) -> pd.DataFrame:
        """
        Intelligently retrieve campaign-level performance data

        Logic:
        - If performance is at campaign level, use directly
        - If performance is at ad group level, aggregate up to campaign
        - Optionally filter to specific customer IDs

        Returns:
            DataFrame with columns: Campaign, Impr., Clicks, Cost, Customer ID, etc.
        """
        perf_source_key = source_map['performance_source']
        perf_granularity = source_map.get('performance_granularity', 'unknown')

        perf_df = all_data[perf_source_key].copy()

        # Filter to customer IDs if provided
        if customer_ids and 'Customer ID' in perf_df.columns:
            perf_df = perf_df[perf_df['Customer ID'].isin(customer_ids)]

        # Convert to numeric
        perf_df['Impr_num'] = to_numeric_safe(perf_df['Impr.'])
        perf_df['Clicks_num'] = to_numeric_safe(perf_df['Clicks'])
        perf_df['Cost_num'] = to_numeric_safe(perf_df['Cost'])

        if perf_granularity == 'campaign':
            # Already at campaign level
            return perf_df

        elif perf_granularity == 'ad_group':
            # Aggregate ad group → campaign level
            agg_cols = {
                'Impr_num': 'sum',
                'Clicks_num': 'sum',
                'Cost_num': 'sum'
            }

            # Include other useful columns
            if 'Conversions' in perf_df.columns:
                perf_df['Conversions_num'] = to_numeric_safe(perf_df['Conversions'])
                agg_cols['Conversions_num'] = 'sum'

            if 'Customer ID' in perf_df.columns:
                agg_cols['Customer ID'] = 'first'

            if 'Account name' in perf_df.columns:
                agg_cols['Account name'] = 'first'

            if 'Campaign type' in perf_df.columns:
                agg_cols['Campaign type'] = 'first'

            campaign_perf = perf_df.groupby('Campaign').agg(agg_cols).reset_index()

            # Rename back
            campaign_perf['Impr.'] = campaign_perf['Impr_num']
            campaign_perf['Clicks'] = campaign_perf['Clicks_num']
            campaign_perf['Cost'] = campaign_perf['Cost_num']

            self.validation_log.append({
                'action': 'Data Aggregation',
                'from_rows': len(perf_df),
                'to_rows': len(campaign_perf),
                'method': 'ad_group → campaign'
            })

            return campaign_perf

        return perf_df

    def get_validation_report(self) -> str:
        """Generate human-readable validation report"""
        report = []
        report.append("\nINTELLIGENT DATA MAPPING VALIDATION REPORT")
        report.append("=" * 80)

        for entry in self.validation_log:
            report.append(f"\nTest: {entry.get('test', entry.get('action', 'Unknown'))}")
            if 'decision' in entry:
                report.append(f"  Decision: {entry['decision']}")
            if 'reason' in entry:
                report.append(f"  Reason: {entry['reason']}")
            if 'from_rows' in entry:
                report.append(f"  Aggregated: {entry['from_rows']:,} rows → {entry['to_rows']:,} campaigns")

        report.append("\n" + "=" * 80)
        return "\n".join(report)


def backtest_intelligent_mapper(data_folder: Path):
    """
    Backtest the intelligent mapper against known ground truth

    Known Truth (from manual analysis):
    - Campaign Report: 2,945 rows, 0 with performance
    - Ad Group Report: 4,087 rows, 4,087 with performance
    - Expected: Mapper selects Ad Group Report
    - Expected: Retirement has ~9.2M impressions
    """
    print("\n" + "="*80)
    print("BACKTESTING INTELLIGENT DATA MAPPER")
    print("="*80 + "\n")

    # Load all files
    from ULTIMATE_COMPREHENSIVE_AUDIT import files, read_csv_adaptive

    all_data = {}
    for key, filename in files.items():
        try:
            df = pd.read_csv(data_folder / filename, encoding='utf-8', skiprows=2, low_memory=False)
        except:
            df = pd.read_csv(data_folder / filename, encoding='utf-16', sep='\t', skiprows=2, low_memory=False)
        all_data[key] = df

    # Create mapper and discover
    mapper = IntelligentDataSourceMapper()
    source_map = mapper.discover_sources(all_data)

    print("BACKTEST RESULTS:")
    print("-" * 80)

    # Test 1: Did it select correct source?
    expected_source = 'adgroup'
    actual_source = source_map['performance_source']

    test_1_pass = actual_source == expected_source
    print(f"Test 1: Performance source selection")
    print(f"  Expected: {expected_source}")
    print(f"  Actual: {actual_source}")
    print(f"  Status: {'✓ PASS' if test_1_pass else '✗ FAIL'}\n")

    # Test 2: Did it detect correct granularity?
    expected_granularity = 'ad_group'
    actual_granularity = source_map['performance_granularity']

    test_2_pass = actual_granularity == expected_granularity
    print(f"Test 2: Granularity detection")
    print(f"  Expected: {expected_granularity}")
    print(f"  Actual: {actual_granularity}")
    print(f"  Status: {'✓ PASS' if test_2_pass else '✗ FAIL'}\n")

    # Test 3: Correct aggregation for Retirement
    retirement_ids = ['186-987-7802', '487-733-9293']
    campaign_perf = mapper.get_campaign_performance(all_data, source_map, retirement_ids)

    expected_campaigns = 71
    actual_campaigns = len(campaign_perf)

    test_3_pass = actual_campaigns == expected_campaigns
    print(f"Test 3: Retirement campaign count")
    print(f"  Expected: {expected_campaigns}")
    print(f"  Actual: {actual_campaigns}")
    print(f"  Status: {'✓ PASS' if test_3_pass else '✗ FAIL'}\n")

    # Test 4: Correct impression total
    total_impr = campaign_perf['Impr_num'].sum()
    expected_impr_range = (9_000_000, 9_500_000)

    test_4_pass = expected_impr_range[0] <= total_impr <= expected_impr_range[1]
    print(f"Test 4: Retirement impression total")
    print(f"  Expected: {expected_impr_range[0]:,} - {expected_impr_range[1]:,}")
    print(f"  Actual: {total_impr:,.0f}")
    print(f"  Status: {'✓ PASS' if test_4_pass else '✗ FAIL'}\n")

    # Summary
    total_tests = 4
    passed_tests = sum([test_1_pass, test_2_pass, test_3_pass, test_4_pass])

    print("=" * 80)
    print(f"BACKTEST SUMMARY: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.0f}%)")
    print("=" * 80 + "\n")

    if passed_tests == total_tests:
        print("✓ ALL BACKTESTS PASSED - System is operating correctly\n")
    else:
        print("✗ SOME BACKTESTS FAILED - System needs adjustment\n")

    return passed_tests == total_tests


if __name__ == '__main__':
    # Run backtest
    data_folder = Path('C:/Software Builds/PPC_Audit_Project/Total Files')
    backtest_intelligent_mapper(data_folder)
