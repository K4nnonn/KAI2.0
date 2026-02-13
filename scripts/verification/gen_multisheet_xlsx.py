import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: gen_multisheet_xlsx.py <output_path>", file=sys.stderr)
        return 2

    out_path = Path(sys.argv[1]).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep this tiny and deterministic: it exists to validate that XLSX multi-sheet
    # inputs are discovered, ingested, and usable by the audit pipeline (not to validate
    # audit scoring correctness).
    #
    # IMPORTANT: The IntelligentDataMapper expects Google Ads-style headers including
    # "Impr.", "Clicks", and "Cost". If the campaigns sheet lacks impressions, the mapper
    # may incorrectly select the keyword sheet as the performance source and later crash
    # on missing campaign-level columns. This fixture therefore includes the minimal
    # required performance columns on both sheets.
    df_campaigns = pd.DataFrame(
        {
            "Customer ID": ["7902313748", "7902313748"],
            "Account name": ["QA MultiSheet", "QA MultiSheet"],
            "Campaign": ["Campaign A", "Campaign B"],
            "Impr.": [1000, 2000],
            "Clicks": [10, 20],
            "Cost": [12.34, 56.78],
            "Conversions": [1, 2],
            "Campaign Status": ["Enabled", "Enabled"],
            "Campaign Type": ["Search", "Search"],
        }
    )
    df_keywords = pd.DataFrame(
        {
            "Customer ID": ["7902313748", "7902313748"],
            "Account name": ["QA MultiSheet", "QA MultiSheet"],
            "Campaign": ["Campaign A", "Campaign A"],
            "Keyword": ["foo", "bar"],
            "Match type": ["Exact", "Phrase"],
            "Impr.": [100, 200],
            "Clicks": [1, 2],
            "Cost": [0.5, 1.25],
            "Conversions": [0, 1],
        }
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_campaigns.to_excel(writer, index=False, sheet_name="Campaigns")
        df_keywords.to_excel(writer, index=False, sheet_name="Keywords")

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
