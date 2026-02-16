import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: gen_multisheet_xlsx.py <output_path>", file=sys.stderr)
        return 2

    out_path = Path(sys.argv[1]).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep this tiny and deterministic: it exists only to validate that XLSX *multi-sheet*
    # inputs are discovered and ingested across the upload pipeline (not to validate audit
    # scoring correctness). We intentionally include 7+ sheets with varied headers so the
    # intelligent mapper sees multiple plausible table "shapes".
    cust = "7902313748"
    acct = "QA MultiSheet"

    df_campaigns = pd.DataFrame(
        {
            "Customer ID": [cust, cust],
            "Account name": [acct, acct],
            "Campaign": ["Campaign A", "Campaign B"],
            "Clicks": [10, 20],
            "Impressions": [100, 210],
            "Cost": [12.34, 56.78],
            "Conversions": [1, 2],
        }
    )

    df_adgroups = pd.DataFrame(
        {
            "Customer ID": [cust],
            "Account name": [acct],
            "Campaign": ["Campaign A"],
            "Ad group": ["Ad Group 1"],
            "Clicks": [7],
            "Cost": [8.9],
        }
    )

    df_ads = pd.DataFrame(
        {
            "Customer ID": [cust],
            "Account name": [acct],
            "Campaign": ["Campaign A"],
            "Ad group": ["Ad Group 1"],
            "Ad headline": ["Test Headline"],
            "CTR": [0.123],
            "CPC": [1.23],
        }
    )

    df_keywords = pd.DataFrame(
        {
            "Customer ID": [cust, cust],
            "Account name": [acct, acct],
            "Campaign": ["Campaign A", "Campaign A"],
            "Keyword": ["foo", "bar"],
            "Match type": ["Exact", "Phrase"],
            "Impressions": [100, 200],
            "Clicks": [4, 3],
        }
    )

    df_search_terms = pd.DataFrame(
        {
            "Customer ID": [cust, cust],
            "Account name": [acct, acct],
            "Search term": ["cheap motor oil", "synthetic oil deals"],
            "Clicks": [2, 1],
            "Cost": [1.11, 0.99],
        }
    )

    df_conversions = pd.DataFrame(
        {
            "Customer ID": [cust, cust],
            "Account name": [acct, acct],
            "Conversion action": ["FR_Intent_Clicks", "Store Visits"],
            "Conversions": [3, 0],
            "All conv.": [3, 5],
        }
    )

    df_urls = pd.DataFrame(
        {
            "Customer ID": [cust],
            "Account name": [acct],
            "Final URL": ["https://example.com/landing"],
            "Status": [200],
        }
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_campaigns.to_excel(writer, index=False, sheet_name="Campaigns")
        df_adgroups.to_excel(writer, index=False, sheet_name="AdGroups")
        df_ads.to_excel(writer, index=False, sheet_name="Ads")
        df_keywords.to_excel(writer, index=False, sheet_name="Keywords")
        df_search_terms.to_excel(writer, index=False, sheet_name="SearchTerms")
        df_conversions.to_excel(writer, index=False, sheet_name="Conversions")
        df_urls.to_excel(writer, index=False, sheet_name="LandingPages")

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

