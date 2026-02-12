"""
Rebuild the comprehensive data cache using the Intelligent Data Sandbox.
This ensures Campaign ID/Ad Group mappings are persisted for rapid reuse.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from INTELLIGENT_DATA_SANDBOX import IntelligentDataSandbox


def main() -> None:
    root_dir = Path(__file__).resolve().parents[2]
    data_dir = (root_dir / "PPC_Audit_Project" / "Total Files").resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    output_path = Path(__file__).resolve().parent / "comprehensive_data_cache.pkl"

    print("=" * 100)
    print("CREATING INTELLIGENT DATA SANDBOX CACHE")
    print("=" * 100)
    print(f"Data directory: {data_dir}")
    print(f"Output cache : {output_path}")

    sandbox = IntelligentDataSandbox(str(data_dir))
    sandbox.initialize(force_rebuild=True)

    cache_payload = {
        **sandbox.raw_data,
        "lookups": sandbox.entity_maps,
        "metadata": sandbox.metadata,
    }

    with open(output_path, "wb") as fh:
        pickle.dump(cache_payload, fh)

    print("\nCACHE SUMMARY")
    print("-" * 80)
    for key, df in sandbox.raw_data.items():
        print(f"{key:25} {len(df):>8,} rows | columns: {len(df.columns)}")
    print("-" * 80)
    print(f"Campaign ID mappings: {len(sandbox.entity_maps.get('campaign_id_to_account', {}))}")
    print(f"Ad group mappings   : {len(sandbox.entity_maps.get('adgroup_to_campaign', {}))}")
    print("=" * 100)
    print("CACHE CREATION COMPLETE")


if __name__ == "__main__":
    main()
