from __future__ import annotations

import argparse
import re
from pathlib import Path

from kai_core.shared.vector_index import index_audit_workbook


def _infer_account_name(path: Path) -> str:
    name = path.stem
    name = re.sub(r"^(Audit_|Kai_)", "", name)
    name = re.sub(r"_\\d{8}_\\d{6}.*$", "", name)
    name = name.replace("__", " ")
    name = name.replace("_", " ")
    return name.strip() or name


def main() -> int:
    parser = argparse.ArgumentParser(description="Index Kai audit Excel outputs into Azure Search.")
    parser.add_argument("--root", required=True, help="Root folder containing Audit_*.xlsx or Kai_*.xlsx files")
    parser.add_argument("--account", default="", help="Override account name for all files")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Folder not found: {root}")

    files = sorted(root.glob("**/*.xlsx"))
    if not files:
        raise SystemExit("No .xlsx files found.")

    for xlsx in files:
        account_name = args.account or _infer_account_name(xlsx)
        result = index_audit_workbook(xlsx, account_name)
        status = result.get("status")
        print(f"{xlsx.name}: {status} ({result})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
