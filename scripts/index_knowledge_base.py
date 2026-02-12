from __future__ import annotations

import argparse
from pathlib import Path

from kai_core.shared.vector_index import index_knowledge_base


def main() -> int:
    parser = argparse.ArgumentParser(description="Index knowledge base docs into Azure Search.")
    parser.add_argument("--root", required=True, help="Root folder containing .md/.txt knowledge base files")
    parser.add_argument("--namespace", default="knowledge_base", help="Document id namespace")
    parser.add_argument("--purge", action="store_true", help="Delete existing knowledge_base docs before indexing")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Folder not found: {root}")

    result = index_knowledge_base(root, namespace=args.namespace, purge_existing=args.purge)
    status = result.get("status", "unknown")
    print(f"knowledge_base: {status} ({result})")
    return 1 if status == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
