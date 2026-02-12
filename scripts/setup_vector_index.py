from __future__ import annotations

import argparse
import os
import sys

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)


def _build_index(name: str, vector_field: str, dimensions: int) -> SearchIndex:
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SimpleField(name="accountName", type=SearchFieldDataType.String, filterable=True, sortable=True, facetable=True),
        SimpleField(name="auditDate", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SimpleField(name="section", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name=vector_field,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dimensions,
            vector_search_profile_name="vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config",
                parameters=HnswParameters(
                    metric=VectorSearchAlgorithmMetric.COSINE,
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                ),
            )
        ],
        profiles=[VectorSearchProfile(name="vector-profile", algorithm_configuration_name="hnsw-config")],
    )

    semantic_settings = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="kai-semantic",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[SemanticField(field_name="section"), SemanticField(field_name="accountName")],
                ),
            )
        ]
    )

    return SearchIndex(
        name=name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_settings,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or update the Azure Search vector index for Kai.")
    parser.add_argument("--endpoint", default=os.environ.get("CONCIERGE_SEARCH_ENDPOINT", "").strip())
    parser.add_argument("--key", default=os.environ.get("CONCIERGE_SEARCH_KEY", "").strip())
    parser.add_argument("--index", default=os.environ.get("CONCIERGE_SEARCH_INDEX", "").strip())
    parser.add_argument(
        "--vector-field",
        default=os.environ.get("CONCIERGE_SEARCH_VECTOR_FIELD", "contentVector").strip() or "contentVector",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=int(os.environ.get("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536") or 1536),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite index if it exists.")
    args = parser.parse_args()

    if not args.endpoint or not args.key or not args.index:
        print("Missing endpoint/key/index. Set CONCIERGE_SEARCH_ENDPOINT/KEY/INDEX or pass args.", file=sys.stderr)
        return 2

    client = SearchIndexClient(endpoint=args.endpoint, credential=AzureKeyCredential(args.key))
    try:
        existing = client.get_index(args.index)
        if not args.force:
            print(f"Index already exists: {existing.name}. Use --force to update.")
            return 0
    except Exception:
        existing = None

    index = _build_index(args.index, args.vector_field, args.dimensions)
    if existing and args.force:
        client.create_or_update_index(index)
        print(f"Index updated: {args.index}")
        return 0

    client.create_index(index)
    print(f"Index created: {args.index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
