from __future__ import annotations

import json
from datetime import date

from akrag.models import AttributeDocument, SearchResult
from akrag.search.base import SearchBackend


def _versioned_name(prefix: str = "attributes") -> str:
    return f"{prefix}_v{date.today().strftime('%Y_%m_%d')}"


def _build_mapping(vector_dim: int) -> dict:
    return {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 512,
            }
        },
        "mappings": {
            "properties": {
                "attribute_id": {"type": "keyword"},
                "type": {"type": "keyword"},
                "business_name": {"type": "text", "analyzer": "english"},
                "technical_field": {"type": "keyword"},
                "domain": {"type": "keyword"},
                "definition": {"type": "text", "analyzer": "english"},
                "synonyms": {"type": "text", "analyzer": "english"},
                "allowed_values": {"type": "keyword"},
                "unit": {"type": "keyword"},
                "governance": {
                    "properties": {
                        "pii": {"type": "boolean"},
                        "allowed_channels": {"type": "keyword"},
                    }
                },
                "embedding_text": {"type": "text", "analyzer": "english"},
                "embedding_model": {"type": "keyword"},
                "embedding_version": {"type": "keyword"},
                "vector": {
                    "type": "knn_vector",
                    "dimension": vector_dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {"ef_construction": 512, "m": 16},
                    },
                },
            }
        },
    }


class OpenSearchBackend(SearchBackend):
    """Production search: OpenSearch BM25 + HNSW kNN + RRF via hybrid query."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        user: str = "admin",
        password: str = "admin",
        use_ssl: bool = False,
        index_prefix: str = "attributes",
        vector_dim: int = 384,
        rrf_k: int = 60,
    ) -> None:
        try:
            from opensearchpy import OpenSearch
        except ImportError as exc:
            raise ImportError("Install opensearch-py: pip install akrag[opensearch]") from exc

        self._os = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=(user, password),
            use_ssl=use_ssl,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        self._prefix = index_prefix
        self._dim = vector_dim
        self._rrf_k = rrf_k
        self._alias = f"{index_prefix}_current"

    # ── Index management ──────────────────────────────────────────────────────

    async def index(
        self,
        documents: list[AttributeDocument],
        index_name: str | None = None,
    ) -> str:
        import asyncio

        name = index_name or _versioned_name(self._prefix)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_index, documents, name)
        return name

    def _sync_index(self, documents: list[AttributeDocument], index_name: str) -> None:
        mapping = _build_mapping(self._dim)
        if self._os.indices.exists(index=index_name):
            self._os.indices.delete(index=index_name)
        self._os.indices.create(index=index_name, body=mapping)

        actions = []
        for doc in documents:
            actions.append(json.dumps({"index": {"_index": index_name, "_id": doc.attribute_id}}))
            d = doc.model_dump()
            actions.append(json.dumps(d))

        if actions:
            body = "\n".join(actions) + "\n"
            self._os.bulk(body=body)

    async def delete_index(self, index_name: str) -> None:
        import asyncio

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._os.indices.delete(index=index_name))

    async def list_indexes(self) -> list[str]:
        import asyncio

        loop = asyncio.get_event_loop()
        cat = await loop.run_in_executor(None, lambda: self._os.cat.indices(format="json"))
        return [entry["index"] for entry in cat if entry["index"].startswith(self._prefix)]

    async def swap_alias(self, alias: str, new_index: str) -> None:
        import asyncio

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_swap, alias, new_index)

    def _sync_swap(self, alias: str, new_index: str) -> None:
        actions: list[dict] = []
        existing = self._os.indices.get_alias(name=alias, ignore_unavailable=True)
        for old_index in existing:
            actions.append({"remove": {"index": old_index, "alias": alias}})
        actions.append({"add": {"index": new_index, "alias": alias}})
        self._os.indices.update_aliases(body={"actions": actions})

    # ── Search ────────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 10,
        index_name: str | None = None,
        channel: str | None = None,
    ) -> list[SearchResult]:
        import asyncio

        target = index_name or self._alias
        loop = asyncio.get_event_loop()
        hits = await loop.run_in_executor(
            None, self._sync_search, query, query_vector, top_k, target, channel
        )
        return hits

    def _sync_search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int,
        index_name: str,
        channel: str | None,
    ) -> list[SearchResult]:
        # Build channel filter
        filters = []
        if channel:
            filters.append({
                "bool": {
                    "should": [
                        {"term": {"governance.allowed_channels": channel}},
                        {"bool": {"must_not": {"exists": {"field": "governance.allowed_channels"}}}},
                    ]
                }
            })

        query_body: dict = {
            "size": top_k,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "business_name^3",
                                    "synonyms^2",
                                    "definition",
                                    "embedding_text",
                                    "technical_field",
                                ],
                            }
                        },
                        {
                            "knn": {
                                "vector": {
                                    "vector": query_vector,
                                    "k": top_k,
                                }
                            }
                        },
                    ]
                }
            },
            "search_pipeline": {"phase_results_processors": [{"normalization-processor": {
                "normalization": {"technique": "min_max"},
                "combination": {"technique": "rrf", "parameters": {"rank_constant": self._rrf_k}},
            }}]},
        }

        if filters:
            query_body["post_filter"] = {"bool": {"filter": filters}}

        resp = self._os.search(index=index_name, body=query_body)
        results = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            doc = AttributeDocument(**src)
            results.append(SearchResult(
                attribute=doc,
                rrf_score=hit["_score"],
            ))
        return results
