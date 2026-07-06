from __future__ import annotations

from abc import ABC, abstractmethod
from akrag.models import AttributeDocument, SearchResult


class SearchBackend(ABC):
    """Abstract base for all search backends.

    All backends must implement the same hybrid search contract so the decision
    engine can be swapped without changing retrieval logic.
    """

    @abstractmethod
    async def index(
        self,
        documents: list[AttributeDocument],
        index_name: str | None = None,
    ) -> str:
        """Index a list of attribute documents. Returns the index name used."""

    @abstractmethod
    async def search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 10,
        index_name: str | None = None,
        channel: str | None = None,
    ) -> list[SearchResult]:
        """Hybrid search: BM25 + vector + RRF. Returns ranked results."""

    @abstractmethod
    async def delete_index(self, index_name: str) -> None:
        """Remove an index (used during rollback or cleanup)."""

    @abstractmethod
    async def list_indexes(self) -> list[str]:
        """Return available index names."""

    async def swap_alias(self, alias: str, new_index: str) -> None:
        """Atomically point alias → new_index. Override in backends that support aliases."""
        pass  # no-op for local backends that don't use aliases


def rrf_fusion(
    bm25_results: list[tuple[AttributeDocument, float]],
    vector_results: list[tuple[AttributeDocument, float]],
    k: int = 60,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion: combines two ranked lists into one fused ranking."""
    scores: dict[str, dict] = {}

    for rank, (doc, bm25_score) in enumerate(bm25_results):
        aid = doc.attribute_id
        scores.setdefault(aid, {"doc": doc, "rrf": 0.0, "bm25_rank": None,
                                 "vector_rank": None, "bm25_score": 0.0, "vector_score": 0.0})
        scores[aid]["rrf"] += 1.0 / (k + rank + 1)
        scores[aid]["bm25_rank"] = rank
        scores[aid]["bm25_score"] = bm25_score

    for rank, (doc, vector_score) in enumerate(vector_results):
        aid = doc.attribute_id
        scores.setdefault(aid, {"doc": doc, "rrf": 0.0, "bm25_rank": None,
                                 "vector_rank": None, "bm25_score": 0.0, "vector_score": 0.0})
        scores[aid]["rrf"] += 1.0 / (k + rank + 1)
        scores[aid]["vector_rank"] = rank
        scores[aid]["vector_score"] = vector_score

    ranked = sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)
    return [
        SearchResult(
            attribute=entry["doc"],
            bm25_score=entry["bm25_score"],
            vector_score=entry["vector_score"],
            rrf_score=entry["rrf"],
            bm25_rank=entry["bm25_rank"],
            vector_rank=entry["vector_rank"],
        )
        for entry in ranked
    ]
