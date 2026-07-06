from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from akrag.models import AttributeDocument, SearchResult
from akrag.search.base import SearchBackend, rrf_fusion


class LocalSearchBackend(SearchBackend):
    """In-memory hybrid search: BM25 (rank-bm25) + cosine similarity + RRF.

    Zero server dependencies — ideal for unit tests, local dev, and CI.
    Identical decision policy as the OpenSearch backend.
    """

    def __init__(self, rrf_k: int = 60) -> None:
        self._rrf_k = rrf_k
        self._indexes: dict[str, list[AttributeDocument]] = {}
        self._current_index: str = "attributes_local"

    # ── Index management ──────────────────────────────────────────────────────

    async def index(
        self,
        documents: list[AttributeDocument],
        index_name: str | None = None,
    ) -> str:
        name = index_name or self._current_index
        self._indexes[name] = list(documents)
        self._current_index = name
        return name

    async def delete_index(self, index_name: str) -> None:
        self._indexes.pop(index_name, None)

    async def list_indexes(self) -> list[str]:
        return list(self._indexes.keys())

    # ── Search ────────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 10,
        index_name: str | None = None,
        channel: str | None = None,
    ) -> list[SearchResult]:
        name = index_name or self._current_index
        docs = self._indexes.get(name, [])
        if not docs:
            return []

        if channel:
            docs = [
                d for d in docs
                if not d.governance.allowed_channels
                or channel in d.governance.allowed_channels
            ]

        bm25_results = self._bm25_search(query, docs, top_k)
        vector_results = self._vector_search(query_vector, docs, top_k)

        return rrf_fusion(bm25_results, vector_results, k=self._rrf_k)[:top_k]

    # ── BM25 (rank-bm25) ─────────────────────────────────────────────────────

    def _bm25_search(
        self, query: str, docs: list[AttributeDocument], top_k: int
    ) -> list[tuple[AttributeDocument, float]]:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError("Install rank-bm25: pip install rank-bm25") from exc

        corpus = [self._tokenize(d.embedding_text or d.compose_embedding_text()) for d in docs]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(self._tokenize(query))
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(docs[i], float(score)) for i, score in indexed if score > 0]

    _STOPWORDS = {
        "a", "an", "and", "at", "by", "for", "from", "in", "into", "is", "of",
        "on", "or", "over", "the", "to", "under", "with",
    }

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        import re
        tokens = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
        return [t for t in tokens if t not in cls._STOPWORDS]

    # ── Cosine similarity ─────────────────────────────────────────────────────

    def _vector_search(
        self,
        query_vector: list[float],
        docs: list[AttributeDocument],
        top_k: int,
    ) -> list[tuple[AttributeDocument, float]]:
        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        results = []
        for doc in docs:
            if doc.vector is None:
                continue
            v = np.array(doc.vector, dtype=np.float32)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                continue
            sim = float(np.dot(q, v) / (q_norm * v_norm))
            results.append((doc, sim))

        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
