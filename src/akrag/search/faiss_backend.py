from __future__ import annotations

import numpy as np
from akrag.models import AttributeDocument, SearchResult
from akrag.search.base import SearchBackend, rrf_fusion


class FAISSBackend(SearchBackend):
    """FAISS HNSW vector index + BM25 lexical search + RRF.

    Fast, in-process, no server required. Good for local dev when the
    local backend is too slow at large corpus sizes.
    """

    def __init__(self, vector_dim: int = 384, rrf_k: int = 60) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError as exc:
            raise ImportError("Install faiss-cpu: pip install akrag[faiss]") from exc
        self._dim = vector_dim
        self._rrf_k = rrf_k
        self._indexes: dict[str, tuple[object, list[AttributeDocument]]] = {}
        self._current: str = "attributes_faiss"

    async def index(
        self,
        documents: list[AttributeDocument],
        index_name: str | None = None,
    ) -> str:
        import faiss

        name = index_name or self._current
        docs_with_vectors = [d for d in documents if d.vector is not None]

        idx = faiss.IndexHNSWFlat(self._dim, 32)
        idx.hnsw.efConstruction = 200

        if docs_with_vectors:
            matrix = np.array([d.vector for d in docs_with_vectors], dtype=np.float32)
            faiss.normalize_L2(matrix)
            idx.add(matrix)

        self._indexes[name] = (idx, docs_with_vectors)
        self._current = name
        return name

    async def delete_index(self, index_name: str) -> None:
        self._indexes.pop(index_name, None)

    async def list_indexes(self) -> list[str]:
        return list(self._indexes.keys())

    async def search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 10,
        index_name: str | None = None,
        channel: str | None = None,
    ) -> list[SearchResult]:
        import faiss

        name = index_name or self._current
        if name not in self._indexes:
            return []

        faiss_index, docs = self._indexes[name]

        if channel:
            docs = [
                d for d in docs
                if not d.governance.allowed_channels
                or channel in d.governance.allowed_channels
            ]

        # BM25
        bm25_results = self._bm25_search(query, docs, top_k)

        # FAISS kNN
        q = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(q)
        distances, indices = faiss_index.search(q, min(top_k, len(docs)))
        vector_results = [
            (docs[i], float(d))
            for i, d in zip(indices[0], distances[0])
            if i >= 0 and i < len(docs)
        ]

        return rrf_fusion(bm25_results, vector_results, k=self._rrf_k)[:top_k]

    def _bm25_search(
        self, query: str, docs: list[AttributeDocument], top_k: int
    ) -> list[tuple[AttributeDocument, float]]:
        from rank_bm25 import BM25Okapi
        import re

        def tok(t: str) -> list[str]:
            return re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()

        corpus = [tok(d.embedding_text or d.compose_embedding_text()) for d in docs]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tok(query))
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(docs[i], float(s)) for i, s in indexed if s > 0]
