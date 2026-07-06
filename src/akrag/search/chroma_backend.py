from __future__ import annotations

import re
from akrag.models import AttributeDocument, SearchResult
from akrag.search.base import SearchBackend, rrf_fusion


def _tok(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


class ChromaBackend(SearchBackend):
    """ChromaDB vector store + BM25 lexical search + RRF.

    ChromaDB is a persistent or ephemeral vector store with a simple HTTP server.
    Run locally with: chroma run --path ./chroma_data
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        rrf_k: int = 60,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError("Install chromadb: pip install akrag[chroma]") from exc

        self._client = chromadb.HttpClient(host=host, port=port)
        self._rrf_k = rrf_k
        self._docs_cache: dict[str, list[AttributeDocument]] = {}
        self._current: str = "attributes_chroma"

    async def index(
        self,
        documents: list[AttributeDocument],
        index_name: str | None = None,
    ) -> str:
        import asyncio

        name = (index_name or self._current).replace(".", "_")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_index, documents, name)
        self._docs_cache[name] = documents
        self._current = name
        return name

    def _sync_index(self, documents: list[AttributeDocument], name: str) -> None:
        try:
            self._client.delete_collection(name)
        except Exception:
            pass

        col = self._client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

        docs_with_vectors = [d for d in documents if d.vector is not None]
        if not docs_with_vectors:
            return

        col.add(
            ids=[d.attribute_id for d in docs_with_vectors],
            embeddings=[d.vector for d in docs_with_vectors],
            documents=[d.embedding_text or d.compose_embedding_text() for d in docs_with_vectors],
            metadatas=[{"business_name": d.business_name, "domain": d.domain}
                       for d in docs_with_vectors],
        )

    async def delete_index(self, index_name: str) -> None:
        import asyncio

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._client.delete_collection(index_name))
        self._docs_cache.pop(index_name, None)

    async def list_indexes(self) -> list[str]:
        import asyncio

        loop = asyncio.get_event_loop()
        cols = await loop.run_in_executor(None, self._client.list_collections)
        return [c.name for c in cols]

    async def search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 10,
        index_name: str | None = None,
        channel: str | None = None,
    ) -> list[SearchResult]:
        import asyncio

        name = (index_name or self._current).replace(".", "_")
        docs = self._docs_cache.get(name, [])

        if channel:
            docs = [
                d for d in docs
                if not d.governance.allowed_channels
                or channel in d.governance.allowed_channels
            ]

        # BM25 on cached documents
        bm25_results = self._bm25_search(query, docs, top_k)

        # Chroma vector search
        loop = asyncio.get_event_loop()
        vector_results = await loop.run_in_executor(
            None, self._sync_vector_search, name, query_vector, docs, top_k
        )

        return rrf_fusion(bm25_results, vector_results, k=self._rrf_k)[:top_k]

    def _sync_vector_search(
        self,
        name: str,
        query_vector: list[float],
        all_docs: list[AttributeDocument],
        top_k: int,
    ) -> list[tuple[AttributeDocument, float]]:
        col = self._client.get_collection(name)
        results = col.query(query_embeddings=[query_vector], n_results=min(top_k, col.count()))
        doc_map = {d.attribute_id: d for d in all_docs}
        out = []
        for aid, dist in zip(results["ids"][0], results["distances"][0]):
            doc = doc_map.get(aid)
            if doc:
                out.append((doc, 1.0 - dist))  # chroma returns distance; convert to similarity
        return out

    def _bm25_search(
        self, query: str, docs: list[AttributeDocument], top_k: int
    ) -> list[tuple[AttributeDocument, float]]:
        from rank_bm25 import BM25Okapi

        if not docs:
            return []
        corpus = [_tok(d.embedding_text or d.compose_embedding_text()) for d in docs]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(_tok(query))
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(docs[i], float(s)) for i, s in indexed if s > 0]
