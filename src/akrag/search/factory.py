from __future__ import annotations

from functools import lru_cache
from akrag.search.base import SearchBackend


@lru_cache(maxsize=1)
def get_search_backend() -> SearchBackend:
    from akrag.config import get_settings
    from akrag.embeddings.factory import get_embedder

    s = get_settings()
    backend = s.search_backend
    dim = get_embedder().dimension

    if backend == "local":
        from akrag.search.local import LocalSearchBackend
        return LocalSearchBackend(rrf_k=s.rrf_k)

    if backend == "opensearch":
        from akrag.search.opensearch_backend import OpenSearchBackend
        return OpenSearchBackend(
            host=s.opensearch_host,
            port=s.opensearch_port,
            user=s.opensearch_user,
            password=s.opensearch_pass,
            use_ssl=s.opensearch_use_ssl,
            index_prefix=s.opensearch_index_prefix,
            vector_dim=dim,
            rrf_k=s.rrf_k,
        )

    if backend == "faiss":
        from akrag.search.faiss_backend import FAISSBackend
        return FAISSBackend(vector_dim=dim, rrf_k=s.rrf_k)

    if backend == "chroma":
        from akrag.search.chroma_backend import ChromaBackend
        return ChromaBackend(host=s.chroma_host, port=s.chroma_port, rrf_k=s.rrf_k)

    raise ValueError(f"Unknown search backend: {backend}")
