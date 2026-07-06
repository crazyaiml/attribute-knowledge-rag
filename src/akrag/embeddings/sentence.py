from __future__ import annotations

import asyncio
from akrag.embeddings.base import EmbeddingProvider

_DIM_MAP = {
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}


class SentenceEmbedder(EmbeddingProvider):
    """Local sentence-transformers embedding (no API key, free, runs on CPU/GPU)."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers: pip install akrag[sentence]"
            ) from exc
        self._model_name = model
        self._encoder = SentenceTransformer(model)

    @property
    def dimension(self) -> int:
        return _DIM_MAP.get(self._model_name, self._encoder.get_sentence_embedding_dimension())

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        vecs = await loop.run_in_executor(
            None, lambda: self._encoder.encode(texts, normalize_embeddings=True)
        )
        return [v.tolist() for v in vecs]
