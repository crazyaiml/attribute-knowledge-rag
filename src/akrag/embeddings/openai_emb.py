from __future__ import annotations

from akrag.embeddings.base import EmbeddingProvider

_DIM_MAP = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(EmbeddingProvider):
    """OpenAI Embeddings API."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = "") -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("Install openai: pip install akrag[openai]") from exc
        self._client = AsyncOpenAI(api_key=api_key or None)
        self._model = model

    @property
    def dimension(self) -> int:
        return _DIM_MAP.get(self._model, 1536)

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
