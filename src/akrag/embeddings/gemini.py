from __future__ import annotations

from akrag.embeddings.base import EmbeddingProvider


class GeminiEmbedder(EmbeddingProvider):
    """Google Gemini text-embedding-004 (768 dims)."""

    def __init__(self, model: str = "models/text-embedding-004", api_key: str = "") -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError("Install google-generativeai: pip install akrag[gemini]") from exc
        if api_key:
            genai.configure(api_key=api_key)
        self._genai = genai
        self._model = model

    @property
    def dimension(self) -> int:
        return 768

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_embed, texts)

    def _sync_embed(self, texts: list[str]) -> list[list[float]]:
        result = self._genai.embed_content(
            model=self._model,
            content=texts,
            task_type="retrieval_document",
        )
        embeddings = result["embedding"]
        if isinstance(embeddings[0], float):
            return [embeddings]
        return [list(e) for e in embeddings]
