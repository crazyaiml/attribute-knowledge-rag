from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base for all embedding backends."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output vector dimension."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Canonical model identifier (used for tagging documents)."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per text."""

    async def embed_one(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]
