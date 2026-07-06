from __future__ import annotations

from functools import lru_cache
from akrag.embeddings.base import EmbeddingProvider


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingProvider:
    from akrag.config import get_settings
    s = get_settings()
    provider = s.embedding_provider
    model = s.resolved_embedding_model

    if provider == "gemini":
        from akrag.embeddings.gemini import GeminiEmbedder
        return GeminiEmbedder(model=model, api_key=s.google_api_key)

    if provider == "openai":
        from akrag.embeddings.openai_emb import OpenAIEmbedder
        return OpenAIEmbedder(model=model, api_key=s.openai_api_key)

    if provider == "sentence":
        from akrag.embeddings.sentence import SentenceEmbedder
        return SentenceEmbedder(model=model)

    if provider == "bedrock":
        from akrag.embeddings.bedrock import BedrockEmbedder
        return BedrockEmbedder(
            model=model,
            region=s.aws_region,
            access_key=s.aws_access_key_id,
            secret_key=s.aws_secret_access_key,
        )

    raise ValueError(f"Unknown embedding provider: {provider}")
