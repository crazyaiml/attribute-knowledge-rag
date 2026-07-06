from __future__ import annotations

from functools import lru_cache
from akrag.llm.base import LLMProvider


@lru_cache(maxsize=1)
def get_llm() -> LLMProvider:
    from akrag.config import get_settings
    s = get_settings()
    provider = s.llm_provider
    model = s.resolved_llm_model

    if provider == "claude":
        from akrag.llm.claude import ClaudeLLM
        return ClaudeLLM(model=model, api_key=s.anthropic_api_key)

    if provider == "openai":
        from akrag.llm.openai_llm import OpenAILLM
        return OpenAILLM(model=model, api_key=s.openai_api_key)

    if provider == "bedrock":
        from akrag.llm.bedrock import BedrockLLM
        return BedrockLLM(
            model=model,
            region=s.aws_region,
            access_key=s.aws_access_key_id,
            secret_key=s.aws_secret_access_key,
        )

    if provider == "ollama":
        from akrag.llm.ollama import OllamaLLM
        return OllamaLLM(model=model, base_url=s.ollama_base_url)

    raise ValueError(f"Unknown LLM provider: {provider}")
