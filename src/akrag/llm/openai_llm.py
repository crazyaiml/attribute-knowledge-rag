from __future__ import annotations

from akrag.llm.base import LLMProvider, Message


class OpenAILLM(LLMProvider):
    """OpenAI Chat Completions."""

    def __init__(self, model: str = "gpt-4o", api_key: str = "") -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("Install openai: pip install akrag[openai]") from exc
        self._client = AsyncOpenAI(api_key=api_key or None)
        self.model = model

    async def complete(self, messages: list[Message], max_tokens: int = 1024) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[m.model_dump() for m in messages],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
