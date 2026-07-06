from __future__ import annotations

from akrag.llm.base import LLMProvider, Message


class ClaudeLLM(LLMProvider):
    """Anthropic Claude via the anthropic SDK."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str = "") -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Install anthropic: pip install akrag[claude]") from exc
        self._client = anthropic.AsyncAnthropic(api_key=api_key or None)
        self.model = model

    async def complete(self, messages: list[Message], max_tokens: int = 1024) -> str:
        import anthropic

        system_text = ""
        chat: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_text = m.content
            else:
                chat.append({"role": m.role, "content": m.content})

        kwargs: dict = {"model": self.model, "max_tokens": max_tokens, "messages": chat}
        if system_text:
            kwargs["system"] = system_text

        response = await self._client.messages.create(**kwargs)
        return response.content[0].text
