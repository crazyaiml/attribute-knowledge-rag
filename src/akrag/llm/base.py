from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal
from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMProvider(ABC):
    """Abstract base for all LLM backends."""

    @abstractmethod
    async def complete(self, messages: list[Message], max_tokens: int = 1024) -> str:
        """Send a conversation and return the assistant reply."""

    async def parse_phrases(self, user_input: str) -> list[str]:
        """Break a natural-language query into individual attribute phrases."""
        system = (
            "You are an enterprise data attribute assistant. "
            "Your only job is to extract individual attribute phrases from a user query. "
            "Return a JSON array of strings — one string per distinct attribute concept. "
            "No explanation, no prose, only the JSON array."
        )
        user_msg = f"Query: {user_input}"
        raw = await self.complete([Message(role="system", content=system),
                                   Message(role="user", content=user_msg)])
        import json, re
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        return [user_input]

    async def generate_clarification(
        self, phrase: str, options: list[dict], outcome: str
    ) -> str:
        """Generate a human-readable clarification prompt for a near/ambiguous match."""
        system = (
            "You are an enterprise data assistant helping users pick the right business attribute. "
            "Be concise and professional."
        )
        options_text = "\n".join(
            f"  {i+1}. {o['business_name']} ({o['attribute_id']}): {o.get('definition','')}"
            for i, o in enumerate(options)
        )
        user_msg = (
            f"The user said: \"{phrase}\"\n"
            f"Match outcome: {outcome}\n"
            f"Available options:\n{options_text}\n\n"
            "Ask the user to choose which attribute they meant. Be concise."
        )
        return await self.complete([Message(role="system", content=system),
                                    Message(role="user", content=user_msg)])

    async def generate_dsl(
        self, filters: list[dict], unresolved: list[str]
    ) -> str:
        """Generate a human-readable segment description from resolved filters."""
        system = (
            "You are an enterprise data assistant. "
            "Summarize the following attribute filters as a clear segment definition. "
            "Use natural language. Do not invent attributes not listed."
        )
        filter_text = "\n".join(
            f"  - {f['business_name']} {f.get('operator','')} {f.get('value','')}"
            for f in filters
        )
        unresolved_text = (
            f"Unresolved phrases (excluded): {', '.join(unresolved)}" if unresolved else ""
        )
        user_msg = f"Filters:\n{filter_text}\n\n{unresolved_text}"
        return await self.complete([Message(role="system", content=system),
                                    Message(role="user", content=user_msg)])
