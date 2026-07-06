from __future__ import annotations

import json
from akrag.llm.base import LLMProvider, Message


class BedrockLLM(LLMProvider):
    """AWS Bedrock — supports Anthropic Claude and Amazon Titan models."""

    def __init__(
        self,
        model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: str = "us-east-1",
        access_key: str = "",
        secret_key: str = "",
    ) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError("Install boto3: pip install akrag[bedrock]") from exc

        kwargs: dict = {"service_name": "bedrock-runtime", "region_name": region}
        if access_key and secret_key:
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key

        self._client = boto3.client(**kwargs)
        self.model = model

    async def complete(self, messages: list[Message], max_tokens: int = 1024) -> str:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_complete, messages, max_tokens)

    def _sync_complete(self, messages: list[Message], max_tokens: int) -> str:
        if "anthropic" in self.model:
            return self._claude_on_bedrock(messages, max_tokens)
        if "titan" in self.model:
            return self._titan(messages, max_tokens)
        raise ValueError(f"Unsupported Bedrock model: {self.model}")

    def _claude_on_bedrock(self, messages: list[Message], max_tokens: int) -> str:
        system_text = ""
        chat: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_text = m.content
            else:
                chat.append({"role": m.role, "content": m.content})

        body: dict = {"anthropic_version": "bedrock-2023-05-31",
                      "max_tokens": max_tokens, "messages": chat}
        if system_text:
            body["system"] = system_text

        resp = self._client.invoke_model(
            modelId=self.model,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        data = json.loads(resp["body"].read())
        return data["content"][0]["text"]

    def _titan(self, messages: list[Message], max_tokens: int) -> str:
        prompt = "\n".join(
            f"{'Human' if m.role == 'user' else 'Assistant'}: {m.content}"
            for m in messages if m.role != "system"
        ) + "\nAssistant:"

        body = {"inputText": prompt,
                "textGenerationConfig": {"maxTokenCount": max_tokens, "temperature": 0.2}}
        resp = self._client.invoke_model(
            modelId=self.model,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        data = json.loads(resp["body"].read())
        return data["results"][0]["outputText"]
