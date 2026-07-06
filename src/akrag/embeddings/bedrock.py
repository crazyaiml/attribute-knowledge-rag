from __future__ import annotations

import asyncio
import json
from akrag.embeddings.base import EmbeddingProvider

_DIM_MAP = {
    "amazon.titan-embed-text-v1": 1536,
    "amazon.titan-embed-text-v2:0": 1024,
    "cohere.embed-english-v3": 1024,
    "cohere.embed-multilingual-v3": 1024,
}


class BedrockEmbedder(EmbeddingProvider):
    """AWS Bedrock Titan / Cohere embeddings."""

    def __init__(
        self,
        model: str = "amazon.titan-embed-text-v2:0",
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
        self._model = model

    @property
    def dimension(self) -> int:
        return _DIM_MAP.get(self._model, 1024)

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_embed, texts)

    def _sync_embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            if "titan" in self._model:
                body = json.dumps({"inputText": text})
            else:
                body = json.dumps({"texts": [text], "input_type": "search_document"})

            resp = self._client.invoke_model(
                modelId=self._model,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            data = json.loads(resp["body"].read())

            if "titan" in self._model:
                results.append(data["embedding"])
            else:
                results.append(data["embeddings"][0])

        return results
