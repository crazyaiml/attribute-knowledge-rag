from __future__ import annotations

import typing

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from akrag.config import Settings

router = APIRouter(prefix="/settings", tags=["Settings"])


class SettingsOptions(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "llm_providers": ["claude", "openai", "bedrock", "ollama"],
            "embedding_providers": ["gemini", "openai", "sentence", "bedrock"],
            "search_backends": ["local", "opensearch", "faiss", "chroma"],
        }
    })

    llm_providers: list[str]
    embedding_providers: list[str]
    search_backends: list[str]


def _literal_values(field_name: str) -> list[str]:
    annotation = Settings.model_fields[field_name].annotation
    return list(typing.get_args(annotation))


@router.get(
    "/options",
    response_model=SettingsOptions,
    summary="List the provider choices each pluggable layer supports",
)
async def options() -> SettingsOptions:
    return SettingsOptions(
        llm_providers=_literal_values("llm_provider"),
        embedding_providers=_literal_values("embedding_provider"),
        search_backends=_literal_values("search_backend"),
    )


class SettingsPublic(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "llm_provider": "ollama",
            "llm_model": "",
            "resolved_llm_model": "llama3.2",
            "anthropic_api_key_set": False,
            "openai_api_key_set": False,
            "aws_configured": False,
            "ollama_base_url": "http://localhost:11434",
            "embedding_provider": "sentence",
            "embedding_model": "",
            "resolved_embedding_model": "all-MiniLM-L6-v2",
            "google_api_key_set": False,
            "search_backend": "local",
            "opensearch_host": "localhost",
            "opensearch_port": 9200,
            "chroma_host": "localhost",
            "chroma_port": 8000,
            "exact_threshold": 0.92,
            "near_threshold": 0.75,
            "rrf_k": 60,
            "top_k": 10,
        }
    })

    llm_provider: str
    llm_model: str
    resolved_llm_model: str
    anthropic_api_key_set: bool
    openai_api_key_set: bool
    aws_configured: bool
    ollama_base_url: str

    embedding_provider: str
    embedding_model: str
    resolved_embedding_model: str
    google_api_key_set: bool

    search_backend: str
    opensearch_host: str
    opensearch_port: int
    chroma_host: str
    chroma_port: int

    exact_threshold: float
    near_threshold: float
    rrf_k: int
    top_k: int


def _to_public(s: Settings) -> SettingsPublic:
    return SettingsPublic(
        llm_provider=s.llm_provider,
        llm_model=s.llm_model,
        resolved_llm_model=s.resolved_llm_model,
        anthropic_api_key_set=bool(s.anthropic_api_key),
        openai_api_key_set=bool(s.openai_api_key),
        aws_configured=bool(s.aws_access_key_id and s.aws_secret_access_key),
        ollama_base_url=s.ollama_base_url,
        embedding_provider=s.embedding_provider,
        embedding_model=s.embedding_model,
        resolved_embedding_model=s.resolved_embedding_model,
        google_api_key_set=bool(s.google_api_key),
        search_backend=s.search_backend,
        opensearch_host=s.opensearch_host,
        opensearch_port=s.opensearch_port,
        chroma_host=s.chroma_host,
        chroma_port=s.chroma_port,
        exact_threshold=s.exact_threshold,
        near_threshold=s.near_threshold,
        rrf_k=s.rrf_k,
        top_k=s.top_k,
    )


@router.get("", response_model=SettingsPublic, summary="Get the active provider configuration")
async def read() -> SettingsPublic:
    from akrag.config import get_settings

    return _to_public(get_settings())


_LLM_FIELDS = {
    "llm_provider", "llm_model", "anthropic_api_key", "openai_api_key",
    "aws_access_key_id", "aws_secret_access_key", "aws_region", "ollama_base_url",
}
_EMBEDDING_FIELDS = {"embedding_provider", "embedding_model", "google_api_key"}
_SEARCH_FIELDS = {
    "search_backend", "opensearch_host", "opensearch_port", "opensearch_user",
    "opensearch_pass", "opensearch_use_ssl", "chroma_host", "chroma_port",
}


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {"llm_provider": "ollama", "llm_model": "llama3.1"}
    })

    llm_provider: typing.Literal["claude", "openai", "bedrock", "ollama"] | None = None
    llm_model: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str | None = None
    ollama_base_url: str | None = None

    embedding_provider: typing.Literal["gemini", "openai", "sentence", "bedrock"] | None = None
    embedding_model: str | None = None
    google_api_key: str | None = None

    search_backend: typing.Literal["local", "opensearch", "faiss", "chroma"] | None = None
    opensearch_host: str | None = None
    opensearch_port: int | None = None
    opensearch_user: str | None = None
    opensearch_pass: str | None = None
    opensearch_use_ssl: bool | None = None
    chroma_host: str | None = None
    chroma_port: int | None = None


class SettingsUpdateResponse(BaseModel):
    settings: SettingsPublic
    index_reset: bool = Field(
        ..., description="True if the embedding provider or search backend changed, "
        "which clears the in-memory index — re-load your catalog after this."
    )


@router.put(
    "",
    response_model=SettingsUpdateResponse,
    summary="Switch LLM / embedding / search providers at runtime",
    description=(
        "Updates the active provider configuration without restarting the process. "
        "Changing `embedding_provider` or `search_backend` resets the in-memory search "
        "index (the embedding dimension or backend itself changes), so `index_reset` "
        "will be `true` — re-ingest your catalog afterward."
    ),
)
async def update(body: SettingsUpdate) -> SettingsUpdateResponse:
    from akrag.config import get_settings, update_settings
    from akrag.embeddings.factory import get_embedder
    from akrag.llm.factory import get_llm
    from akrag.search.factory import get_search_backend

    before = get_settings()
    overrides = body.model_dump(exclude_none=True)
    # The UI form resubmits every field, not just the one the user edited — only
    # fields whose value actually changed should count, or every save would reset
    # the in-memory index.
    changed = {f for f, v in overrides.items() if getattr(before, f, None) != v}

    new_settings = update_settings(**overrides)

    if changed & _LLM_FIELDS:
        get_llm.cache_clear()

    index_reset = bool(changed & (_EMBEDDING_FIELDS | _SEARCH_FIELDS))
    if index_reset:
        get_embedder.cache_clear()
        get_search_backend.cache_clear()

    return SettingsUpdateResponse(settings=_to_public(new_settings), index_reset=index_reset)
