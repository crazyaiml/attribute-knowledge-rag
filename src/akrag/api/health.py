from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(prefix="/health", tags=["Health"])


class HealthResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"status": "ok"}})
    status: str = Field(..., description="Always `ok` when the service is running")


class ProviderStatus(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "llm": "claude",
            "llm_model": "claude-sonnet-4-6",
            "embedding": "sentence",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "search": "local",
            "indexes": ["attributes_v2026_06_26"],
        }
    })

    llm: str = Field(..., description="Active LLM provider name (`claude` | `openai` | `bedrock` | `ollama`)")
    llm_model: str = Field(..., description="Model identifier in use for the active LLM provider")
    embedding: str = Field(..., description="Active embedding provider (`gemini` | `openai` | `sentence` | `bedrock`)")
    embedding_model: str = Field(..., description="Embedding model identifier")
    embedding_dim: int = Field(..., description="Vector dimension produced by the embedding model")
    search: str = Field(..., description="Active search backend (`local` | `opensearch` | `faiss` | `chroma`)")
    indexes: list[str] = Field(..., description="Available attribute index names in the search backend")


@router.get(
    "",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns `{\"status\": \"ok\"}` as long as the service process is running. Use this for load-balancer health checks.",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/providers",
    response_model=ProviderStatus,
    summary="Active provider configuration",
    description=(
        "Returns the currently active LLM, embedding, and search backend together with "
        "their resolved model identifiers and the list of indexed attribute collections.\n\n"
        "Use this to verify that the service started with the expected configuration before "
        "running ingestion or queries."
    ),
)
async def providers() -> ProviderStatus:
    from akrag.config import get_settings
    from akrag.embeddings.factory import get_embedder
    from akrag.search.factory import get_search_backend

    s = get_settings()
    embedder = get_embedder()
    search = get_search_backend()
    indexes = await search.list_indexes()

    return ProviderStatus(
        llm=s.llm_provider,
        llm_model=s.resolved_llm_model,
        embedding=s.embedding_provider,
        embedding_model=embedder.model_name,
        embedding_dim=embedder.dimension,
        search=s.search_backend,
        indexes=indexes,
    )
