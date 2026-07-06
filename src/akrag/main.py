from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from akrag.api.health import router as health_router
from akrag.api.ingest import router as ingest_router
from akrag.api.query import router as query_router
from akrag.api.settings import router as settings_router

_DESCRIPTION = """
**AK-RAG** transforms enterprise attribute metadata into an AI-searchable knowledge layer.

Instead of embedding free-form documents, AK-RAG indexes business attributes as individual
knowledge objects. Each attribute carries its business name, technical field, synonyms,
allowed values, governance rules, and a pre-computed embedding.

The system translates natural language into approved enterprise attributes with:
- Lower hallucination risk (generation is constrained to retrieved, governed attributes)
- Transparent clarification paths (users always see which attribute was chosen and why)
- Zero invented fields (the system asks rather than guesses)

---

## Pluggable Providers

| Layer | Options | Set via |
|---|---|---|
| **LLM** | `claude` · `openai` · `bedrock` · `ollama` | `LLM_PROVIDER` env var |
| **Embedding** | `sentence` · `openai` · `gemini` · `bedrock` | `EMBEDDING_PROVIDER` env var |
| **Search** | `local` · `opensearch` · `faiss` · `chroma` | `SEARCH_BACKEND` env var |

Default (no API key needed): `EMBEDDING_PROVIDER=sentence`, `SEARCH_BACKEND=local`

---

## Typical Integration Flow

```
1. POST /ingest/validate      Validate your attribute spreadsheet
2. POST /ingest/upload        Embed + index attributes (creates versioned index)
3. POST /ingest/swap-alias    Promote new index to attributes_current (production)

4. POST /query/evaluate       Translate user query → phrases → hybrid search → DSL
   └─ for unresolved phrases:
5. POST /query/clarify        User picks from options → resolved attribute
```

---

## Decision Policy

Each searched phrase is classified by the system:

| Outcome | Threshold | Action |
|---|---|---|
| `exact` | RRF confidence ≥ 0.92 | Use attribute directly |
| `near` | RRF confidence ≥ 0.75 | Show top option, ask for confirmation |
| `ambiguous` | Multiple near-equal candidates | Present ranked options, require user selection |
| `none` | No candidates above threshold | Ask user to rephrase or use a broader term |

Thresholds are tunable via `EXACT_THRESHOLD` and `NEAR_THRESHOLD` environment variables.
"""

_TAGS = [
    {
        "name": "Health",
        "description": "Liveness probe and active provider configuration.",
    },
    {
        "name": "Ingestion",
        "description": (
            "Upload, validate, embed, and index attribute files. "
            "Supports CSV, XLSX, and pre-built NDJSON. "
            "Versioned indexes with atomic alias promotion for zero-downtime releases."
        ),
    },
    {
        "name": "Query",
        "description": (
            "Natural-language to attribute translation. "
            "Hybrid retrieval (BM25 + vector + RRF) with exact / near / ambiguous / none "
            "decision policy and LLM-generated clarification messages."
        ),
    },
    {
        "name": "Settings",
        "description": (
            "Read and switch the active LLM / embedding / search backend at runtime, "
            "without restarting the process."
        ),
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    import logging
    logger = logging.getLogger("akrag")
    try:
        from akrag.embeddings.factory import get_embedder
        from akrag.search.factory import get_search_backend
        get_embedder()
        get_search_backend()
        logger.info("Providers warmed up successfully")
    except Exception as exc:
        logger.warning("Provider warm-up skipped: %s", exc)
    yield


app = FastAPI(
    title="Attribute Knowledge RAG (AK-RAG)",
    description=_DESCRIPTION,
    version="0.1.0",
    openapi_tags=_TAGS,
    contact={"name": "superml.ai", "url": "https://superml.ai"},
    license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(settings_router)


if __name__ == "__main__":
    import uvicorn
    from akrag.config import get_settings

    s = get_settings()
    uvicorn.run("akrag.main:app", host=s.api_host, port=s.api_port,
                log_level=s.log_level, reload=True)
