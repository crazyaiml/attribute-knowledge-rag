from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from akrag.io import read_file, read_ndjson
from akrag.models import IngestResult, ValidationResult

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


class SwapAliasResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {"alias": "attributes_current", "index": "attributes_v2026_06_26", "status": "swapped"}
    })
    alias: str = Field(..., description="The alias name that was updated")
    index: str = Field(..., description="The versioned index now pointed to by the alias")
    status: str = Field(..., description="Always `swapped` on success")


def _get_orchestrator():
    from akrag.embeddings.factory import get_embedder
    from akrag.llm.factory import get_llm
    from akrag.orchestrator import Orchestrator
    from akrag.search.factory import get_search_backend

    return Orchestrator(get_embedder(), get_search_backend(), get_llm())


@router.post(
    "/validate",
    response_model=ValidationResult,
    summary="Validate an attribute file",
    description=(
        "Upload a CSV or XLSX attribute file and check every row against the AK-RAG contract schema.\n\n"
        "**Required columns:** `attribute_id`, `type`, `business_name`, `technical_field`, `domain`, `definition`\n\n"
        "**Optional columns:** `synonyms` (pipe-separated), `operators` (pipe-separated), `unit`, "
        "`example_values` (pipe-separated), `allowed_values` (pipe-separated), `pii` (true/false)\n\n"
        "Returns `valid: true` with zero errors if the file is ready to ingest. "
        "Returns `valid: false` with a per-row error list otherwise — fix the errors before calling `/ingest/upload`."
    ),
    responses={
        200: {"description": "Validation completed (inspect the `valid` field — HTTP 200 even when invalid)"},
    },
)
async def validate(
    file: UploadFile = File(..., description="CSV (`.csv`) or Excel (`.xlsx`) attribute file"),
) -> ValidationResult:
    from akrag.contract import validate_records

    suffix = Path(file.filename or "upload.csv").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    records = read_file(tmp_path)
    return validate_records(records)


@router.post(
    "/upload",
    response_model=IngestResult,
    summary="Upload and index an attribute file",
    description=(
        "Upload a CSV or XLSX file, validate it, embed each attribute document, and index it "
        "in the configured search backend.\n\n"
        "**Pipeline:**\n"
        "1. Parse CSV/XLSX rows\n"
        "2. Validate against contract schema (all rows must pass)\n"
        "3. Compose embedding text per attribute\n"
        "4. Embed in batches of 64 using the configured embedding provider\n"
        "5. Create a versioned index (`attributes_vYYYY_MM_DD`)\n"
        "6. Bulk-index all documents\n"
        "7. Optionally swap the `attributes_current` alias to the new index\n\n"
        "Set `swap_alias=false` to stage the index without making it live — "
        "then call `/ingest/swap-alias` after your smoke tests pass."
    ),
    responses={
        200: {"description": "Ingestion completed. Check `failed` count and `errors` list."},
        422: {"description": "Multipart form parse error"},
    },
)
async def upload(
    file: UploadFile = File(..., description="CSV (`.csv`) or Excel (`.xlsx`) attribute file"),
    index_name: str = Form(
        default="",
        description="Override the auto-generated versioned index name (e.g. `attributes_v2026_06_26`). Leave blank to auto-generate.",
    ),
    swap_alias: bool = Form(
        default=True,
        description="If `true`, atomically point `attributes_current` alias to the new index after indexing completes.",
    ),
) -> IngestResult:
    suffix = Path(file.filename or "upload.csv").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    records = read_file(tmp_path)
    orc = _get_orchestrator()
    return await orc.ingest(records, index_name=index_name or None, swap_alias=swap_alias)


@router.post(
    "/from-ndjson",
    response_model=IngestResult,
    summary="Index a pre-built NDJSON attribute file",
    description=(
        "Upload an NDJSON file where each line is a complete `AttributeDocument` JSON object "
        "(one document per line, no wrapping array).\n\n"
        "Use this when you have already generated NDJSON via the CLI (`akrag to-ndjson`) "
        "or a custom pipeline and want to load it directly without re-parsing a spreadsheet.\n\n"
        "Documents must include a `vector` field if the search backend is OpenSearch, FAISS, or ChromaDB. "
        "For the `local` backend, vectors are regenerated automatically if absent."
    ),
    responses={
        200: {"description": "Ingestion completed"},
        422: {"description": "Multipart form parse error"},
    },
)
async def from_ndjson(
    file: UploadFile = File(..., description="NDJSON file (`.ndjson` or `.jsonl`) — one `AttributeDocument` per line"),
    index_name: str = Form(default="", description="Override auto-generated index name"),
    swap_alias: bool = Form(default=True, description="Swap `attributes_current` alias after indexing"),
) -> IngestResult:
    with tempfile.NamedTemporaryFile(suffix=".ndjson", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    records = read_ndjson(tmp_path)
    orc = _get_orchestrator()
    return await orc.ingest(records, index_name=index_name or None, swap_alias=swap_alias)


@router.get(
    "/indexes",
    response_model=list[str],
    summary="List available attribute indexes",
    description=(
        "Returns the names of all attribute indexes present in the configured search backend.\n\n"
        "For OpenSearch this includes versioned indexes (`attributes_v*`) and the current alias. "
        "For local/FAISS/ChromaDB backends this lists all in-memory or persisted collections."
    ),
    responses={200: {"content": {"application/json": {"example": ["attributes_v2026_06_26", "attributes_v2026_05_01"]}}}},
)
async def list_indexes() -> list[str]:
    from akrag.search.factory import get_search_backend

    return await get_search_backend().list_indexes()


@router.post(
    "/swap-alias",
    response_model=SwapAliasResponse,
    summary="Promote a versioned index to the current alias",
    description=(
        "Atomically redirect the `attributes_current` alias (or a custom alias) "
        "to a specific versioned index.\n\n"
        "This is a zero-downtime promotion: queries in-flight against the old alias complete "
        "normally before the pointer changes.\n\n"
        "**Rollback:** call this endpoint again with the previous index name to revert.\n\n"
        "> **Note:** This operation is a no-op for the `local`, `faiss`, and `chroma` backends "
        "which do not use OpenSearch-style aliases."
    ),
    responses={
        200: {"description": "Alias swapped successfully"},
        404: {"description": "Index not found"},
    },
)
async def swap_alias(
    index_name: str = Form(..., description="Versioned index to promote, e.g. `attributes_v2026_06_26`"),
    alias: str = Form(default="attributes_current", description="Alias name to update"),
) -> SwapAliasResponse:
    from akrag.search.factory import get_search_backend

    await get_search_backend().swap_alias(alias, index_name)
    return SwapAliasResponse(alias=alias, index=index_name, status="swapped")
