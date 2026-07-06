from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class GovernanceMetadata(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "pii": False,
            "sensitive": False,
            "allowed_channels": ["care_management", "population_health"],
            "eligible_domains": ["clinical", "labs"],
        }
    })

    pii: bool = Field(False, description="Whether this attribute contains PII")
    sensitive: bool = Field(False, description="Whether this attribute is marked sensitive")
    allowed_channels: list[str] = Field(
        default_factory=list,
        description="Channels permitted to use this attribute (empty = all channels allowed)",
    )
    eligible_domains: list[str] = Field(
        default_factory=list,
        description="Business domains eligible to query this attribute",
    )


class AttributeDocument(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "attribute_id": "labs.hba1c",
            "type": "numeric",
            "business_name": "HbA1c Level",
            "technical_field": "hba1c_pct",
            "domain": "labs",
            "definition": "Glycated hemoglobin percentage — measures average blood glucose over the preceding 3 months.",
            "synonyms": ["A1C", "glycated hemoglobin", "hemoglobin A1c", "blood sugar control"],
            "operators": [">", ">=", "<", "<=", "="],
            "unit": "%",
            "example_values": [6.5, 7.5, 9.0],
            "allowed_values": [],
            "governance": {
                "pii": False,
                "allowed_channels": ["care_management", "population_health"],
            },
            "embedding_text": "numeric | HbA1c Level | hba1c_pct | A1C | glycated hemoglobin | hemoglobin A1c | % | labs",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_version": "2026-06-26",
        }
    })

    attribute_id: str = Field(
        ..., description="Globally unique attribute identifier, e.g. `labs.hba1c`"
    )
    type: Literal["numeric", "categorical", "boolean", "date", "text"] = Field(
        ..., description="Data type of the attribute value"
    )
    business_name: str = Field(..., description="Human-readable business name shown to end-users")
    technical_field: str = Field(..., description="Physical column / field name in the data store")
    domain: str = Field(..., description="Business domain that owns this attribute")
    definition: str = Field(..., description="Plain-language definition of what this attribute measures")
    synonyms: list[str] = Field(default_factory=list, description="Alternative phrases that map to this attribute")
    operators: list[str] = Field(default_factory=list, description="Valid comparison operators for this attribute")
    unit: str | None = Field(None, description="Unit of measure, e.g. `%`, `mmHg`, `mg/dL`")
    example_values: list[Any] = Field(default_factory=list, description="Representative values for documentation")
    allowed_values: list[str] = Field(
        default_factory=list,
        description="For categorical attributes, the complete enumeration of valid values",
    )
    ranges: dict[str, Any] = Field(default_factory=dict, description="Optional min/max range constraints")
    governance: GovernanceMetadata = Field(
        default_factory=GovernanceMetadata, description="Governance and access control metadata"
    )
    embedding_text: str | None = Field(
        None, description="The text string that was embedded (auto-composed at ingest time)"
    )
    embedding_model: str | None = Field(None, description="Embedding model used to produce the vector")
    embedding_version: str | None = Field(None, description="Date the embedding was generated (YYYY-MM-DD)")
    vector: list[float] | None = Field(None, description="Dense embedding vector (excluded from API responses)")

    def compose_embedding_text(self) -> str:
        parts = [
            self.type,
            self.business_name,
            self.technical_field,
            " | ".join(self.synonyms) if self.synonyms else "",
            self.definition,
            " | ".join(self.allowed_values) if self.allowed_values else "",
            " | ".join(str(v) for v in self.example_values) if self.example_values else "",
            self.unit or "",
            self.domain,
        ]
        return " | ".join(p for p in parts if p)


class SearchResult(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "attribute": {
                "attribute_id": "labs.hba1c",
                "type": "numeric",
                "business_name": "HbA1c Level",
                "technical_field": "hba1c_pct",
                "domain": "labs",
                "definition": "Glycated hemoglobin percentage — measures average blood glucose over the preceding 3 months.",
            },
            "bm25_score": 4.21,
            "vector_score": 0.91,
            "rrf_score": 0.031,
            "bm25_rank": 0,
            "vector_rank": 0,
        }
    })

    attribute: AttributeDocument = Field(..., description="The matched attribute document")
    bm25_score: float = Field(0.0, description="Raw BM25 lexical relevance score")
    vector_score: float = Field(0.0, description="Cosine similarity score from the vector index")
    rrf_score: float = Field(0.0, description="Reciprocal Rank Fusion score (combines BM25 + vector ranks)")
    bm25_rank: int | None = Field(None, description="Rank position in the BM25 result list (0-indexed)")
    vector_rank: int | None = Field(None, description="Rank position in the vector result list (0-indexed)")


class DecisionOutcome(str, Enum):
    EXACT = "exact"
    NEAR = "near"
    AMBIGUOUS = "ambiguous"
    NONE = "none"


class PhraseDecision(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "phrase": "diabetic patients",
            "outcome": "exact",
            "selected": {
                "attribute_id": "clinical.is_diabetic",
                "business_name": "Has Diabetes Diagnosis",
                "type": "boolean",
                "technical_field": "is_diabetic_flag",
                "domain": "clinical",
                "definition": "Indicates an active Type 1 or Type 2 diabetes diagnosis on the patient's problem list.",
            },
            "options": [],
            "clarification_message": None,
        }
    })

    phrase: str = Field(..., description="The input phrase that was searched")
    outcome: DecisionOutcome = Field(
        ...,
        description=(
            "Classification of the match quality:\n"
            "- `exact` — high-confidence single match, use as-is\n"
            "- `near` — likely match but confirmation recommended\n"
            "- `ambiguous` — multiple equally-plausible candidates, user must choose\n"
            "- `none` — no supported attribute found"
        ),
    )
    selected: AttributeDocument | None = Field(
        None, description="The chosen attribute (populated for `exact` outcomes only)"
    )
    options: list[SearchResult] = Field(
        default_factory=list,
        description="Ranked candidate attributes (populated for `near`, `ambiguous`, and `none` outcomes)",
    )
    clarification_message: str | None = Field(
        None, description="Human-readable message to display to the user when clarification is needed"
    )


class DSLFilter(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "attribute_id": "labs.hba1c",
            "business_name": "HbA1c Level",
            "technical_field": "hba1c_pct",
            "operator": ">=",
            "value": 7.5,
            "source_phrase": "elevated HbA1c",
        }
    })

    attribute_id: str = Field(..., description="Resolved attribute identifier")
    business_name: str = Field(..., description="Human-readable business name of the resolved attribute")
    technical_field: str = Field(..., description="Physical column name to use in query generation")
    operator: str | None = Field(None, description="Comparison operator supplied by the user or inferred")
    value: Any = Field(None, description="Filter value supplied by the user or inferred")
    source_phrase: str = Field(
        ..., description="The original natural-language phrase this filter was derived from"
    )


class QueryResult(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "input": "diabetic patients over 65 with elevated HbA1c and blood pressure issues",
            "phrases": ["diabetic patients", "over 65", "elevated HbA1c", "blood pressure issues"],
            "filters": [
                {
                    "attribute_id": "clinical.is_diabetic",
                    "business_name": "Has Diabetes Diagnosis",
                    "technical_field": "is_diabetic_flag",
                    "operator": None,
                    "value": None,
                    "source_phrase": "diabetic patients",
                },
                {
                    "attribute_id": "patient.age",
                    "business_name": "Patient Age",
                    "technical_field": "patient_age_years",
                    "operator": ">",
                    "value": 65,
                    "source_phrase": "over 65",
                },
            ],
            "unresolved": [
                {
                    "phrase": "elevated HbA1c",
                    "outcome": "near",
                    "selected": None,
                    "options": [
                        {
                            "attribute": {
                                "attribute_id": "labs.hba1c",
                                "business_name": "HbA1c Level",
                                "type": "numeric",
                                "technical_field": "hba1c_pct",
                                "domain": "labs",
                                "definition": "Glycated hemoglobin percentage — measures average blood glucose over the preceding 3 months.",
                            },
                            "rrf_score": 0.029,
                            "bm25_score": 3.1,
                            "vector_score": 0.88,
                        }
                    ],
                    "clarification_message": "Found a likely match for \"elevated HbA1c\":\n  1. HbA1c Level (labs.hba1c): Glycated hemoglobin percentage\nWhat threshold should be applied (e.g. >= 7.5%)?",
                },
                {
                    "phrase": "blood pressure issues",
                    "outcome": "ambiguous",
                    "selected": None,
                    "options": [
                        {
                            "attribute": {
                                "attribute_id": "vitals.systolic_bp",
                                "business_name": "Systolic Blood Pressure",
                                "type": "numeric",
                                "technical_field": "systolic_bp_mmhg",
                                "domain": "vitals",
                                "definition": "Peak arterial pressure during a heartbeat.",
                            },
                            "rrf_score": 0.027,
                            "bm25_score": 2.4,
                            "vector_score": 0.82,
                        },
                        {
                            "attribute": {
                                "attribute_id": "vitals.diastolic_bp",
                                "business_name": "Diastolic Blood Pressure",
                                "type": "numeric",
                                "technical_field": "diastolic_bp_mmhg",
                                "domain": "vitals",
                                "definition": "Arterial pressure between heartbeats.",
                            },
                            "rrf_score": 0.026,
                            "bm25_score": 2.2,
                            "vector_score": 0.81,
                        },
                        {
                            "attribute": {
                                "attribute_id": "clinical.is_hypertensive",
                                "business_name": "Has Hypertension",
                                "type": "boolean",
                                "technical_field": "is_hypertensive_flag",
                                "domain": "clinical",
                                "definition": "Indicates an active hypertension diagnosis on the patient's problem list.",
                            },
                            "rrf_score": 0.025,
                            "bm25_score": 2.1,
                            "vector_score": 0.79,
                        },
                    ],
                    "clarification_message": "No exact match found for \"blood pressure issues\". Did you mean:\n  1. Systolic Blood Pressure (vitals.systolic_bp)\n  2. Diastolic Blood Pressure (vitals.diastolic_bp)\n  3. Has Hypertension (clinical.is_hypertensive)\n\nReply with the number or provide an alternative description.",
                },
            ],
            "dsl": {
                "version": "1.0",
                "type": "attribute_filter",
                "filters": [
                    {
                        "attribute_id": "clinical.is_diabetic",
                        "business_name": "Has Diabetes Diagnosis",
                        "technical_field": "is_diabetic_flag",
                        "operator": None,
                        "value": None,
                        "source_phrase": "diabetic patients",
                    },
                    {
                        "attribute_id": "patient.age",
                        "business_name": "Patient Age",
                        "technical_field": "patient_age_years",
                        "operator": ">",
                        "value": 65,
                        "source_phrase": "over 65",
                    },
                ],
            },
        }
    })

    input: str = Field(..., description="The original natural-language query from the user")
    phrases: list[str] = Field(
        ..., description="Individual attribute phrases extracted from the input by the LLM"
    )
    filters: list[DSLFilter] = Field(
        default_factory=list,
        description="Resolved filters ready for DSL generation (exact-match phrases only)",
    )
    unresolved: list[PhraseDecision] = Field(
        default_factory=list,
        description="Phrases that require user clarification (near, ambiguous, or none outcomes)",
    )
    dsl: dict[str, Any] | None = Field(
        None,
        description="Structured segment definition assembled from resolved filters (null if no exact matches yet)",
    )


class IngestResult(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "total": 18,
            "indexed": 18,
            "failed": 0,
            "errors": [],
            "index_name": "attributes_v2026_06_26",
        }
    })

    total: int = Field(..., description="Total number of attribute records in the source file")
    indexed: int = Field(..., description="Number of records successfully embedded and indexed")
    failed: int = Field(..., description="Number of records that failed validation or indexing")
    errors: list[str] = Field(default_factory=list, description="Error messages for failed records")
    index_name: str | None = Field(None, description="Name of the index that was created or updated")


class ValidationResult(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "valid": False,
            "total": 3,
            "errors": [
                {
                    "row": 2,
                    "field": "type",
                    "error": "type must be one of ['boolean', 'categorical', 'date', 'numeric', 'text'], got 'lab_value'",
                }
            ],
        }
    })

    valid: bool = Field(..., description="`true` if every record passed schema validation")
    total: int = Field(..., description="Total number of records checked")
    errors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-row validation errors. Each entry has `row`, `field`, and `error` keys.",
    )


# ── Request bodies ────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "phrases": ["elevated HbA1c", "blood pressure issues"],
            "top_k": 5,
            "channel": "care_management",
        }
    })

    phrases: list[str] = Field(
        ..., description="One or more attribute phrases to search for independently"
    )
    top_k: int = Field(10, ge=1, le=50, description="Maximum candidates to return per phrase")
    channel: str | None = Field(
        None,
        description="Filter to attributes permitted in this channel (e.g. `care_management`, `population_health`)",
    )


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "input": "diabetic patients over 65 with elevated HbA1c and blood pressure issues",
            "channel": "population_health",
            "top_k": 5,
        }
    })

    input: str = Field(
        ...,
        description="Full natural-language query from the user. The LLM will parse this into individual attribute phrases.",
    )
    channel: str | None = Field(
        None, description="Restrict results to attributes permitted in this channel"
    )
    top_k: int = Field(10, ge=1, le=50, description="Maximum candidates to retrieve per phrase")


class ClarifyRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "session_id": "sess_abc123",
            "phrase": "blood pressure issues",
            "chosen_attribute_id": "vitals.systolic_bp",
        }
    })

    session_id: str = Field(
        ..., description="Client-managed session identifier for conversation tracking"
    )
    phrase: str = Field(..., description="The ambiguous phrase the user is resolving")
    chosen_attribute_id: str = Field(
        ...,
        description="The `attribute_id` of the attribute the user selected from the options list",
    )
