from __future__ import annotations

from typing import Any
from pydantic import ValidationError

from akrag.models import AttributeDocument, ValidationResult


REQUIRED_FIELDS = {
    "attribute_id",
    "type",
    "business_name",
    "technical_field",
    "domain",
    "definition",
}

VALID_TYPES = {"numeric", "categorical", "boolean", "date", "text"}


def validate_row(row: dict[str, Any], row_number: int) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    missing = REQUIRED_FIELDS - set(k for k, v in row.items() if v not in (None, ""))
    if missing:
        errors.append({"row": row_number, "field": list(missing), "error": "required field missing"})
        return errors

    if row.get("type") not in VALID_TYPES:
        errors.append({
            "row": row_number,
            "field": "type",
            "error": f"type must be one of {sorted(VALID_TYPES)}, got {row.get('type')!r}",
        })

    try:
        _coerce_row(row)
        AttributeDocument(**_coerce_row(row))
    except (ValidationError, ValueError) as exc:
        errors.append({"row": row_number, "field": "*", "error": str(exc)})

    return errors


def _coerce_row(raw: dict[str, Any]) -> dict[str, Any]:
    row = dict(raw)

    for list_field in ("synonyms", "operators", "allowed_values", "example_values"):
        val = row.get(list_field, "")
        if isinstance(val, str):
            row[list_field] = [v.strip() for v in val.split("|") if v.strip()] if val.strip() else []
        elif val is None:
            row[list_field] = []

    gov_raw = row.pop("governance", {}) or {}
    if isinstance(gov_raw, str):
        import json
        gov_raw = json.loads(gov_raw) if gov_raw.strip() else {}
    if isinstance(row.get("pii"), (str, bool, int)):
        pii = row.pop("pii", False)
        gov_raw.setdefault("pii", str(pii).lower() in ("true", "1", "yes"))
    if isinstance(row.get("allowed_channels"), list):
        gov_raw.setdefault("allowed_channels", row.pop("allowed_channels"))
    row["governance"] = gov_raw

    for drop in ("vector", "embedding_text", "embedding_model", "embedding_version"):
        row.pop(drop, None)

    return row


def validate_records(records: list[dict[str, Any]]) -> ValidationResult:
    all_errors: list[dict[str, Any]] = []
    for i, rec in enumerate(records, start=1):
        all_errors.extend(validate_row(rec, i))
    return ValidationResult(
        valid=len(all_errors) == 0,
        total=len(records),
        errors=all_errors,
    )


def coerce_to_document(raw: dict[str, Any]) -> AttributeDocument:
    coerced = _coerce_row(raw)
    doc = AttributeDocument(**coerced)
    doc.embedding_text = doc.compose_embedding_text()
    return doc
