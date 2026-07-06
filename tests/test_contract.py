from __future__ import annotations

from akrag.contract import coerce_to_document, validate_records, validate_row
from akrag.models import AttributeDocument


VALID_ROW = {
    "attribute_id": "labs.hba1c",
    "type": "numeric",
    "business_name": "HbA1c Level",
    "technical_field": "hba1c_pct",
    "domain": "labs",
    "definition": "Glycated hemoglobin percentage — measures average blood glucose over the preceding 3 months.",
    "synonyms": "A1C | glycated hemoglobin | hemoglobin A1c | blood sugar control",
    "operators": "> | >= | < | <= | =",
    "unit": "%",
    "example_values": "6.5 | 7.5 | 9.0",
    "pii": "false",
}


def test_valid_row_no_errors():
    errors = validate_row(VALID_ROW, 1)
    assert errors == []


def test_missing_required_field():
    row = {**VALID_ROW}
    del row["business_name"]
    errors = validate_row(row, 1)
    assert any("business_name" in str(e) for e in errors)


def test_invalid_type():
    row = {**VALID_ROW, "type": "lab_value"}
    errors = validate_row(row, 1)
    assert any("type" in str(e) for e in errors)


def test_validate_records_all_valid():
    result = validate_records([VALID_ROW])
    assert result.valid is True
    assert result.total == 1
    assert result.errors == []


def test_validate_records_with_error():
    bad = {**VALID_ROW, "type": "bad"}
    result = validate_records([VALID_ROW, bad])
    assert result.valid is False
    assert len(result.errors) >= 1


def test_coerce_to_document_synonyms():
    doc = coerce_to_document(VALID_ROW)
    assert isinstance(doc, AttributeDocument)
    assert "A1C" in doc.synonyms
    assert "glycated hemoglobin" in doc.synonyms
    assert "hemoglobin A1c" in doc.synonyms


def test_coerce_to_document_embedding_text():
    doc = coerce_to_document(VALID_ROW)
    assert doc.embedding_text is not None
    assert "HbA1c Level" in doc.embedding_text
    assert "hba1c_pct" in doc.embedding_text
    assert "labs" in doc.embedding_text


def test_coerce_to_document_governance_pii():
    doc = coerce_to_document(VALID_ROW)
    assert doc.governance.pii is False


def test_coerce_pii_true():
    row = {**VALID_ROW, "attribute_id": "patient.date_of_birth",
           "business_name": "Date of Birth", "pii": "true"}
    doc = coerce_to_document(row)
    assert doc.governance.pii is True
