from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("EMBEDDING_PROVIDER", "sentence")
os.environ.setdefault("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
os.environ.setdefault("SEARCH_BACKEND", "local")


@pytest.fixture(scope="module")
def client():
    from akrag.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_providers(client):
    resp = client.get("/health/providers")
    assert resp.status_code == 200
    data = resp.json()
    assert data["search"] == "local"
    assert data["embedding"] == "sentence"


def test_validate_valid_csv(client):
    csv_content = (
        "attribute_id,type,business_name,technical_field,domain,definition,synonyms,operators\n"
        "labs.hba1c,numeric,HbA1c Level,hba1c_pct,labs,"
        "Glycated hemoglobin percentage.,A1C | glycated hemoglobin,> | >=\n"
    )
    resp = client.post(
        "/ingest/validate",
        files={"file": ("attrs.csv", csv_content.encode(), "text/csv")},
    )
    assert resp.status_code == 200
    assert resp.json()["valid"] is True


def test_validate_invalid_csv(client):
    csv_content = (
        "attribute_id,type,business_name,technical_field,domain,definition\n"
        "bad.attr,lab_value,Bad Attr,bad_field,labs,Invalid type.\n"
    )
    resp = client.post(
        "/ingest/validate",
        files={"file": ("bad.csv", csv_content.encode(), "text/csv")},
    )
    assert resp.status_code == 200
    assert resp.json()["valid"] is False


def test_upload_and_search(client):
    csv_content = (
        "attribute_id,type,business_name,technical_field,domain,definition,synonyms,operators\n"
        "labs.hba1c,numeric,HbA1c Level,hba1c_pct,labs,"
        "Glycated hemoglobin percentage — measures average blood glucose over 3 months.,"
        "A1C | glycated hemoglobin | hemoglobin A1c | blood sugar control,> | >=\n"
        "clinical.is_diabetic,boolean,Has Diabetes Diagnosis,is_diabetic_flag,clinical,"
        "Indicates an active Type 1 or Type 2 diabetes diagnosis.,"
        "diabetic | diabetes | T1D | T2D | type 1 | type 2,=\n"
    )
    resp = client.post(
        "/ingest/upload",
        files={"file": ("attrs.csv", csv_content.encode(), "text/csv")},
        data={"swap_alias": "false"},
    )
    assert resp.status_code == 200
    ingest_data = resp.json()
    assert ingest_data["indexed"] == 2

    resp = client.post(
        "/query/search",
        json={"phrases": ["diabetic patients with elevated HbA1c"], "top_k": 3},
    )
    assert resp.status_code == 200
    decisions = resp.json()
    assert len(decisions) == 1
    assert decisions[0]["phrase"] == "diabetic patients with elevated HbA1c"
    assert decisions[0]["outcome"] in ("exact", "near", "ambiguous", "none")


def test_list_indexes(client):
    resp = client.get("/ingest/indexes")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
