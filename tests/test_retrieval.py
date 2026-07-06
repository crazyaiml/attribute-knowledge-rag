from __future__ import annotations

import asyncio
import pytest

from akrag.contract import coerce_to_document
from akrag.decision import classify
from akrag.models import AttributeDocument, DecisionOutcome
from akrag.search.base import rrf_fusion
from akrag.search.local import LocalSearchBackend


# ── Fixtures ──────────────────────────────────────────────────────────────────

ROWS = [
    {
        "attribute_id": "labs.hba1c",
        "type": "numeric",
        "business_name": "HbA1c Level",
        "technical_field": "hba1c_pct",
        "domain": "labs",
        "definition": "Glycated hemoglobin percentage — measures average blood glucose over the preceding 3 months.",
        "synonyms": "A1C | glycated hemoglobin | hemoglobin A1c | blood sugar control",
        "operators": "> | >=",
    },
    {
        "attribute_id": "vitals.systolic_bp",
        "type": "numeric",
        "business_name": "Systolic Blood Pressure",
        "technical_field": "systolic_bp_mmhg",
        "domain": "vitals",
        "definition": "Peak arterial pressure during a heartbeat.",
        "synonyms": "upper blood pressure | top number | systolic pressure | SBP",
        "operators": "> | >=",
    },
    {
        "attribute_id": "vitals.diastolic_bp",
        "type": "numeric",
        "business_name": "Diastolic Blood Pressure",
        "technical_field": "diastolic_bp_mmhg",
        "domain": "vitals",
        "definition": "Arterial pressure between heartbeats.",
        "synonyms": "lower blood pressure | bottom number | diastolic pressure | DBP",
        "operators": "> | >=",
    },
    {
        "attribute_id": "clinical.is_diabetic",
        "type": "boolean",
        "business_name": "Has Diabetes Diagnosis",
        "technical_field": "is_diabetic_flag",
        "domain": "clinical",
        "definition": "Indicates an active Type 1 or Type 2 diabetes diagnosis on the patient's problem list.",
        "synonyms": "diabetic | diabetes | T1D | T2D | type 1 | type 2 | DM",
        "operators": "=",
    },
]


def _make_docs_with_fake_vectors(dim: int = 8) -> list[AttributeDocument]:
    import hashlib
    import math
    import random

    docs = []
    for row in ROWS:
        doc = coerce_to_document(row)
        seed = int(hashlib.md5(doc.embedding_text.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        vec = [rng.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x**2 for x in vec))
        doc.vector = [x / norm for x in vec]
        docs.append(doc)
    return docs


@pytest.fixture
def docs():
    return _make_docs_with_fake_vectors()


@pytest.fixture
def backend(docs):
    b = LocalSearchBackend()
    asyncio.get_event_loop().run_until_complete(b.index(docs))
    return b, docs


# ── RRF fusion ────────────────────────────────────────────────────────────────

def test_rrf_fusion_combines_results(docs):
    bm25 = [(docs[0], 3.0), (docs[1], 1.5)]
    vec = [(docs[1], 0.95), (docs[0], 0.80)]
    fused = rrf_fusion(bm25, vec, k=60)
    assert len(fused) == 2
    ids = [r.attribute.attribute_id for r in fused]
    assert "labs.hba1c" in ids
    assert "vitals.systolic_bp" in ids


def test_rrf_fusion_missing_vector_rank(docs):
    bm25 = [(docs[0], 2.0)]
    vec = []
    fused = rrf_fusion(bm25, vec, k=60)
    assert len(fused) == 1
    assert fused[0].vector_rank is None


# ── Decision policy ───────────────────────────────────────────────────────────

def test_classify_none_on_empty_results():
    decision = classify("unknown clinical concept xyz", [])
    assert decision.outcome == DecisionOutcome.NONE


def test_classify_exact_high_score(docs):
    from akrag.models import SearchResult

    results = [
        SearchResult(attribute=docs[3], rrf_score=0.032, bm25_rank=0, vector_rank=0),
    ]
    decision = classify("diabetic patients", results, exact_threshold=0.92)
    assert decision.outcome == DecisionOutcome.EXACT
    assert decision.selected is not None
    assert decision.selected.attribute_id == "clinical.is_diabetic"


def test_classify_ambiguous_blood_pressure(docs):
    from akrag.models import SearchResult

    results = [
        SearchResult(attribute=docs[1], rrf_score=0.025, bm25_rank=0, vector_rank=0),
        SearchResult(attribute=docs[2], rrf_score=0.024, bm25_rank=1, vector_rank=1),
    ]
    decision = classify("blood pressure issues", results, exact_threshold=0.92, near_threshold=0.75)
    assert decision.outcome in (DecisionOutcome.AMBIGUOUS, DecisionOutcome.NEAR)
    assert len(decision.options) >= 2


# ── Local search ──────────────────────────────────────────────────────────────

def test_local_backend_indexes_and_lists(backend):
    b, _ = backend
    indexes = asyncio.get_event_loop().run_until_complete(b.list_indexes())
    assert len(indexes) == 1


def test_local_bm25_finds_hba1c(backend):
    b, docs = backend
    bm25_results = b._bm25_search("glycated hemoglobin A1C", docs, top_k=5)
    assert len(bm25_results) > 0
    top_id = bm25_results[0][0].attribute_id
    assert top_id == "labs.hba1c"


def test_local_search_returns_results(backend):
    b, docs = backend
    q_vec = docs[3].vector  # is_diabetic vector
    results = asyncio.get_event_loop().run_until_complete(
        b.search("diabetic patients", q_vec, top_k=3)
    )
    assert len(results) > 0


def test_local_channel_filter(backend):
    b, docs = backend
    # Restrict labs.hba1c to analytics channel only
    docs[0].governance.allowed_channels = ["analytics"]
    asyncio.get_event_loop().run_until_complete(b.index(docs))

    q_vec = docs[0].vector
    results = asyncio.get_event_loop().run_until_complete(
        b.search("HbA1c", q_vec, top_k=5, channel="care_management")
    )
    ids = [r.attribute.attribute_id for r in results]
    assert "labs.hba1c" not in ids
