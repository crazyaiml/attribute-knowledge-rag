from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

from akrag.config import get_settings
from akrag.contract import coerce_to_document
from akrag.decision import classify
from akrag.embeddings.base import EmbeddingProvider
from akrag.llm.base import LLMProvider
from akrag.models import (
    DSLFilter,
    DecisionOutcome,
    IngestResult,
    PhraseDecision,
    QueryResult,
)
from akrag.search.base import SearchBackend


class Orchestrator:
    """Central coordinator: ingestion, phrase evaluation, clarification, DSL assembly."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        search: SearchBackend,
        llm: LLMProvider,
    ) -> None:
        self._embedder = embedder
        self._search = search
        self._llm = llm
        self._settings = get_settings()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    async def ingest(
        self,
        raw_records: list[dict[str, Any]],
        index_name: str | None = None,
        swap_alias: bool = True,
    ) -> IngestResult:
        from akrag.contract import validate_records

        validation = validate_records(raw_records)
        if not validation.valid:
            return IngestResult(
                total=validation.total,
                indexed=0,
                failed=len(validation.errors),
                errors=[str(e) for e in validation.errors],
            )

        documents = [coerce_to_document(r) for r in raw_records]

        # Embed in batches of 64
        texts = [d.embedding_text or d.compose_embedding_text() for d in documents]
        vectors = await self._batch_embed(texts)

        today = date.today().strftime("%Y_%m_%d")
        model_name = self._embedder.model_name

        for doc, vec in zip(documents, vectors):
            doc.vector = vec
            doc.embedding_model = model_name
            doc.embedding_version = today

        final_index = await self._search.index(documents, index_name=index_name)

        if swap_alias:
            alias = f"{self._settings.opensearch_index_prefix}_current"
            await self._search.swap_alias(alias, final_index)

        return IngestResult(
            total=len(documents),
            indexed=len(documents),
            failed=0,
            index_name=final_index,
        )

    async def _batch_embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = await self._embedder.embed(batch)
            all_vectors.extend(vectors)
        return all_vectors

    # ── Query evaluation ──────────────────────────────────────────────────────

    async def evaluate(
        self,
        user_input: str,
        channel: str | None = None,
        top_k: int | None = None,
        index_name: str | None = None,
    ) -> QueryResult:
        s = self._settings
        k = top_k or s.top_k

        phrases = await self._llm.parse_phrases(user_input)

        decisions = await asyncio.gather(
            *[self._evaluate_phrase(phrase, channel, k, index_name) for phrase in phrases]
        )

        filters: list[DSLFilter] = []
        unresolved: list[PhraseDecision] = []

        for decision in decisions:
            if decision.outcome == DecisionOutcome.EXACT and decision.selected:
                filters.append(DSLFilter(
                    attribute_id=decision.selected.attribute_id,
                    business_name=decision.selected.business_name,
                    technical_field=decision.selected.technical_field,
                    source_phrase=decision.phrase,
                ))
            else:
                if decision.clarification_message is None and decision.options:
                    decision.clarification_message = await self._llm.generate_clarification(
                        decision.phrase,
                        [r.attribute.model_dump() for r in decision.options],
                        decision.outcome.value,
                    )
                unresolved.append(decision)

        dsl = self._build_dsl(filters) if filters else None

        return QueryResult(
            input=user_input,
            phrases=phrases,
            filters=filters,
            unresolved=unresolved,
            dsl=dsl,
        )

    async def search_phrase(
        self,
        phrase: str,
        channel: str | None = None,
        top_k: int = 10,
        index_name: str | None = None,
    ) -> PhraseDecision:
        return await self._evaluate_phrase(phrase, channel, top_k, index_name)

    async def _evaluate_phrase(
        self,
        phrase: str,
        channel: str | None,
        top_k: int,
        index_name: str | None,
    ) -> PhraseDecision:
        s = self._settings
        query_vector = await self._embedder.embed_one(phrase)
        results = await self._search.search(
            query=phrase,
            query_vector=query_vector,
            top_k=top_k,
            index_name=index_name,
            channel=channel,
        )
        return classify(phrase, results, s.exact_threshold, s.near_threshold)

    # ── DSL assembly ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_dsl(filters: list[DSLFilter]) -> dict[str, Any]:
        return {
            "version": "1.0",
            "type": "attribute_filter",
            "filters": [
                {
                    "attribute_id": f.attribute_id,
                    "business_name": f.business_name,
                    "technical_field": f.technical_field,
                    "operator": f.operator,
                    "value": f.value,
                    "source_phrase": f.source_phrase,
                }
                for f in filters
            ],
        }
