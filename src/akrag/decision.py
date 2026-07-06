from __future__ import annotations

from akrag.models import DecisionOutcome, PhraseDecision, SearchResult


def classify(
    phrase: str,
    results: list[SearchResult],
    exact_threshold: float = 0.92,
    near_threshold: float = 0.75,
) -> PhraseDecision:
    """Apply the exact / near / ambiguous / none policy to a ranked result list.

    Thresholds are based on RRF score, which ranges from 0 to ~0.033 per list.
    Normalise to [0, 1] based on the theoretical max for two lists of k=60:
        max_rrf = 1/(60+1) + 1/(60+1) ≈ 0.0328

    We use the top result's rrf_score relative to the two-list max as confidence.
    """
    if not results:
        return PhraseDecision(phrase=phrase, outcome=DecisionOutcome.NONE, options=[])

    max_rrf = 2.0 / 61.0  # two ranked lists, rank-1 from each
    top = results[0]
    confidence = min(top.rrf_score / max_rrf, 1.0)

    if confidence >= exact_threshold:
        return PhraseDecision(
            phrase=phrase,
            outcome=DecisionOutcome.EXACT,
            selected=top.attribute,
            options=results[:1],
        )

    if confidence >= near_threshold:
        # Show top options if multiple candidates are close
        close = [r for r in results if r.rrf_score / max_rrf >= near_threshold * 0.85]
        if len(close) > 1:
            return PhraseDecision(
                phrase=phrase,
                outcome=DecisionOutcome.AMBIGUOUS,
                options=close[:5],
                clarification_message=_default_clarification(phrase, close),
            )
        return PhraseDecision(
            phrase=phrase,
            outcome=DecisionOutcome.NEAR,
            options=results[:5],
            clarification_message=_default_clarification(phrase, results[:5]),
        )

    if results:
        return PhraseDecision(
            phrase=phrase,
            outcome=DecisionOutcome.AMBIGUOUS if len(results) > 1 else DecisionOutcome.NONE,
            options=results[:5],
            clarification_message=_default_clarification(phrase, results[:5]),
        )

    return PhraseDecision(phrase=phrase, outcome=DecisionOutcome.NONE, options=[])


def _default_clarification(phrase: str, options: list[SearchResult]) -> str:
    lines = [f'No exact match found for "{phrase}". Did you mean:']
    for i, r in enumerate(options, 1):
        a = r.attribute
        lines.append(f"  {i}. {a.business_name} ({a.attribute_id})")
        if a.definition:
            lines.append(f"     {a.definition}")
    lines.append("\nReply with the number or provide an alternative description.")
    return "\n".join(lines)


def apply_selection(
    decision: PhraseDecision,
    chosen_attribute_id: str,
) -> PhraseDecision:
    """Resolve an ambiguous/near decision after the user picks an option."""
    for result in decision.options:
        if result.attribute.attribute_id == chosen_attribute_id:
            return PhraseDecision(
                phrase=decision.phrase,
                outcome=DecisionOutcome.EXACT,
                selected=result.attribute,
                options=decision.options,
            )
    raise ValueError(f"attribute_id {chosen_attribute_id!r} not in options for phrase {decision.phrase!r}")
