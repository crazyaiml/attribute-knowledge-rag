from __future__ import annotations

from fastapi import APIRouter

from akrag.models import (
    ClarifyRequest,
    EvaluateRequest,
    PhraseDecision,
    QueryResult,
    SearchRequest,
)

router = APIRouter(prefix="/query", tags=["Query"])


def _get_orchestrator():
    from akrag.embeddings.factory import get_embedder
    from akrag.llm.factory import get_llm
    from akrag.orchestrator import Orchestrator
    from akrag.search.factory import get_search_backend

    return Orchestrator(get_embedder(), get_search_backend(), get_llm())


@router.post(
    "/search",
    response_model=list[PhraseDecision],
    summary="Search for attributes matching one or more phrases",
    description=(
        "Run hybrid retrieval (BM25 + vector search + RRF) for each supplied phrase independently "
        "and classify each result as `exact`, `near`, `ambiguous`, or `none`.\n\n"
        "**When to use this vs `/query/evaluate`:**\n"
        "- Use `/query/search` when you already know the individual attribute phrases "
        "(e.g. from a structured clinical UI where the user selects conditions).\n"
        "- Use `/query/evaluate` when you have raw natural language and want the LLM to "
        "parse phrases before searching.\n\n"
        "**Healthcare example phrases:** `\"diabetic patients\"`, `\"elevated HbA1c\"`, "
        "`\"blood pressure issues\"`, `\"medication non-adherent\"`, `\"high clinical risk score\"`\n\n"
        "**Decision outcomes:**\n"
        "| Outcome | Meaning | Next step |\n"
        "|---|---|---|\n"
        "| `exact` | High-confidence single match | Use `selected` attribute directly |\n"
        "| `near` | Likely match, low confidence | Show top `options` and ask user to confirm |\n"
        "| `ambiguous` | Multiple equally-plausible candidates | Show `options`, ask user to pick |\n"
        "| `none` | No supported attribute found | Ask user to rephrase or pick a broader category |\n\n"
        "Call `/query/clarify` once the user picks an attribute from the options list."
    ),
    responses={
        200: {"description": "One `PhraseDecision` per input phrase"},
        422: {"description": "Request body validation error"},
    },
)
async def search(request: SearchRequest) -> list[PhraseDecision]:
    orc = _get_orchestrator()
    results = []
    for phrase in request.phrases:
        decision = await orc.search_phrase(
            phrase=phrase,
            channel=request.channel,
            top_k=request.top_k,
        )
        results.append(decision)
    return results


@router.post(
    "/evaluate",
    response_model=QueryResult,
    summary="Full pipeline: parse → retrieve → classify → clarify → DSL",
    description=(
        "End-to-end evaluation of a natural-language clinical query.\n\n"
        "**Pipeline steps:**\n"
        "1. **Parse** — LLM extracts individual attribute phrases from the free-text input\n"
        "2. **Retrieve** — Hybrid search (BM25 + vector + RRF) runs in parallel for each phrase\n"
        "3. **Classify** — Each result is labelled `exact`, `near`, `ambiguous`, or `none`\n"
        "4. **Clarify** — For non-exact matches the LLM generates a clarification message\n"
        "5. **DSL** — Resolved (exact) attributes are assembled into a structured filter definition\n\n"
        "**Healthcare example input:**\n"
        "> `\"diabetic patients over 65 with elevated HbA1c and blood pressure issues\"`\n\n"
        "- `\"diabetic patients\"` → **exact** → `clinical.is_diabetic`\n"
        "- `\"over 65\"` → **exact** → `patient.age`\n"
        "- `\"elevated HbA1c\"` → **near** → `labs.hba1c` (asks: what threshold?)\n"
        "- `\"blood pressure issues\"` → **ambiguous** → systolic BP vs diastolic BP vs hypertension flag\n\n"
        "**Typical conversation flow:**\n"
        "```\n"
        "POST /query/evaluate   → { filters: [...], unresolved: [...] }\n"
        "   ↓ show clarification_message to user for each unresolved phrase\n"
        "POST /query/clarify    → resolved PhraseDecision\n"
        "   ↓ merge into DSL on the client side\n"
        "```\n\n"
        "Exact-match phrases are immediately added to `filters` and included in the `dsl` block. "
        "Ambiguous phrases appear in `unresolved` with a `clarification_message` ready to display."
    ),
    responses={
        200: {"description": "Full evaluation result with resolved filters and unresolved phrases"},
        422: {"description": "Request body validation error"},
    },
)
async def evaluate(request: EvaluateRequest) -> QueryResult:
    orc = _get_orchestrator()
    return await orc.evaluate(
        user_input=request.input,
        channel=request.channel,
        top_k=request.top_k,
    )


@router.post(
    "/clarify",
    response_model=PhraseDecision,
    summary="Resolve an ambiguous phrase after the user picks an attribute",
    description=(
        "After `/query/search` or `/query/evaluate` returns a `near` or `ambiguous` decision, "
        "the client presents the `options` list to the user. Once the user picks one, call this "
        "endpoint to confirm the selection and get back an `exact` `PhraseDecision`.\n\n"
        "**Steps:**\n"
        "1. Display the `clarification_message` and `options` from the previous response to the user\n"
        "2. Collect the user's choice (the `attribute_id` they selected)\n"
        "3. POST here with the original `phrase` and the chosen `attribute_id`\n"
        "4. Receive an `exact` `PhraseDecision` with `selected` populated\n"
        "5. Add the resolved attribute to your DSL\n\n"
        "The `session_id` field is client-managed — use it to correlate clarification rounds "
        "in a multi-turn conversation."
    ),
    responses={
        200: {"description": "Resolved `PhraseDecision` with `outcome: exact` and `selected` populated"},
        400: {"description": "The chosen `attribute_id` was not in the options list for this phrase"},
        422: {"description": "Request body validation error"},
    },
)
async def clarify(request: ClarifyRequest) -> PhraseDecision:
    from akrag.decision import apply_selection

    orc = _get_orchestrator()
    original = await orc.search_phrase(phrase=request.phrase)
    return apply_selection(original, request.chosen_attribute_id)
