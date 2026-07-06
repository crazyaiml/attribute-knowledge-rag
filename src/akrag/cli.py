from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

app = typer.Typer(name="akrag", help="Attribute Knowledge RAG CLI")


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@app.command()
def validate(
    path: str = typer.Argument(..., help="CSV or XLSX attribute file to validate"),
) -> None:
    """Validate an attribute contract file against the schema."""
    from akrag.contract import validate_records
    from akrag.io import read_file

    records = read_file(path)
    result = validate_records(records)

    if result.valid:
        rprint(f"[green]✓ Valid[/green] — {result.total} attributes passed.")
    else:
        rprint(f"[red]✗ Invalid[/red] — {len(result.errors)} error(s) in {result.total} records.")
        for e in result.errors[:20]:
            rprint(f"  Row {e.get('row')}: {e.get('field')} — {e.get('error')}")
        raise typer.Exit(1)


@app.command("to-ndjson")
def to_ndjson(
    input_path: str = typer.Argument(..., help="CSV or XLSX input file"),
    output_path: str = typer.Argument(..., help="Output NDJSON path"),
) -> None:
    """Convert an attribute CSV/XLSX to NDJSON (one document per line)."""
    from akrag.contract import coerce_to_document, validate_records
    from akrag.io import read_file, write_ndjson

    records = read_file(input_path)
    result = validate_records(records)
    if not result.valid:
        rprint(f"[red]Validation failed with {len(result.errors)} errors. Fix before converting.[/red]")
        raise typer.Exit(1)

    documents = [coerce_to_document(r) for r in records]
    write_ndjson(documents, output_path)
    rprint(f"[green]✓[/green] Wrote {len(documents)} documents → {output_path}")


@app.command()
def mapping(
    output_path: str = typer.Argument(..., help="Output JSON path for the OpenSearch mapping"),
    vector_dim: int = typer.Option(384, help="Embedding vector dimension"),
) -> None:
    """Generate an OpenSearch index mapping for the attribute index."""
    from akrag.search.opensearch_backend import _build_mapping

    m = _build_mapping(vector_dim)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(m, indent=2))
    rprint(f"[green]✓[/green] Mapping written → {output_path}")


@app.command()
def ingest(
    path: str = typer.Argument(..., help="CSV, XLSX, or NDJSON attribute file"),
    index_name: Optional[str] = typer.Option(None, help="Override the generated index name"),
    no_alias: bool = typer.Option(False, "--no-alias", help="Skip alias swap after indexing"),
) -> None:
    """Embed and index attributes using the configured search backend."""
    from akrag.embeddings.factory import get_embedder
    from akrag.io import read_file
    from akrag.llm.factory import get_llm
    from akrag.orchestrator import Orchestrator
    from akrag.search.factory import get_search_backend

    records = read_file(path)
    orc = Orchestrator(get_embedder(), get_search_backend(), get_llm())
    result = _run(orc.ingest(records, index_name=index_name, swap_alias=not no_alias))

    rprint(f"[green]✓[/green] Indexed {result.indexed}/{result.total} → {result.index_name}")
    if result.errors:
        for e in result.errors[:10]:
            rprint(f"  [yellow]WARN[/yellow] {e}")


@app.command()
def query(
    ndjson_path: str = typer.Argument(..., help="NDJSON attribute file to search against"),
    phrases: list[str] = typer.Argument(..., help="One or more attribute phrases to search"),
    top_k: int = typer.Option(5, help="Number of candidates per phrase"),
) -> None:
    """Run local hybrid retrieval and decision policy against an NDJSON file.

    No server required — uses the local in-memory backend.
    """
    from akrag.contract import coerce_to_document
    from akrag.embeddings.factory import get_embedder
    from akrag.io import read_ndjson
    from akrag.search.local import LocalSearchBackend

    records = read_ndjson(ndjson_path)
    documents = [coerce_to_document(r) for r in records]

    embedder = get_embedder()
    search = LocalSearchBackend()

    # Embed documents
    texts = [d.compose_embedding_text() for d in documents]
    vectors = _run(embedder.embed(texts))
    for doc, vec in zip(documents, vectors):
        doc.vector = vec

    _run(search.index(documents))

    # Build a fake evaluate call using only the local LLM-free path
    from akrag.decision import classify

    async def _search_all():
        results = []
        for phrase in phrases:
            q_vec = await embedder.embed_one(phrase)
            hits = await search.search(phrase, q_vec, top_k=top_k)
            decision = classify(phrase, hits)
            results.append(decision)
        return results

    decisions = _run(_search_all())

    filters = []
    unresolved = []
    for d in decisions:
        if d.outcome.value == "exact" and d.selected:
            filters.append({
                "attribute_id": d.selected.attribute_id,
                "business_name": d.selected.business_name,
                "source_phrase": d.phrase,
            })
        else:
            unresolved.append({
                "phrase": d.phrase,
                "outcome": d.outcome.value,
                "options": [
                    {"attribute_id": r.attribute.attribute_id,
                     "business_name": r.attribute.business_name}
                    for r in d.options
                ],
            })

    output = {"filters": filters, "unresolved": unresolved}
    rprint(json.dumps(output, indent=2))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8080),
    reload: bool = typer.Option(False),
) -> None:
    """Start the AK-RAG FastAPI server."""
    import uvicorn

    uvicorn.run("akrag.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
