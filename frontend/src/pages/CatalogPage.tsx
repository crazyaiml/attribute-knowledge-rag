import { useEffect, useState, type FormEvent } from "react";
import { ingestFile, ingestSampleAttributes, listIndexes, searchPhrases } from "../api";
import type { IngestResult, PhraseDecision } from "../types";

export function CatalogPage({ onCatalogChanged }: { onCatalogChanged: () => void }) {
  const [indexes, setIndexes] = useState<string[]>([]);
  const [indexesError, setIndexesError] = useState<string | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [indexName, setIndexName] = useState("");
  const [ingesting, setIngesting] = useState(false);
  const [ingestResult, setIngestResult] = useState<IngestResult | null>(null);
  const [ingestError, setIngestError] = useState<string | null>(null);

  const [phrase, setPhrase] = useState("");
  const [topK, setTopK] = useState(5);
  const [searching, setSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<PhraseDecision[]>([]);
  const [searchError, setSearchError] = useState<string | null>(null);

  async function refreshIndexes() {
    try {
      setIndexes(await listIndexes());
      setIndexesError(null);
    } catch (err) {
      setIndexesError(err instanceof Error ? err.message : String(err));
    }
  }

  useEffect(() => {
    refreshIndexes();
  }, []);

  async function handleIngest(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setIngesting(true);
    setIngestError(null);
    try {
      const result = await ingestFile(file, indexName || undefined);
      setIngestResult(result);
      await refreshIndexes();
      onCatalogChanged();
    } catch (err) {
      setIngestError(err instanceof Error ? err.message : String(err));
    } finally {
      setIngesting(false);
    }
  }

  async function handleLoadSample() {
    setIngesting(true);
    setIngestError(null);
    try {
      const result = await ingestSampleAttributes();
      setIngestResult(result);
      await refreshIndexes();
      onCatalogChanged();
    } catch (err) {
      setIngestError(err instanceof Error ? err.message : String(err));
    } finally {
      setIngesting(false);
    }
  }

  async function handleSearch(e: FormEvent) {
    e.preventDefault();
    if (!phrase.trim()) return;
    setSearching(true);
    setSearchError(null);
    try {
      setSearchResults(await searchPhrases([phrase], topK));
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : String(err));
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  }

  return (
    <main className="page">
      <section className="panel">
        <h2>Index a catalog</h2>
        <p className="muted">
          Upload a CSV, XLSX, or NDJSON attribute file. Each row/document becomes one governed
          attribute — embedded once and indexed into the active search backend.
        </p>
        <form className="stacked-form" onSubmit={handleIngest}>
          <input
            type="file"
            accept=".csv,.xlsx,.ndjson,.jsonl"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <input
            type="text"
            placeholder="index name (optional, auto-generated if blank)"
            value={indexName}
            onChange={(e) => setIndexName(e.target.value)}
          />
          <button className="btn" type="submit" disabled={!file || ingesting}>
            {ingesting ? "Indexing…" : "Ingest & index"}
          </button>
        </form>
        <p className="muted">
          Or{" "}
          <button className="link-btn" onClick={handleLoadSample} disabled={ingesting}>
            load the built-in 18-attribute sample catalog
          </button>{" "}
          instead.
        </p>
        {ingestError && <p className="error">{ingestError}</p>}
        {ingestResult && (
          <p className="success">
            Indexed {ingestResult.indexed}/{ingestResult.total} attributes
            {ingestResult.index_name ? ` → ${ingestResult.index_name}` : ""}.
            {ingestResult.failed > 0 && ` ${ingestResult.failed} row(s) failed.`}
          </p>
        )}
        {ingestResult && ingestResult.errors.length > 0 && (
          <ul className="error-list">
            {ingestResult.errors.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        )}
      </section>

      <section className="panel">
        <h2>Active indexes</h2>
        <button className="link-btn" onClick={refreshIndexes}>
          refresh
        </button>
        {indexesError && <p className="error">{indexesError}</p>}
        {indexes.length === 0 ? (
          <p className="muted">No indexes yet.</p>
        ) : (
          <ul className="index-list">
            {indexes.map((name) => (
              <li key={name}>
                <code>{name}</code>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="panel">
        <h2>Search the embeddings directly</h2>
        <p className="muted">
          Runs raw hybrid retrieval (BM25 + vector + RRF) against the indexed catalog — no LLM
          involved. Useful for testing embedding/search quality independent of phrase extraction.
        </p>
        <form className="search-form" onSubmit={handleSearch}>
          <input
            type="text"
            placeholder='e.g. "elevated blood sugar"'
            value={phrase}
            onChange={(e) => setPhrase(e.target.value)}
          />
          <input
            type="number"
            min={1}
            max={20}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            title="top_k"
          />
          <button className="btn" type="submit" disabled={searching || !phrase.trim()}>
            {searching ? "Searching…" : "Search"}
          </button>
        </form>
        {searchError && <p className="error">{searchError}</p>}
        {searchResults.map((decision) => (
          <div key={decision.phrase} className="search-decision">
            <div className="search-decision__head">
              <span>"{decision.phrase}"</span>
              <span className={`outcome-badge outcome-badge--${decision.outcome}`}>
                {decision.outcome}
              </span>
            </div>
            <table className="score-table">
              <thead>
                <tr>
                  <th>Attribute</th>
                  <th>BM25</th>
                  <th>Vector</th>
                  <th>RRF</th>
                </tr>
              </thead>
              <tbody>
                {decision.selected && (
                  <tr className="score-table__selected">
                    <td>
                      {decision.selected.business_name} <code>{decision.selected.attribute_id}</code>
                    </td>
                    <td colSpan={3}>selected (exact)</td>
                  </tr>
                )}
                {decision.outcome !== "exact" && decision.options.map((opt) => (
                  <tr key={opt.attribute.attribute_id}>
                    <td>
                      {opt.attribute.business_name} <code>{opt.attribute.attribute_id}</code>
                    </td>
                    <td>{opt.bm25_score?.toFixed(2) ?? "—"}</td>
                    <td>{opt.vector_score?.toFixed(2) ?? "—"}</td>
                    <td>{opt.rrf_score?.toFixed(4) ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </section>
    </main>
  );
}
