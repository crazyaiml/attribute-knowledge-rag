import { useId, useState } from "react";
import { clarifyPhrase, evaluateQuery } from "../api";
import { ClarificationCard } from "../components/ClarificationCard";
import { DslPanel } from "../components/DslPanel";
import { FilterChip } from "../components/FilterChip";
import { SetupPanel } from "../components/SetupPanel";
import type { DSLFilter, PhraseDecision, ProvidersHealth } from "../types";

const EXAMPLE_QUERIES = [
  "diabetic patients over 65 with elevated HbA1c",
  "high risk members with recent ER visits",
  "patients with clinical outcomes within 12 months",
];

interface Turn {
  id: string;
  input: string;
  filters: DSLFilter[];
  unresolved: PhraseDecision[];
  error?: string;
}

function mergeFilters(existing: DSLFilter[], incoming: DSLFilter[]): DSLFilter[] {
  const byId = new Map(existing.map((f) => [f.attribute_id, f]));
  for (const f of incoming) byId.set(f.attribute_id, f);
  return [...byId.values()];
}

export function DemoPage({ providers }: { providers: ProvidersHealth | null }) {
  const sessionId = useId();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [resolvedFilters, setResolvedFilters] = useState<DSLFilter[]>([]);
  const [queryText, setQueryText] = useState("");
  const [evaluating, setEvaluating] = useState(false);
  const [clarifyingKey, setClarifyingKey] = useState<string | null>(null);

  async function runQuery(input: string) {
    if (!input.trim() || evaluating) return;
    setEvaluating(true);
    const turnId = crypto.randomUUID();
    try {
      const result = await evaluateQuery(input);
      setResolvedFilters((prev) => mergeFilters(prev, result.filters));
      setTurns((prev) => [
        ...prev,
        { id: turnId, input, filters: result.filters, unresolved: result.unresolved },
      ]);
      setQueryText("");
    } catch (err) {
      setTurns((prev) => [
        ...prev,
        {
          id: turnId,
          input,
          filters: [],
          unresolved: [],
          error: err instanceof Error ? err.message : String(err),
        },
      ]);
    } finally {
      setEvaluating(false);
    }
  }

  async function handleChoose(turnId: string, phrase: string, attributeId: string) {
    const key = `${turnId}:${phrase}`;
    setClarifyingKey(key);
    try {
      const decision = await clarifyPhrase(sessionId, phrase, attributeId);
      if (decision.selected) {
        const filter: DSLFilter = {
          attribute_id: decision.selected.attribute_id,
          business_name: decision.selected.business_name,
          technical_field: decision.selected.technical_field ?? "",
          source_phrase: phrase,
        };
        setResolvedFilters((prev) => mergeFilters(prev, [filter]));
        setTurns((prev) =>
          prev.map((t) =>
            t.id === turnId
              ? {
                  ...t,
                  filters: [...t.filters, filter],
                  unresolved: t.unresolved.filter((u) => u.phrase !== phrase),
                }
              : t
          )
        );
      }
    } catch (err) {
      alert(`Could not resolve "${phrase}": ${err instanceof Error ? err.message : err}`);
    } finally {
      setClarifyingKey(null);
    }
  }

  return (
    <main className="app-body">
      <SetupPanel indexes={providers?.indexes ?? []} />

      <section className="conversation-panel">
        <h2>Ask for a population</h2>
        <div className="example-chips">
          {EXAMPLE_QUERIES.map((q) => (
            <button key={q} className="chip-btn" onClick={() => setQueryText(q)}>
              {q}
            </button>
          ))}
        </div>

        <div className="transcript">
          {turns.length === 0 && (
            <p className="muted">
              Describe a population in plain English below. The system will never invent a field —
              it will either resolve to a governed attribute or ask you to clarify.
            </p>
          )}
          {turns.map((turn) => (
            <div key={turn.id} className="turn">
              <div className="turn__user">{turn.input}</div>
              {turn.error && <p className="error">Evaluation failed: {turn.error}</p>}
              {turn.filters.length > 0 && (
                <div className="turn__filters">
                  {turn.filters.map((f) => (
                    <FilterChip key={f.attribute_id + f.source_phrase} filter={f} />
                  ))}
                </div>
              )}
              {turn.unresolved.map((decision) => (
                <ClarificationCard
                  key={decision.phrase}
                  decision={decision}
                  busy={clarifyingKey === `${turn.id}:${decision.phrase}`}
                  onChoose={(phrase, attrId) => handleChoose(turn.id, phrase, attrId)}
                />
              ))}
            </div>
          ))}
        </div>

        <form
          className="query-form"
          onSubmit={(e) => {
            e.preventDefault();
            runQuery(queryText);
          }}
        >
          <input
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            placeholder='e.g. "diabetic patients over 65 with elevated HbA1c"'
            disabled={evaluating}
          />
          <button className="btn" type="submit" disabled={evaluating || !queryText.trim()}>
            {evaluating ? "Thinking…" : "Evaluate"}
          </button>
        </form>
      </section>

      <DslPanel filters={resolvedFilters} />
    </main>
  );
}
