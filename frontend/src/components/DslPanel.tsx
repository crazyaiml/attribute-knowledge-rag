import type { DSLFilter } from "../types";
import { FilterChip } from "./FilterChip";

function buildDsl(filters: DSLFilter[]) {
  return {
    version: "1.0",
    type: "attribute_filter",
    filters: filters.map((f) => ({
      attribute_id: f.attribute_id,
      business_name: f.business_name,
      technical_field: f.technical_field,
      operator: f.operator ?? null,
      value: f.value ?? null,
      source_phrase: f.source_phrase,
    })),
  };
}

export function DslPanel({ filters }: { filters: DSLFilter[] }) {
  const dsl = buildDsl(filters);

  return (
    <aside className="dsl-panel">
      <h2>Resolved DSL</h2>
      <p className="muted">
        Deterministic output — every entry is a real <code>attribute_id</code> from the governed
        catalog. Nothing here was generated freeform by the LLM.
      </p>
      {filters.length === 0 ? (
        <p className="muted">No attributes resolved yet.</p>
      ) : (
        <>
          <div className="filter-list">
            {filters.map((f) => (
              <FilterChip key={f.attribute_id} filter={f} />
            ))}
          </div>
          <pre className="dsl-json">{JSON.stringify(dsl, null, 2)}</pre>
        </>
      )}
      <p className="muted footnote">{filters.length} attribute(s) in this segment definition.</p>
    </aside>
  );
}
