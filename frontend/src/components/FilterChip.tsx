import type { DSLFilter } from "../types";

export function FilterChip({ filter }: { filter: DSLFilter }) {
  return (
    <div className="filter-chip" title={`from: "${filter.source_phrase}"`}>
      <span className="filter-chip__name">{filter.business_name}</span>
      <code className="filter-chip__id">{filter.attribute_id}</code>
    </div>
  );
}
