import { Link } from "react-router-dom";

const PIPELINE_STEPS = [
  ["Natural language", "User describes a population in plain English."],
  ["LLM phrase extraction", "The LLM splits the request into individual attribute phrases — nothing else."],
  ["Hybrid retrieval", "BM25 + kNN vector search run per phrase, fused with Reciprocal Rank Fusion."],
  ["Decision policy", "Each phrase is classified exact / near / ambiguous / none against fixed thresholds."],
  ["Clarify or resolve", "Non-exact phrases ask the user to choose — the system never guesses."],
  ["Governed DSL", "Only attribute_ids that exist in the catalog can appear in the final filter."],
];

export function SetupPanel({ indexes }: { indexes: string[] }) {
  return (
    <aside className="setup-panel">
      <h2>How this pattern works</h2>
      <ol className="pipeline">
        {PIPELINE_STEPS.map(([title, desc]) => (
          <li key={title}>
            <strong>{title}</strong>
            <span>{desc}</span>
          </li>
        ))}
      </ol>

      <h2>Catalog status</h2>
      {indexes.length === 0 ? (
        <p className="error">
          No catalog indexed yet. Go to <Link to="/catalog">Catalog</Link> to load one before
          asking questions below.
        </p>
      ) : (
        <p className="success">
          {indexes.length} index(es) loaded: {indexes.join(", ")}.{" "}
          <Link to="/catalog">Manage catalog</Link>
        </p>
      )}
    </aside>
  );
}
