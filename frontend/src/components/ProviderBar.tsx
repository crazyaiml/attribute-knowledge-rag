import type { ProvidersHealth } from "../types";

export function ProviderBar({ providers }: { providers: ProvidersHealth | null }) {
  if (!providers) {
    return <div className="provider-bar provider-bar--loading">connecting to AK-RAG API…</div>;
  }
  return (
    <div className="provider-bar">
      <span className="provider-chip">
        <b>LLM</b> {providers.llm} <code>{providers.llm_model}</code>
      </span>
      <span className="provider-chip">
        <b>Embedding</b> {providers.embedding} <code>{providers.embedding_model}</code> ({providers.embedding_dim}d)
      </span>
      <span className="provider-chip">
        <b>Search</b> {providers.search}
      </span>
    </div>
  );
}
