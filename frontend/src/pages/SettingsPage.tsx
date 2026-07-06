import { useEffect, useState, type FormEvent } from "react";
import { getSettingsOptions, getSettingsPublic, updateSettings } from "../api";
import type {
  EmbeddingProvider,
  LlmProvider,
  SearchBackend,
  SettingsOptions,
  SettingsPublic,
} from "../types";

export function SettingsPage({ onProvidersChanged }: { onProvidersChanged: () => void }) {
  const [options, setOptions] = useState<SettingsOptions | null>(null);
  const [current, setCurrent] = useState<SettingsPublic | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [llmProvider, setLlmProvider] = useState<LlmProvider>("ollama");
  const [llmModel, setLlmModel] = useState("");
  const [anthropicKey, setAnthropicKey] = useState("");
  const [openaiKey, setOpenaiKey] = useState("");
  const [awsAccessKey, setAwsAccessKey] = useState("");
  const [awsSecretKey, setAwsSecretKey] = useState("");
  const [ollamaBaseUrl, setOllamaBaseUrl] = useState("");

  const [embeddingProvider, setEmbeddingProvider] = useState<EmbeddingProvider>("sentence");
  const [embeddingModel, setEmbeddingModel] = useState("");
  const [googleKey, setGoogleKey] = useState("");

  const [searchBackend, setSearchBackend] = useState<SearchBackend>("local");
  const [opensearchHost, setOpensearchHost] = useState("");
  const [opensearchPort, setOpensearchPort] = useState(9200);
  const [chromaHost, setChromaHost] = useState("");
  const [chromaPort, setChromaPort] = useState(8000);

  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [indexResetWarning, setIndexResetWarning] = useState(false);

  function applyCurrent(s: SettingsPublic) {
    setCurrent(s);
    setLlmProvider(s.llm_provider);
    setLlmModel(s.llm_model);
    setOllamaBaseUrl(s.ollama_base_url);
    setEmbeddingProvider(s.embedding_provider);
    setEmbeddingModel(s.embedding_model);
    setSearchBackend(s.search_backend);
    setOpensearchHost(s.opensearch_host);
    setOpensearchPort(s.opensearch_port);
    setChromaHost(s.chroma_host);
    setChromaPort(s.chroma_port);
  }

  useEffect(() => {
    Promise.all([getSettingsOptions(), getSettingsPublic()])
      .then(([opts, s]) => {
        setOptions(opts);
        applyCurrent(s);
      })
      .catch((err) => setLoadError(err instanceof Error ? err.message : String(err)));
  }, []);

  async function handleSave(e: FormEvent) {
    e.preventDefault();
    setSaving(true);
    setSaveError(null);
    setIndexResetWarning(false);
    try {
      const { settings, index_reset } = await updateSettings({
        llm_provider: llmProvider,
        llm_model: llmModel,
        anthropic_api_key: anthropicKey || undefined,
        openai_api_key: openaiKey || undefined,
        aws_access_key_id: awsAccessKey || undefined,
        aws_secret_access_key: awsSecretKey || undefined,
        ollama_base_url: ollamaBaseUrl,
        embedding_provider: embeddingProvider,
        embedding_model: embeddingModel,
        google_api_key: googleKey || undefined,
        search_backend: searchBackend,
        opensearch_host: opensearchHost,
        opensearch_port: opensearchPort,
        chroma_host: chromaHost,
        chroma_port: chromaPort,
      });
      applyCurrent(settings);
      setIndexResetWarning(index_reset);
      setAnthropicKey("");
      setOpenaiKey("");
      setAwsAccessKey("");
      setAwsSecretKey("");
      setGoogleKey("");
      onProvidersChanged();
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  if (loadError) return <main className="page"><p className="error">{loadError}</p></main>;
  if (!options || !current) return <main className="page"><p className="muted">Loading settings…</p></main>;

  return (
    <main className="page">
      <form className="panel" onSubmit={handleSave}>
        <h2>LLM provider</h2>
        <p className="muted">
          Used for phrase extraction and clarification messages (<code>/query/evaluate</code>).
          Retrieval-only search does not need this.
        </p>
        <label className="field">
          <span>Provider</span>
          <select value={llmProvider} onChange={(e) => setLlmProvider(e.target.value as LlmProvider)}>
            {options.llm_providers.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Model (blank = provider default: {current.resolved_llm_model})</span>
          <input value={llmModel} onChange={(e) => setLlmModel(e.target.value)} placeholder={current.resolved_llm_model} />
        </label>
        {llmProvider === "claude" && (
          <label className="field">
            <span>Anthropic API key {current.anthropic_api_key_set && "(already set — leave blank to keep)"}</span>
            <input type="password" value={anthropicKey} onChange={(e) => setAnthropicKey(e.target.value)} placeholder="sk-ant-…" />
          </label>
        )}
        {llmProvider === "openai" && (
          <label className="field">
            <span>OpenAI API key {current.openai_api_key_set && "(already set — leave blank to keep)"}</span>
            <input type="password" value={openaiKey} onChange={(e) => setOpenaiKey(e.target.value)} placeholder="sk-…" />
          </label>
        )}
        {llmProvider === "bedrock" && (
          <>
            <label className="field">
              <span>AWS access key {current.aws_configured && "(already set — leave blank to keep)"}</span>
              <input type="password" value={awsAccessKey} onChange={(e) => setAwsAccessKey(e.target.value)} />
            </label>
            <label className="field">
              <span>AWS secret key</span>
              <input type="password" value={awsSecretKey} onChange={(e) => setAwsSecretKey(e.target.value)} />
            </label>
          </>
        )}
        {llmProvider === "ollama" && (
          <label className="field">
            <span>Ollama base URL</span>
            <input value={ollamaBaseUrl} onChange={(e) => setOllamaBaseUrl(e.target.value)} />
          </label>
        )}

        <h2>Embedding provider</h2>
        <p className="muted">
          Changing this resets the in-memory index (dimensions differ per model) — you'll need to
          re-index your catalog on the Catalog page afterward.
        </p>
        <label className="field">
          <span>Provider</span>
          <select value={embeddingProvider} onChange={(e) => setEmbeddingProvider(e.target.value as EmbeddingProvider)}>
            {options.embedding_providers.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Model (blank = provider default: {current.resolved_embedding_model})</span>
          <input value={embeddingModel} onChange={(e) => setEmbeddingModel(e.target.value)} placeholder={current.resolved_embedding_model} />
        </label>
        {embeddingProvider === "gemini" && (
          <label className="field">
            <span>Google API key {current.google_api_key_set && "(already set — leave blank to keep)"}</span>
            <input type="password" value={googleKey} onChange={(e) => setGoogleKey(e.target.value)} />
          </label>
        )}
        {embeddingProvider === "openai" && (
          <label className="field">
            <span>OpenAI API key {current.openai_api_key_set && "(already set — leave blank to keep)"}</span>
            <input type="password" value={openaiKey} onChange={(e) => setOpenaiKey(e.target.value)} />
          </label>
        )}

        <h2>Search backend</h2>
        <p className="muted">
          <code>local</code> and <code>faiss</code> need no server. <code>opensearch</code> and{" "}
          <code>chroma</code> require the extra installed (<code>pip install -e ".[opensearch]"</code>{" "}
          etc.) and a reachable server.
        </p>
        <label className="field">
          <span>Backend</span>
          <select value={searchBackend} onChange={(e) => setSearchBackend(e.target.value as SearchBackend)}>
            {options.search_backends.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </label>
        {searchBackend === "opensearch" && (
          <>
            <label className="field">
              <span>OpenSearch host</span>
              <input value={opensearchHost} onChange={(e) => setOpensearchHost(e.target.value)} />
            </label>
            <label className="field">
              <span>OpenSearch port</span>
              <input type="number" value={opensearchPort} onChange={(e) => setOpensearchPort(Number(e.target.value))} />
            </label>
          </>
        )}
        {searchBackend === "chroma" && (
          <>
            <label className="field">
              <span>Chroma host</span>
              <input value={chromaHost} onChange={(e) => setChromaHost(e.target.value)} />
            </label>
            <label className="field">
              <span>Chroma port</span>
              <input type="number" value={chromaPort} onChange={(e) => setChromaPort(Number(e.target.value))} />
            </label>
          </>
        )}

        <button className="btn" type="submit" disabled={saving}>
          {saving ? "Applying…" : "Apply settings"}
        </button>
        {saveError && <p className="error">{saveError}</p>}
        {indexResetWarning && (
          <p className="error">
            Index was reset by this change — go to the Catalog page to re-index your data.
          </p>
        )}
      </form>
    </main>
  );
}
