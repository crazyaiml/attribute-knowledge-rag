export type DecisionOutcome = "exact" | "near" | "ambiguous" | "none";

export interface GovernanceMetadata {
  pii: boolean;
  sensitive: boolean;
  allowed_channels: string[];
  eligible_domains: string[];
}

export interface AttributeSummary {
  attribute_id: string;
  business_name: string;
  type?: string;
  technical_field?: string;
  domain?: string;
  definition?: string;
  governance?: GovernanceMetadata;
}

export interface SearchResult {
  attribute: AttributeSummary;
  bm25_score?: number;
  vector_score?: number;
  rrf_score?: number;
}

export interface PhraseDecision {
  phrase: string;
  outcome: DecisionOutcome;
  selected?: AttributeSummary | null;
  options: SearchResult[];
  clarification_message?: string | null;
}

export interface DSLFilter {
  attribute_id: string;
  business_name: string;
  technical_field: string;
  operator?: string | null;
  value?: unknown;
  source_phrase: string;
}

export interface QueryResult {
  input: string;
  phrases: string[];
  filters: DSLFilter[];
  unresolved: PhraseDecision[];
  dsl?: Record<string, unknown> | null;
}

export interface ProvidersHealth {
  llm: string;
  llm_model: string;
  embedding: string;
  embedding_model: string;
  embedding_dim: number;
  search: string;
  indexes: string[];
}

export interface IngestResult {
  total: number;
  indexed: number;
  failed: number;
  errors: string[];
  index_name?: string;
}

export type LlmProvider = "claude" | "openai" | "bedrock" | "ollama";
export type EmbeddingProvider = "gemini" | "openai" | "sentence" | "bedrock";
export type SearchBackend = "local" | "opensearch" | "faiss" | "chroma";

export interface SettingsOptions {
  llm_providers: LlmProvider[];
  embedding_providers: EmbeddingProvider[];
  search_backends: SearchBackend[];
}

export interface SettingsPublic {
  llm_provider: LlmProvider;
  llm_model: string;
  resolved_llm_model: string;
  anthropic_api_key_set: boolean;
  openai_api_key_set: boolean;
  aws_configured: boolean;
  ollama_base_url: string;

  embedding_provider: EmbeddingProvider;
  embedding_model: string;
  resolved_embedding_model: string;
  google_api_key_set: boolean;

  search_backend: SearchBackend;
  opensearch_host: string;
  opensearch_port: number;
  chroma_host: string;
  chroma_port: number;

  exact_threshold: number;
  near_threshold: number;
  rrf_k: number;
  top_k: number;
}

export interface SettingsUpdate {
  llm_provider?: LlmProvider;
  llm_model?: string;
  anthropic_api_key?: string;
  openai_api_key?: string;
  aws_access_key_id?: string;
  aws_secret_access_key?: string;
  aws_region?: string;
  ollama_base_url?: string;

  embedding_provider?: EmbeddingProvider;
  embedding_model?: string;
  google_api_key?: string;

  search_backend?: SearchBackend;
  opensearch_host?: string;
  opensearch_port?: number;
  opensearch_user?: string;
  opensearch_pass?: string;
  opensearch_use_ssl?: boolean;
  chroma_host?: string;
  chroma_port?: number;
}

export interface SettingsUpdateResponse {
  settings: SettingsPublic;
  index_reset: boolean;
}
