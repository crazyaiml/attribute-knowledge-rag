import type {
  IngestResult,
  PhraseDecision,
  ProvidersHealth,
  QueryResult,
  SettingsOptions,
  SettingsPublic,
  SettingsUpdate,
  SettingsUpdateResponse,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8080";

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export async function getProviders(): Promise<ProvidersHealth> {
  const res = await fetch(`${BASE}/health/providers`);
  return asJson(res);
}

export async function ingestSampleAttributes(): Promise<IngestResult> {
  const csv = await fetch("/sample_attributes.csv").then((r) => r.blob());
  const form = new FormData();
  form.append("file", csv, "sample_attributes.csv");
  form.append("index_name", "attributes_demo");
  const res = await fetch(`${BASE}/ingest/upload`, { method: "POST", body: form });
  return asJson(res);
}

export async function evaluateQuery(input: string, channel?: string): Promise<QueryResult> {
  const res = await fetch(`${BASE}/query/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input, channel: channel || undefined }),
  });
  return asJson(res);
}

export async function clarifyPhrase(
  sessionId: string,
  phrase: string,
  chosenAttributeId: string
): Promise<PhraseDecision> {
  const res = await fetch(`${BASE}/query/clarify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      phrase,
      chosen_attribute_id: chosenAttributeId,
    }),
  });
  return asJson(res);
}

const NDJSON_EXTENSIONS = [".ndjson", ".jsonl"];

export async function ingestFile(file: File, indexName?: string): Promise<IngestResult> {
  const isNdjson = NDJSON_EXTENSIONS.some((ext) => file.name.toLowerCase().endsWith(ext));
  const form = new FormData();
  form.append("file", file, file.name);
  if (indexName) form.append("index_name", indexName);
  const endpoint = isNdjson ? "/ingest/from-ndjson" : "/ingest/upload";
  const res = await fetch(`${BASE}${endpoint}`, { method: "POST", body: form });
  return asJson(res);
}

export async function listIndexes(): Promise<string[]> {
  const res = await fetch(`${BASE}/ingest/indexes`);
  return asJson(res);
}

export async function searchPhrases(phrases: string[], topK = 10): Promise<PhraseDecision[]> {
  const res = await fetch(`${BASE}/query/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ phrases, top_k: topK }),
  });
  return asJson(res);
}

export async function getSettingsOptions(): Promise<SettingsOptions> {
  const res = await fetch(`${BASE}/settings/options`);
  return asJson(res);
}

export async function getSettingsPublic(): Promise<SettingsPublic> {
  const res = await fetch(`${BASE}/settings`);
  return asJson(res);
}

export async function updateSettings(update: SettingsUpdate): Promise<SettingsUpdateResponse> {
  const res = await fetch(`${BASE}/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(update),
  });
  return asJson(res);
}
