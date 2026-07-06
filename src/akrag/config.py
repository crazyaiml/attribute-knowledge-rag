from __future__ import annotations

from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["claude", "openai", "bedrock", "ollama"] = "ollama"
    llm_model: str = ""  # empty → use provider default

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    ollama_base_url: str = "http://localhost:11434"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_provider: Literal["gemini", "openai", "sentence", "bedrock"] = "sentence"
    embedding_model: str = ""  # empty → use provider default

    google_api_key: str = ""

    # ── Search ────────────────────────────────────────────────────────────────
    search_backend: Literal["local", "opensearch", "faiss", "chroma"] = "local"

    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_user: str = "admin"
    opensearch_pass: str = "admin"
    opensearch_use_ssl: bool = False
    opensearch_index_prefix: str = "attributes"

    chroma_host: str = "localhost"
    chroma_port: int = 8000

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    log_level: str = "info"

    # ── Search tuning ─────────────────────────────────────────────────────────
    top_k: int = 10
    exact_threshold: float = 0.92
    near_threshold: float = 0.75
    rrf_k: int = 60

    @property
    def resolved_llm_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        defaults = {
            "claude": "claude-sonnet-4-6",
            "openai": "gpt-4o",
            "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "ollama": "llama3.2",
        }
        return defaults[self.llm_provider]

    @property
    def resolved_embedding_model(self) -> str:
        if self.embedding_model:
            return self.embedding_model
        defaults = {
            "gemini": "models/text-embedding-004",
            "openai": "text-embedding-3-small",
            "sentence": "all-MiniLM-L6-v2",
            "bedrock": "amazon.titan-embed-text-v2:0",
        }
        return defaults[self.embedding_provider]


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def update_settings(**overrides: object) -> Settings:
    """Merge `overrides` onto the current settings and re-validate. Used by the
    /settings API so the demo UI can switch providers without restarting the process."""
    global _settings
    current = get_settings()
    merged = {**current.model_dump(), **{k: v for k, v in overrides.items() if v is not None}}
    _settings = Settings(**merged)
    return _settings
