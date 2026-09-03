"""Central application configuration.

All settings load from environment variables (prefix ``SHAMELA_``) or a local ``.env`` file.
Defaults match the docker-compose dev profile, so the app runs against local services with no
``.env`` required.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SHAMELA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Postgres (relational / provenance store). Host port 5433 avoids clashing with a native
    # Postgres commonly bound to 5432; the container still listens on 5432 internally.
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_user: str = "shamela"
    postgres_password: str = "shamela"
    postgres_db: str = "shamela_rag"

    # Qdrant (vector store)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "shamela_general"
    qdrant_dense_dim: int = 1024  # BGE-M3 default; Qwen3-8B would be larger

    # Models (final dense model chosen by the M6 benchmark)
    dense_embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Dense embedding backend: local weights or OpenRouter embeddings API.
    embedding_backend: Literal["local", "openrouter"] = "local"
    embedding_api_base_url: str = "https://openrouter.ai/api/v1"
    embedding_api_key: str = ""
    embedding_api_batch_size: int = Field(default=32, gt=0)

    # Local generation (default: in-memory stub; CI stays offline)
    llm_backend: Literal["memory", "llamacpp", "ollama", "openai_compatible"] = "memory"
    llm_gguf_path: Path | None = None
    llm_ollama_url: str = "http://localhost:11434"
    llm_ollama_model: str = ""
    llm_max_tokens: int = Field(default=512, gt=0)
    llm_temperature: float = Field(default=0.1, ge=0)
    llm_n_ctx: int = Field(default=4096, gt=0)
    llm_n_threads: int | None = None
    llm_n_gpu_layers: int = Field(default=0, ge=0)

    # Hosted OpenAI-compatible generation API (Together.ai, DeepSeek, DashScope, ...)
    llm_api_base_url: str = "https://api.together.xyz/v1"
    llm_api_key: str = ""
    llm_api_model: str = ""

    # Query translation for retrieval (default: offline in-memory stub; CI stays offline).
    translator_backend: Literal["memory", "openai_compatible"] = "memory"
    translator_api_model: str = ""
    translator_max_tokens: int = Field(default=256, gt=0)
    translator_timeout_seconds: float = Field(default=30.0, gt=0)

    # Corpus location (the extracted Shamela4 dataset root)
    corpus_root: Path = Path(".")

    # Persisted surface-BM25 encoder state (vocab/IDF), shared by ingestion and query retrieval.
    bm25_state_path: Path = Path("bm25_state.json")

    # Root-expansion sparse field (off by default; low-weight extra arm).
    root_expansion_enabled: bool = False
    root_expansion_weight: float = Field(default=0.25, ge=0)
    root_dictionary_path: Path = Path("_meta/root_dictionary.jsonl")
    root_expansion_state_path: Path = Path("root_expansion_state.json")

    # Chunking starting values (tuned in M6; see review_general_module_chunking_embeddings_brief.md)
    chunk_min_tokens: int = Field(default=128, gt=0)
    chunk_max_tokens: int = Field(default=768, gt=0)
    chunk_split_target_tokens: int = Field(default=448, gt=0)
    chunk_overlap_tokens: int = Field(default=64, ge=0)

    @property
    def resolved_root_dictionary_path(self) -> Path:
        if self.root_dictionary_path.is_absolute():
            return self.root_dictionary_path
        return self.corpus_root / self.root_dictionary_path

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sqlalchemy_dsn(self) -> str:
        return self.postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
