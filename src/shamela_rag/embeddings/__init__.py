"""Dense embedding providers (interface, in-memory test double, Qwen3 backend)."""

from __future__ import annotations

from shamela_rag.embeddings.provider import EmbeddingProvider, InMemoryEmbeddingProvider
from shamela_rag.embeddings.qwen import (
    DEFAULT_EMBEDDING_DIMS,
    DEFAULT_TASK_DESCRIPTION,
    QWEN3_EMBEDDING_MODEL_ID,
    Qwen3EmbeddingProvider,
    format_qwen_query,
)

__all__ = [
    "DEFAULT_EMBEDDING_DIMS",
    "DEFAULT_TASK_DESCRIPTION",
    "EmbeddingProvider",
    "InMemoryEmbeddingProvider",
    "QWEN3_EMBEDDING_MODEL_ID",
    "Qwen3EmbeddingProvider",
    "format_qwen_query",
]
