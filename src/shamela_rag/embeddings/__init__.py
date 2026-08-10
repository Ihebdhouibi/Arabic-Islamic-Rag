"""Dense embedding providers (interface, in-memory test double, Qwen3 and BGE-M3 backends)."""

from __future__ import annotations

from shamela_rag.embeddings.bge_m3 import (
    BGE_M3_DIMS,
    BGE_M3_MODEL_ID,
    BgeM3EmbeddingProvider,
    SparseEmbedding,
    lexical_weights_to_sparse,
)
from shamela_rag.embeddings.provider import EmbeddingProvider, InMemoryEmbeddingProvider
from shamela_rag.embeddings.qwen import (
    DEFAULT_EMBEDDING_DIMS,
    DEFAULT_TASK_DESCRIPTION,
    QWEN3_EMBEDDING_MODEL_ID,
    Qwen3EmbeddingProvider,
    format_qwen_query,
)

__all__ = [
    "BGE_M3_DIMS",
    "BGE_M3_MODEL_ID",
    "BgeM3EmbeddingProvider",
    "DEFAULT_EMBEDDING_DIMS",
    "DEFAULT_TASK_DESCRIPTION",
    "EmbeddingProvider",
    "InMemoryEmbeddingProvider",
    "QWEN3_EMBEDDING_MODEL_ID",
    "Qwen3EmbeddingProvider",
    "SparseEmbedding",
    "format_qwen_query",
    "lexical_weights_to_sparse",
]
