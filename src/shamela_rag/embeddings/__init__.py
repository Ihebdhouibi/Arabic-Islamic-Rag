"""Dense embedding providers (interface, in-memory test double, Qwen3 and BGE-M3 backends).

Also exposes the surface-form BM25 sparse encoder (``Bm25Encoder``).
"""

from __future__ import annotations

from shamela_rag.embeddings.bge_m3 import (
    BGE_M3_DIMS,
    BGE_M3_MODEL_ID,
    BgeM3EmbeddingProvider,
    SparseEmbedding,
    lexical_weights_to_sparse,
)
from shamela_rag.embeddings.bm25 import Bm25Encoder, SparseVector, tokenize
from shamela_rag.embeddings.provider import EmbeddingProvider, InMemoryEmbeddingProvider
from shamela_rag.embeddings.qwen import (
    DEFAULT_EMBEDDING_DIMS,
    DEFAULT_TASK_DESCRIPTION,
    QWEN3_EMBEDDING_GGUF_Q4_K_M,
    QWEN3_EMBEDDING_GGUF_Q8_0,
    QWEN3_EMBEDDING_GGUF_REPO_ID,
    QWEN3_EMBEDDING_MODEL_ID,
    SUPPORTED_QUANTIZATIONS,
    QuantizationMode,
    Qwen3EmbeddingProvider,
    download_qwen_gguf,
    format_qwen_query,
)

__all__ = [
    "BGE_M3_DIMS",
    "BGE_M3_MODEL_ID",
    "BgeM3EmbeddingProvider",
    "Bm25Encoder",
    "DEFAULT_EMBEDDING_DIMS",
    "DEFAULT_TASK_DESCRIPTION",
    "EmbeddingProvider",
    "InMemoryEmbeddingProvider",
    "QWEN3_EMBEDDING_GGUF_Q4_K_M",
    "QWEN3_EMBEDDING_GGUF_Q8_0",
    "QWEN3_EMBEDDING_GGUF_REPO_ID",
    "QWEN3_EMBEDDING_MODEL_ID",
    "QuantizationMode",
    "Qwen3EmbeddingProvider",
    "SUPPORTED_QUANTIZATIONS",
    "SparseEmbedding",
    "SparseVector",
    "download_qwen_gguf",
    "format_qwen_query",
    "lexical_weights_to_sparse",
    "tokenize",
]
