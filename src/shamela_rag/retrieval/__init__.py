"""Retrieval pipeline pieces (query prep, dense/sparse search, fusion, expansion)."""

from __future__ import annotations

from shamela_rag.retrieval.dense import DenseRetriever
from shamela_rag.retrieval.expand import (
    ChunkNotFoundError,
    ContextExpander,
    ExpandedChunkPart,
    ExpandedPassage,
    ExpandMode,
    ExpansionConfig,
)
from shamela_rag.retrieval.filters import RetrievalFilter
from shamela_rag.retrieval.fusion import FusedChunk, reciprocal_rank_fusion
from shamela_rag.retrieval.rerank import (
    CrossEncoderReranker,
    LexicalOverlapReranker,
    RerankCandidate,
    RerankedChunk,
    Reranker,
)
from shamela_rag.retrieval.results import RetrievedChunk
from shamela_rag.retrieval.sparse import SparseRetriever
from shamela_rag.retrieval.translate import (
    InMemoryTranslator,
    PreparedQuery,
    QueryLanguage,
    Translator,
    detect_query_language,
    prepare_retrieval_query,
)

__all__ = [
    "ChunkNotFoundError",
    "ContextExpander",
    "CrossEncoderReranker",
    "DenseRetriever",
    "ExpandMode",
    "ExpandedChunkPart",
    "ExpandedPassage",
    "ExpansionConfig",
    "FusedChunk",
    "InMemoryTranslator",
    "LexicalOverlapReranker",
    "PreparedQuery",
    "QueryLanguage",
    "RerankCandidate",
    "RerankedChunk",
    "Reranker",
    "RetrievalFilter",
    "RetrievedChunk",
    "SparseRetriever",
    "Translator",
    "detect_query_language",
    "prepare_retrieval_query",
    "reciprocal_rank_fusion",
]
