"""Assemble the real general-module services from configuration (M7-05).

``build_general_qa_service`` wires the production pipeline: a dense embedding backend + Qdrant + a
persisted surface-BM25 encoder + a cross-encoder reranker + Postgres-backed context expansion +
answer generation. Heavy backends load lazily and can be injected (tests pass offline doubles).
"""

from __future__ import annotations

from pathlib import Path
from typing import assert_never

from shamela_rag.config import Settings, get_settings
from shamela_rag.embeddings.bm25 import Bm25Encoder
from shamela_rag.embeddings.provider import EmbeddingProvider
from shamela_rag.embeddings.root_field import RootExpansionEncoder
from shamela_rag.generation.provider import GenerationProvider
from shamela_rag.generation.service import GeneralQAService
from shamela_rag.retrieval.rerank import Reranker
from shamela_rag.retrieval.translate import Translator


def build_embedder(model: str | None) -> EmbeddingProvider:
    """Construct a dense embedding provider by name (``bge-m3`` default, or ``qwen3``)."""
    if (model or "bge-m3") == "qwen3":
        from shamela_rag.embeddings.qwen import Qwen3EmbeddingProvider

        return Qwen3EmbeddingProvider()
    from shamela_rag.embeddings.bge_m3 import BgeM3EmbeddingProvider

    return BgeM3EmbeddingProvider()


def build_generation_provider() -> GenerationProvider:
    """In-memory stub, llama.cpp GGUF, local Ollama, or a hosted OpenAI-compatible API —
    from ``SHAMELA_LLM_*`` settings."""
    settings = get_settings()
    backend = settings.llm_backend
    if backend == "memory":
        from shamela_rag.generation.provider import InMemoryGenerationProvider

        return InMemoryGenerationProvider()
    if backend == "llamacpp":
        if settings.llm_gguf_path is None:
            raise ValueError("SHAMELA_LLM_GGUF_PATH is required when SHAMELA_LLM_BACKEND=llamacpp")
        from shamela_rag.generation.local import LlamaCppGenerationProvider

        return LlamaCppGenerationProvider(
            settings.llm_gguf_path,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            n_ctx=settings.llm_n_ctx,
            n_threads=settings.llm_n_threads,
            n_gpu_layers=settings.llm_n_gpu_layers,
        )
    if backend == "ollama":
        if not settings.llm_ollama_model.strip():
            raise ValueError("SHAMELA_LLM_OLLAMA_MODEL is required when SHAMELA_LLM_BACKEND=ollama")
        from shamela_rag.generation.local import OllamaGenerationProvider

        return OllamaGenerationProvider(
            settings.llm_ollama_model,
            base_url=settings.llm_ollama_url,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
    if backend == "openai_compatible":
        if not settings.llm_api_key.strip():
            raise ValueError(
                "SHAMELA_LLM_API_KEY is required when SHAMELA_LLM_BACKEND=openai_compatible"
            )
        if not settings.llm_api_model.strip():
            raise ValueError(
                "SHAMELA_LLM_API_MODEL is required when SHAMELA_LLM_BACKEND=openai_compatible"
            )
        from shamela_rag.generation.local import OpenAICompatibleGenerationProvider

        return OpenAICompatibleGenerationProvider(
            settings.llm_api_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_api_base_url,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
    assert_never(backend)


def build_general_qa_service(
    *,
    model: str = "bge-m3",
    embedder: EmbeddingProvider | None = None,
    reranker: Reranker | None = None,
    sparse_encoder: Bm25Encoder | None = None,
    translator: Translator | None = None,
    generation_provider: GenerationProvider | None = None,
    root_encoder: RootExpansionEncoder | None = None,
) -> GeneralQAService:
    from shamela_rag.db.engine import get_sessionmaker
    from shamela_rag.generation.answer import AnswerAssembler
    from shamela_rag.retrieval.dense import DenseRetriever
    from shamela_rag.retrieval.expand import ContextExpander
    from shamela_rag.retrieval.service import RetrievalConfig, RetrievalService
    from shamela_rag.retrieval.sparse import SparseRetriever
    from shamela_rag.retrieval.translate import InMemoryTranslator
    from shamela_rag.vectorstore.qdrant_store import ROOT_VECTOR_NAME, QdrantStore

    settings = get_settings()
    dense = embedder or build_embedder(model)
    store = QdrantStore(
        url=settings.qdrant_url, collection=settings.qdrant_collection, dense_dim=dense.dims
    )
    encoder = sparse_encoder or _load_bm25(settings.bm25_state_path)
    cross = reranker or _build_reranker()
    session_factory = get_sessionmaker()
    root_retriever = None
    if settings.root_expansion_enabled:
        loaded_root = root_encoder or _load_root_expansion(settings)
        root_retriever = SparseRetriever(
            encoder=loaded_root, store=store, vector_name=ROOT_VECTOR_NAME
        )

    retrieval = RetrievalService(
        translator=translator or InMemoryTranslator(),
        dense_retriever=DenseRetriever(embedder=dense, store=store),
        sparse_retriever=SparseRetriever(encoder=encoder, store=store),
        reranker=cross,
        expander=ContextExpander(session_factory),
        session_factory=session_factory,
        config=RetrievalConfig(use_root_expansion=root_retriever is not None),
        root_retriever=root_retriever,
    )
    assembler = AnswerAssembler(generation_provider or build_generation_provider())
    return GeneralQAService(retrieval_service=retrieval, assembler=assembler)


def _load_bm25(path: Path) -> Bm25Encoder:
    if not path.exists():
        raise FileNotFoundError(
            f"BM25 state not found at {path}; run 'shamela-rag build-bm25' first"
        )
    return Bm25Encoder.load(path)


def _load_root_expansion(settings: Settings) -> RootExpansionEncoder:
    from shamela_rag.data.root_dictionary import load_root_dictionary

    path = settings.root_expansion_state_path
    if not path.exists():
        raise FileNotFoundError(
            f"Root expansion state not found at {path}; "
            "ingest with SHAMELA_ROOT_EXPANSION_ENABLED=true first"
        )
    dictionary = load_root_dictionary(settings.resolved_root_dictionary_path)
    return RootExpansionEncoder.load(path, dictionary)


def _build_reranker() -> Reranker:
    from shamela_rag.retrieval.rerank import CrossEncoderReranker

    return CrossEncoderReranker()
