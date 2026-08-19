"""Wire a ``GeneralQAService`` for the Streamlit demo (M7-04)."""

from __future__ import annotations

from shamela_rag.config import get_settings
from shamela_rag.db.engine import get_sessionmaker
from shamela_rag.embeddings.bm25 import Bm25Encoder
from shamela_rag.embeddings.provider import EmbeddingProvider
from shamela_rag.generation.answer import AnswerAssembler
from shamela_rag.generation.provider import GenerationProvider
from shamela_rag.generation.service import GeneralQAService
from shamela_rag.retrieval.dense import DenseRetriever
from shamela_rag.retrieval.expand import ContextExpander
from shamela_rag.retrieval.rerank import CrossEncoderReranker, Reranker
from shamela_rag.retrieval.service import RetrievalConfig, RetrievalService
from shamela_rag.retrieval.sparse import SparseRetriever
from shamela_rag.retrieval.translate import QueryLanguage, Translator
from shamela_rag.vectorstore.qdrant_store import ROOT_VECTOR_NAME, QdrantStore


class _PassthroughTranslator(Translator):
    def translate(
        self,
        text: str,
        *,
        source: QueryLanguage,
        target: QueryLanguage,
    ) -> str:
        return text


def _build_embedder(model: str | None) -> EmbeddingProvider:
    if (model or "bge-m3") == "qwen3":
        from shamela_rag.embeddings.qwen import Qwen3EmbeddingProvider

        return Qwen3EmbeddingProvider()
    from shamela_rag.embeddings.bge_m3 import BgeM3EmbeddingProvider

    return BgeM3EmbeddingProvider()


def build_general_qa_service(
    *,
    generation_provider: GenerationProvider,
    translator: Translator | None = None,
    reranker: Reranker | None = None,
    embedder: EmbeddingProvider | None = None,
    model: str | None = None,
) -> GeneralQAService:
    settings = get_settings()
    state_path = settings.bm25_state_path
    if not state_path.exists():
        raise FileNotFoundError(
            f"BM25 state not found at {state_path}; run `shamela-rag build-bm25` first"
        )

    resolved_embedder = embedder if embedder is not None else _build_embedder(model)
    resolved_translator = translator if translator is not None else _PassthroughTranslator()
    resolved_reranker = (
        reranker if reranker is not None else CrossEncoderReranker(model_id=settings.reranker_model)
    )
    encoder = Bm25Encoder.load(state_path)
    store = QdrantStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        dense_dim=resolved_embedder.dims,
    )
    session_factory = get_sessionmaker()
    root_retriever = None
    if settings.root_expansion_enabled:
        from shamela_rag.factory import _load_root_expansion

        loaded_root = _load_root_expansion(settings)
        root_retriever = SparseRetriever(
            encoder=loaded_root, store=store, vector_name=ROOT_VECTOR_NAME
        )

    retrieval = RetrievalService(
        translator=resolved_translator,
        dense_retriever=DenseRetriever(embedder=resolved_embedder, store=store),
        sparse_retriever=SparseRetriever(encoder=encoder, store=store),
        reranker=resolved_reranker,
        expander=ContextExpander(session_factory),
        session_factory=session_factory,
        config=RetrievalConfig(use_root_expansion=root_retriever is not None),
        root_retriever=root_retriever,
    )
    return GeneralQAService(
        retrieval_service=retrieval,
        assembler=AnswerAssembler(generation_provider),
    )
