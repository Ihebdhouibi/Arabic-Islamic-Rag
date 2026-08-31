from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy import delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from shamela_rag.chunking.orchestrator import chunk_book
from shamela_rag.config import get_settings
from shamela_rag.data.discovery import BookLocation
from shamela_rag.db.engine import get_engine, get_sessionmaker
from shamela_rag.db.models import Base, Book, Chunk, Section
from shamela_rag.embeddings.bm25 import Bm25Encoder
from shamela_rag.embeddings.provider import InMemoryEmbeddingProvider
from shamela_rag.generation.answer import AnswerAssembler
from shamela_rag.generation.provider import InMemoryGenerationProvider
from shamela_rag.generation.service import GeneralQAService
from shamela_rag.ingestion.pipeline import IngestionService
from shamela_rag.retrieval.dense import DenseRetriever
from shamela_rag.retrieval.expand import ContextExpander
from shamela_rag.retrieval.rerank import LexicalOverlapReranker
from shamela_rag.retrieval.service import RetrievalConfig, RetrievalService
from shamela_rag.retrieval.sparse import SparseRetriever
from shamela_rag.retrieval.translate import InMemoryTranslator
from shamela_rag.retrieval.stable_ids import resolve_stable_chunk_id
from shamela_rag.vectorstore.qdrant_store import QdrantStore

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "book_1021"
_BOOK_ID = 1021
_DIMS = 8


@dataclass
class _World:
    qa: GeneralQAService
    engine: Engine


def _location() -> BookLocation:
    return BookLocation(book_dir=_FIXTURE_DIR, book_id=_BOOK_ID, category_id=1, has_all_files=True)


@pytest.fixture
def world() -> Iterator[_World]:
    engine = get_engine()
    try:
        with engine.connect():
            pass
    except Exception:  # noqa: BLE001 - Postgres unavailable
        pytest.skip("Postgres not reachable")

    store = QdrantStore(
        url=get_settings().qdrant_url, collection=f"test_qa_{uuid.uuid4().hex}", dense_dim=_DIMS
    )
    try:
        store.client.get_collections()
    except Exception:  # noqa: BLE001 - Qdrant unavailable
        pytest.skip("Qdrant not reachable")

    Base.metadata.create_all(engine)
    embedder = InMemoryEmbeddingProvider(dims=_DIMS)
    session_factory = get_sessionmaker(engine)
    IngestionService(session_factory=session_factory, store=store, embedder=embedder).ingest_book(
        _location()
    )

    encoder = Bm25Encoder().fit(chunk.retrieval_text for chunk in chunk_book(_FIXTURE_DIR).chunks)
    retrieval = RetrievalService(
        translator=InMemoryTranslator(),
        dense_retriever=DenseRetriever(embedder=embedder, store=store),
        sparse_retriever=SparseRetriever(encoder=encoder, store=store),
        reranker=LexicalOverlapReranker(),
        expander=ContextExpander(session_factory),
        session_factory=session_factory,
        config=RetrievalConfig(final_k=5),
    )
    qa = GeneralQAService(
        retrieval_service=retrieval, assembler=AnswerAssembler(InMemoryGenerationProvider())
    )
    try:
        yield _World(qa=qa, engine=engine)
    finally:
        store.delete_collection()
        with Session(engine) as session, session.begin():
            session.execute(delete(Chunk).where(Chunk.book_id == _BOOK_ID))
            session.execute(delete(Section).where(Section.book_id == _BOOK_ID))
            session.execute(delete(Book).where(Book.book_id == _BOOK_ID))


def _query(engine: Engine) -> str:
    with Session(engine) as session:
        row = session.execute(
            select(Chunk.retrieval_text)
            .where(Chunk.book_id == _BOOK_ID)
            .order_by(Chunk.token_count.desc().nulls_last(), Chunk.id)
            .limit(1)
        ).one()
    return " ".join((row.retrieval_text or "").split()[:6])


def test_answer_general_question_is_cited_end_to_end(world: _World) -> None:
    answer = world.qa.answer(_query(world.engine))

    assert answer.text != ""
    assert not answer.deflected
    assert answer.citations, "expected at least one citation"

    with Session(world.engine) as session:
        for citation in answer.citations:
            chunk = session.get(Chunk, citation.chunk_id)
            assert chunk is not None, f"citation {citation.chunk_id} does not resolve to a chunk"
            assert chunk.book_id == _BOOK_ID
            resolved = resolve_stable_chunk_id(session, citation.id)
            assert resolved is not None
            assert resolved.id == chunk.id


def test_answer_with_retrieval_returns_passage_scores(world: _World) -> None:
    answer, passages = world.qa.answer_with_retrieval(_query(world.engine))

    assert not answer.deflected
    assert passages
    assert all(isinstance(passage.score, float) for passage in passages)
    assert {citation.chunk_id for citation in answer.citations} == {
        passage.hit_chunk_id for passage in passages
    }
