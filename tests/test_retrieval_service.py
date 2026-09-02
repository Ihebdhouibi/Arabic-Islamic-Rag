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
from shamela_rag.ingestion.pipeline import IngestionService
from shamela_rag.retrieval.dense import DenseRetriever
from shamela_rag.retrieval.expand import ContextExpander
from shamela_rag.retrieval.filters import RetrievalFilter
from shamela_rag.retrieval.rerank import LexicalOverlapReranker
from shamela_rag.retrieval.service import RetrievalConfig, RetrievalService
from shamela_rag.retrieval.sparse import SparseRetriever
from shamela_rag.retrieval.translate import InMemoryTranslator
from shamela_rag.vectorstore.qdrant_store import QdrantStore

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "book_1021"
_BOOK_ID = 1021
_DIMS = 8


@dataclass
class _World:
    service: RetrievalService
    engine: Engine


def _location() -> BookLocation:
    return BookLocation(book_dir=_FIXTURE_DIR, book_id=_BOOK_ID, category_id=1, has_all_files=True)


@pytest.fixture
def world() -> Iterator[_World]:
    engine = get_engine()
    try:
        with engine.connect():
            pass
    except Exception:  # noqa: BLE001 - any connection error means Postgres is unavailable
        pytest.skip("Postgres not reachable")

    store = QdrantStore(
        url=get_settings().qdrant_url,
        collection=f"test_service_{uuid.uuid4().hex}",
        dense_dim=_DIMS,
    )
    try:
        store.client.get_collections()
    except Exception:  # noqa: BLE001 - any connection error means Qdrant is unavailable
        pytest.skip("Qdrant not reachable")

    Base.metadata.create_all(engine)
    embedder = InMemoryEmbeddingProvider(dims=_DIMS)
    session_factory = get_sessionmaker(engine)
    IngestionService(session_factory=session_factory, store=store, embedder=embedder).ingest_book(
        _location()
    )

    # Reconstruct the same BM25 vocabulary/idf the ingestion fitted (deterministic on the chunks).
    encoder = Bm25Encoder().fit(chunk.retrieval_text for chunk in chunk_book(_FIXTURE_DIR).chunks)
    service = RetrievalService(
        translator=InMemoryTranslator(),
        dense_retriever=DenseRetriever(embedder=embedder, store=store),
        sparse_retriever=SparseRetriever(encoder=encoder, store=store),
        reranker=LexicalOverlapReranker(),
        expander=ContextExpander(session_factory),
        session_factory=session_factory,
        config=RetrievalConfig(final_k=5, expansion=RetrievalConfig().expansion),
    )
    try:
        yield _World(service=service, engine=engine)
    finally:
        store.delete_collection()
        with Session(engine) as session, session.begin():
            session.execute(delete(Chunk).where(Chunk.book_id == _BOOK_ID))
            session.execute(delete(Section).where(Section.book_id == _BOOK_ID))
            session.execute(delete(Book).where(Book.book_id == _BOOK_ID))


def _target(engine: Engine) -> tuple[int, str, str]:
    """Return (chunk_id, query, book_title) for the largest chunk of the fixture."""
    with Session(engine) as session:
        row = session.execute(
            select(Chunk.id, Chunk.retrieval_text)
            .where(Chunk.book_id == _BOOK_ID)
            .order_by(Chunk.token_count.desc().nulls_last(), Chunk.id)
            .limit(1)
        ).one()
        book_title = session.get(Book, _BOOK_ID).title_ar  # type: ignore[union-attr]
    query = " ".join((row.retrieval_text or "").split()[:6])
    return row.id, query, book_title


def test_retrieve_returns_cite_ready_passages(world: _World) -> None:
    target_id, query, book_title = _target(world.engine)

    passages = world.service.retrieve(query, k=5)

    assert passages, "expected at least one passage"
    assert target_id in {passage.hit_chunk_id for passage in passages}
    top = passages[0]
    assert top.text.strip() != ""
    assert top.payload["book_title"] == book_title
    assert top.payload["book_id"] == _BOOK_ID
    sources = top.payload["retrieval_sources"]
    assert isinstance(sources, list) and sources
    assert "rerank" not in sources  # lexical fallback does not tag rerank


def test_retrieve_book_filter_scopes_results(world: _World) -> None:
    _target_id, query, _title = _target(world.engine)

    in_book = world.service.retrieve(query, filters=RetrievalFilter(book_id=_BOOK_ID))
    assert in_book

    other_book = world.service.retrieve(query, filters=RetrievalFilter(book_id=999_999))
    assert other_book == []


def test_retrieve_empty_question_is_rejected(world: _World) -> None:
    with pytest.raises(ValueError, match="question"):
        world.service.retrieve("   ")


def test_english_query_matches_arabic_with_mapped_translator(world: _World) -> None:
    target_id, arabic_query, _ = _target(world.engine)
    english = "What does this passage say?"
    world.service._translator = InMemoryTranslator({english: arabic_query})

    english_hits = world.service.retrieve(english, k=5)
    arabic_hits = world.service.retrieve(arabic_query, k=5)

    assert target_id in {passage.hit_chunk_id for passage in english_hits}
    assert english_hits[0].hit_chunk_id == arabic_hits[0].hit_chunk_id
