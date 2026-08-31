from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from shamela_rag.config import get_settings
from shamela_rag.data.discovery import BookLocation
from shamela_rag.db.engine import get_engine, get_sessionmaker
from shamela_rag.db.models import Base, Book, Chunk, Section
from shamela_rag.embeddings.provider import InMemoryEmbeddingProvider
from shamela_rag.ingestion.pipeline import IngestionService
from shamela_rag.retrieval.stable_ids import (
    format_stable_chunk_id,
    page_ordinal,
    parse_stable_chunk_id,
    resolve_stable_chunk_id,
    stable_chunk_id,
)
from shamela_rag.vectorstore.qdrant_store import QdrantStore

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "book_1021"
_BOOK_ID = 1021
_DIMS = 8


def _require_engine():
    engine = get_engine()
    try:
        with engine.connect():
            pass
    except Exception:  # noqa: BLE001
        pytest.skip("Postgres not reachable")
    return engine


def _logical_key(chunk: Chunk) -> tuple[int | None, int | None, str]:
    return (chunk.start_page_id, chunk.start_offset, chunk.content_role)


def test_format_and_parse_roundtrip() -> None:
    stable_id = format_stable_chunk_id(1021, 4471, 2)
    assert stable_id == "shamela:1021:4471:2"
    key = parse_stable_chunk_id(stable_id)
    assert (key.book_id, key.page_id, key.ordinal) == (1021, 4471, 2)


def test_parse_rejects_invalid_ids() -> None:
    with pytest.raises(ValueError, match="invalid stable chunk id"):
        parse_stable_chunk_id("chunk-5")


def test_two_chunks_on_same_page_get_distinct_ordinals() -> None:
    engine = _require_engine()
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        session.execute(delete(Chunk).where(Chunk.book_id == _BOOK_ID))
        session.execute(delete(Section).where(Section.book_id == _BOOK_ID))
        session.execute(delete(Book).where(Book.book_id == _BOOK_ID))
        session.add(Book(book_id=_BOOK_ID, title_ar="test"))
        first = Chunk(
            book_id=_BOOK_ID,
            content_role="body",
            source_text="alpha",
            start_page_id=100,
            start_offset=10,
        )
        second = Chunk(
            book_id=_BOOK_ID,
            content_role="body",
            source_text="beta",
            start_page_id=100,
            start_offset=20,
        )
        session.add_all([first, second])
        session.flush()
        assert page_ordinal(session, first) == 1
        assert page_ordinal(session, second) == 2
        assert stable_chunk_id(session, first) == "shamela:1021:100:1"
        assert stable_chunk_id(session, second) == "shamela:1021:100:2"


@pytest.fixture
def ingested_book() -> Iterator[tuple[object, BookLocation]]:
    engine = _require_engine()
    store = QdrantStore(
        url=get_settings().qdrant_url,
        collection=f"test_stable_ids_{uuid.uuid4().hex}",
        dense_dim=_DIMS,
    )
    try:
        store.client.get_collections()
    except Exception:  # noqa: BLE001
        pytest.skip("Qdrant not reachable")

    location = BookLocation(
        book_dir=_FIXTURE_DIR, book_id=_BOOK_ID, category_id=1, has_all_files=True
    )
    session_factory = get_sessionmaker(engine)
    Base.metadata.create_all(engine)
    IngestionService(
        session_factory=session_factory,
        store=store,
        embedder=InMemoryEmbeddingProvider(dims=_DIMS),
    ).ingest_book(location)

    try:
        yield session_factory, location
    finally:
        store.delete_collection()
        with Session(engine) as session, session.begin():
            session.execute(delete(Chunk).where(Chunk.book_id == _BOOK_ID))
            session.execute(delete(Section).where(Section.book_id == _BOOK_ID))
            session.execute(delete(Book).where(Book.book_id == _BOOK_ID))


def test_stable_id_survives_reingest(ingested_book: tuple[object, BookLocation]) -> None:
    session_factory, location = ingested_book
    store = QdrantStore(
        url=get_settings().qdrant_url,
        collection=f"test_stable_ids_re_{uuid.uuid4().hex}",
        dense_dim=_DIMS,
    )
    service = IngestionService(
        session_factory=session_factory,
        store=store,
        embedder=InMemoryEmbeddingProvider(dims=_DIMS),
    )

    with session_factory() as session:
        first_pass = list(session.execute(select(Chunk).where(Chunk.book_id == _BOOK_ID)).scalars())
        first_ids = {chunk.id for chunk in first_pass}
        first_stable = {
            _logical_key(chunk): stable_chunk_id(session, chunk) for chunk in first_pass
        }

    service.ingest_book(location)

    with session_factory() as session:
        second_pass = list(
            session.execute(select(Chunk).where(Chunk.book_id == _BOOK_ID)).scalars()
        )
        second_ids = {chunk.id for chunk in second_pass}
        second_stable = {
            _logical_key(chunk): stable_chunk_id(session, chunk) for chunk in second_pass
        }

    assert first_ids != second_ids
    assert first_stable == second_stable
    assert first_stable

    sample_key = next(iter(first_stable))
    with session_factory() as session:
        resolved = resolve_stable_chunk_id(session, first_stable[sample_key])
        assert resolved is not None
        assert _logical_key(resolved) == sample_key
