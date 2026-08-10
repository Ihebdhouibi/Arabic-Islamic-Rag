from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from shamela_rag.data.discovery import BookLocation
from shamela_rag.db.engine import get_engine, get_sessionmaker
from shamela_rag.db.models import Base, Book, Chunk, Section
from shamela_rag.embeddings.provider import InMemoryEmbeddingProvider
from shamela_rag.ingestion.pipeline import IngestionService
from shamela_rag.vectorstore.qdrant_store import QdrantStore

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "book_1021"
_BOOK_ID = 1021
_DIMS = 8


def _location() -> BookLocation:
    return BookLocation(book_dir=_FIXTURE_DIR, book_id=_BOOK_ID, category_id=1, has_all_files=True)


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = get_engine()
    try:
        with eng.connect():
            pass
    except Exception:  # noqa: BLE001 - any connection error means Postgres is unavailable
        pytest.skip("Postgres not reachable")
    Base.metadata.create_all(eng)
    try:
        yield eng
    finally:
        with Session(eng) as session, session.begin():
            session.execute(delete(Chunk).where(Chunk.book_id == _BOOK_ID))
            session.execute(delete(Section).where(Section.book_id == _BOOK_ID))
            session.execute(delete(Book).where(Book.book_id == _BOOK_ID))


@pytest.fixture
def store() -> Iterator[QdrantStore]:
    instance = QdrantStore(
        url=url_from_settings(), collection=f"test_ingest_{uuid.uuid4().hex}", dense_dim=_DIMS
    )
    try:
        instance.client.get_collections()
    except Exception:  # noqa: BLE001 - any connection error means Qdrant is unavailable
        pytest.skip("Qdrant not reachable")
    try:
        yield instance
    finally:
        instance.delete_collection()


def url_from_settings() -> str:
    from shamela_rag.config import get_settings

    return get_settings().qdrant_url


def _service(engine: Engine, store: QdrantStore) -> IngestionService:
    factory: sessionmaker[Session] = get_sessionmaker(engine)
    return IngestionService(
        session_factory=factory, store=store, embedder=InMemoryEmbeddingProvider(dims=_DIMS)
    )


def _chunk_count(engine: Engine) -> int:
    with Session(engine) as session:
        return session.execute(
            select(func.count()).select_from(Chunk).where(Chunk.book_id == _BOOK_ID)
        ).scalar_one()


def _point_count(store: QdrantStore) -> int:
    return store.count()


def test_ingests_fixture_book_into_postgres_and_qdrant(engine: Engine, store: QdrantStore) -> None:
    service = _service(engine, store)

    summary = service.ingest_book(_location())

    assert summary.chunk_count > 0
    assert summary.upserted_points == summary.chunk_count
    assert not summary.skipped
    assert _chunk_count(engine) == summary.chunk_count
    assert _point_count(store) == summary.chunk_count

    with Session(engine) as session:
        book = session.get(Book, _BOOK_ID)
        assert book is not None
        sample = session.execute(
            select(Chunk).where(Chunk.book_id == _BOOK_ID).limit(1)
        ).scalar_one()
        assert sample.source_text.strip() != ""


def test_reingesting_does_not_duplicate(engine: Engine, store: QdrantStore) -> None:
    service = _service(engine, store)

    first = service.ingest_book(_location())
    second = service.ingest_book(_location())

    assert second.chunk_count == first.chunk_count
    assert _chunk_count(engine) == first.chunk_count
    assert _point_count(store) == first.chunk_count


def test_dry_run_writes_nothing(engine: Engine, store: QdrantStore) -> None:
    service = _service(engine, store)

    summary = service.ingest_book(_location(), dry_run=True)

    assert summary.dry_run is True
    assert summary.chunk_count > 0
    assert summary.upserted_points == 0
    assert _chunk_count(engine) == 0


def test_missing_files_book_is_skipped(engine: Engine, store: QdrantStore) -> None:
    service = _service(engine, store)
    location = BookLocation(
        book_dir=_FIXTURE_DIR, book_id=_BOOK_ID, category_id=1, has_all_files=False
    )

    summary = service.ingest_book(location)

    assert summary.skipped
    assert summary.skipped_reason == "missing_files"
    assert _chunk_count(engine) == 0
