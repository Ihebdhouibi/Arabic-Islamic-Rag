"""CI ingest smoke: miniature fixture into Postgres + Qdrant, then one retrieval assertion."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import delete
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
from shamela_rag.retrieval.sparse import SparseRetriever
from shamela_rag.vectorstore.qdrant_store import QdrantStore

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ci_ingest_book"
_BOOK_ID = 800058
_DIMS = 8
_KNOWN_TERM = "الشافعي"


def _location() -> BookLocation:
    return BookLocation(book_dir=_FIXTURE_DIR, book_id=_BOOK_ID, category_id=11, has_all_files=True)


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = get_engine()
    try:
        with eng.connect():
            pass
    except Exception:  # noqa: BLE001 - Postgres unavailable
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
        url=get_settings().qdrant_url,
        collection=f"test_ci_ingest_{uuid.uuid4().hex}",
        dense_dim=_DIMS,
    )
    try:
        instance.client.get_collections()
    except Exception:  # noqa: BLE001 - Qdrant unavailable
        pytest.skip("Qdrant not reachable")
    try:
        yield instance
    finally:
        instance.delete_collection()


def test_ci_ingests_fixture_and_retrieves_known_passage(engine: Engine, store: QdrantStore) -> None:
    location = _location()
    encoder = Bm25Encoder().fit(
        chunk.retrieval_text for chunk in chunk_book(location.book_dir).chunks
    )
    IngestionService(
        session_factory=get_sessionmaker(engine),
        store=store,
        embedder=InMemoryEmbeddingProvider(dims=_DIMS),
        sparse_encoder=encoder,
    ).ingest_book(location)

    hits = SparseRetriever(encoder=encoder, store=store).search(_KNOWN_TERM, limit=1)
    assert hits, "expected a retrieval hit after ingesting the fixture"
    with Session(engine) as session:
        chunk = session.get(Chunk, hits[0].chunk_id)
        assert chunk is not None
        assert chunk.book_id == _BOOK_ID
        assert _KNOWN_TERM in chunk.source_text
