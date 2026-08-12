"""M6-05 single-book end-to-end smoke test.

Writes a tiny self-contained book with a known body passage, ingests it, and asks one hand-written
question through the full QA service. Asserts the known passage is retrieved, expanded, and cited.
Deterministic (offline embedder/reranker/generation); runs in CI where Postgres + Qdrant are up.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
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
from shamela_rag.vectorstore.qdrant_store import QdrantStore

_BOOK_ID = 910021
_BOOK_TITLE = "كتاب العلم"
_DIMS = 8
_QUESTION = "ما قال الإمام الشافعي في طلب العلم؟"
_KNOWN_ENTITY = "الشافعي"
_BODY = (
    "<span data-type='title' id=toc-1>باب فضل العلم</span>\r"
    "قال الإمام الشافعي رحمه الله طلب العلم فريضة على كل مسلم ومن أراد الدنيا "
    "فعليه بالعلم ومن أراد الآخرة فعليه بالعلم فالعلم نور يهدي صاحبه إلى الحق."
)


def _write_book(directory: Path) -> BookLocation:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "book_metadata.json").write_text(
        json.dumps(
            {
                "book_id": _BOOK_ID,
                "title_ar": _BOOK_TITLE,
                "book_type_label": "كتاب",
                "category_id": 11,
                "main_author_name_ar": "مؤلف",
                "main_author_death_hijri": 300,
            }
        ),
        encoding="utf-8",
    )
    (directory / "toc.jsonl").write_text(
        json.dumps(
            {
                "title_id": 1,
                "book_id": _BOOK_ID,
                "page_id": 1,
                "parent_id": None,
                "shamela_title_id": 1,
                "title_text": "باب فضل العلم",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "pages.jsonl").write_text(
        json.dumps(
            {"page_id": 1, "book_id": _BOOK_ID, "shamela_page_id": 1, "body": _BODY},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return BookLocation(book_dir=directory, book_id=_BOOK_ID, category_id=11, has_all_files=True)


@dataclass
class _World:
    qa: GeneralQAService
    engine: Engine


@pytest.fixture
def world(tmp_path: Path) -> Iterator[_World]:
    engine = get_engine()
    try:
        with engine.connect():
            pass
    except Exception:  # noqa: BLE001 - Postgres unavailable
        pytest.skip("Postgres not reachable")

    store = QdrantStore(
        url=get_settings().qdrant_url, collection=f"test_smoke_{uuid.uuid4().hex}", dense_dim=_DIMS
    )
    try:
        store.client.get_collections()
    except Exception:  # noqa: BLE001 - Qdrant unavailable
        pytest.skip("Qdrant not reachable")

    Base.metadata.create_all(engine)
    location = _write_book(tmp_path / "book")
    embedder = InMemoryEmbeddingProvider(dims=_DIMS)
    session_factory = get_sessionmaker(engine)
    IngestionService(session_factory=session_factory, store=store, embedder=embedder).ingest_book(
        location
    )

    encoder = Bm25Encoder().fit(
        chunk.retrieval_text for chunk in chunk_book(location.book_dir).chunks
    )
    retrieval = RetrievalService(
        translator=InMemoryTranslator(),
        dense_retriever=DenseRetriever(embedder=embedder, store=store),
        sparse_retriever=SparseRetriever(encoder=encoder, store=store),
        reranker=LexicalOverlapReranker(),
        expander=ContextExpander(session_factory),
        session_factory=session_factory,
        config=RetrievalConfig(final_k=3),
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


def test_single_book_end_to_end_smoke(world: _World) -> None:
    answer = world.qa.answer(_QUESTION)

    assert not answer.deflected
    assert answer.citations, "expected at least one citation"
    # The known passage was retrieved and fed to generation (InMemory echoes the prompt).
    assert _KNOWN_ENTITY in answer.text

    top = answer.citations[0]
    assert top.book_title == _BOOK_TITLE
    with Session(world.engine) as session:
        chunk = session.get(Chunk, top.chunk_id)
        assert chunk is not None, "citation must resolve to a real chunk"
        assert chunk.book_id == _BOOK_ID
        assert _KNOWN_ENTITY in chunk.source_text
