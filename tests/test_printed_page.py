from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from shamela_rag.chunking.orchestrator import chunk_book
from shamela_rag.chunking.sections import build_sections
from shamela_rag.data.models import load_book, load_toc
from shamela_rag.db.engine import get_engine, get_sessionmaker
from shamela_rag.db.models import Base, Book, Chunk, Section
from shamela_rag.ingestion.pipeline import IngestionService

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "book_1021"
_BOOK_ID = 1021
_PAGE_ID = 954949
_PRINTED_PAGE = 147


def test_chunk_book_carries_printed_page_num() -> None:
    chunks = chunk_book(_FIXTURE_DIR, min_content_tokens=1).chunks
    on_page = [c for c in chunks if c.page_id == _PAGE_ID]
    assert on_page
    assert all(c.page_num == _PRINTED_PAGE for c in on_page)
    assert _PAGE_ID != _PRINTED_PAGE


def test_insert_chunks_persists_printed_page_num() -> None:
    engine = get_engine()
    try:
        with engine.connect():
            pass
    except Exception:  # noqa: BLE001
        pytest.skip("Postgres not reachable")

    book_meta = load_book(_FIXTURE_DIR)
    sections = build_sections(list(load_toc(_FIXTURE_DIR)))
    chunks = chunk_book(_FIXTURE_DIR, min_content_tokens=1).chunks
    assert chunks

    Base.metadata.create_all(engine)
    session_factory = get_sessionmaker(engine)
    try:
        with session_factory() as session, session.begin():
            session.execute(delete(Chunk).where(Chunk.book_id == _BOOK_ID))
            session.execute(delete(Section).where(Section.book_id == _BOOK_ID))
            session.execute(delete(Book).where(Book.book_id == _BOOK_ID))
            session.add(
                Book(
                    book_id=book_meta.book_id,
                    title_ar=book_meta.title_ar,
                    author_name_ar=book_meta.main_author_name_ar,
                    author_death_hijri=book_meta.main_author_death_hijri,
                    category_id=book_meta.category_id,
                    book_type_label=book_meta.book_type_label,
                )
            )
            session.flush()
            id_by_trail = IngestionService._insert_sections(session, _BOOK_ID, sections)
            IngestionService._insert_chunks(session, _BOOK_ID, chunks, id_by_trail, book_meta)

        with session_factory() as session:
            rows = session.scalars(
                select(Chunk).where(Chunk.book_id == _BOOK_ID, Chunk.start_page_id == _PAGE_ID)
            ).all()
            assert rows
            for row in rows:
                assert row.start_page_num == _PRINTED_PAGE
                assert row.end_page_num == _PRINTED_PAGE
                assert row.start_page_id == _PAGE_ID
    finally:
        with Session(engine) as session, session.begin():
            session.execute(delete(Chunk).where(Chunk.book_id == _BOOK_ID))
            session.execute(delete(Section).where(Section.book_id == _BOOK_ID))
            session.execute(delete(Book).where(Book.book_id == _BOOK_ID))
