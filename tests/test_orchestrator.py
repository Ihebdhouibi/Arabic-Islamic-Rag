from __future__ import annotations

from pathlib import Path

from shamela_rag.chunking.content_roles import ContentRole
from shamela_rag.chunking.orchestrator import chunk_book
from shamela_rag.data.models import load_pages

FIXTURE = Path(__file__).parent / "fixtures" / "book_1021"


def test_source_text_round_trips_verbatim() -> None:
    pages = {p.page_id: p for p in load_pages(FIXTURE)}
    chunks = chunk_book(FIXTURE, min_content_tokens=1)
    assert chunks
    for chunk in chunks:
        page = pages[chunk.page_id]
        haystack = page.footnotes if chunk.content_role is ContentRole.FOOTNOTE else page.body
        assert haystack is not None
        assert chunk.source_text in haystack  # verbatim, unmutated
        assert haystack[chunk.start_offset : chunk.end_offset] == chunk.source_text


def test_chunking_is_deterministic() -> None:
    assert chunk_book(FIXTURE, min_content_tokens=1) == chunk_book(FIXTURE, min_content_tokens=1)


def test_footnote_chunk_is_emitted_with_role() -> None:
    footnotes = [
        c
        for c in chunk_book(FIXTURE, min_content_tokens=1)
        if c.content_role is ContentRole.FOOTNOTE
    ]
    assert any(c.source_text.strip() == "حاشية المحقق" for c in footnotes)


def test_body_chunk_trail_carries_section_heading() -> None:
    trails = {t for c in chunk_book(FIXTURE, min_content_tokens=1) for t in c.trail}
    assert "باب الهمزة" in trails


def test_chunks_have_context_header_and_normalized_retrieval_text() -> None:
    chunk = chunk_book(FIXTURE, min_content_tokens=1)[0]
    assert "الكتاب:" in chunk.context_header
    assert chunk.retrieval_text == chunk.retrieval_text.strip()
