from __future__ import annotations

import json
from pathlib import Path

from shamela_rag.chunking.content_roles import ContentRole
from shamela_rag.chunking.orchestrator import BookChunk, chunk_book
from shamela_rag.chunking.sizing import SizePolicy
from shamela_rag.data.models import load_pages

FIXTURE = Path(__file__).parent / "fixtures" / "book_1021"


def _chunks() -> list[BookChunk]:
    return chunk_book(FIXTURE, min_content_tokens=1).chunks


def test_source_text_round_trips_verbatim() -> None:
    pages = {p.page_id: p for p in load_pages(FIXTURE)}
    chunks = _chunks()
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
    footnotes = [c for c in _chunks() if c.content_role is ContentRole.FOOTNOTE]
    assert any(c.source_text.strip() == "حاشية المحقق" for c in footnotes)


def test_body_chunk_trail_carries_section_heading() -> None:
    trails = {t for c in _chunks() for t in c.trail}
    assert "باب الهمزة" in trails


def test_chunks_have_context_header_and_normalized_retrieval_text() -> None:
    chunk = _chunks()[0]
    assert "الكتاب:" in chunk.context_header
    assert chunk.retrieval_text == chunk.retrieval_text.strip()


def test_stats_are_reported() -> None:
    result = chunk_book(FIXTURE, min_content_tokens=1)
    assert result.stats.chunk_count == len(result.chunks)
    assert sum(result.stats.confidence_counts.values()) >= 1
    assert result.stats.heading_recovery_candidates >= 0


def _write_book(directory: Path, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "book_metadata.json").write_text(
        json.dumps({"book_id": 1, "title_ar": "كتاب", "main_author_name_ar": "مؤلف"}),
        encoding="utf-8",
    )
    (directory / "toc.jsonl").write_text("", encoding="utf-8")
    (directory / "pages.jsonl").write_text(
        json.dumps({"page_id": 1, "book_id": 1, "body": body, "footnotes": None}) + "\n",
        encoding="utf-8",
    )


def test_short_tail_fragment_is_merged_into_previous(tmp_path: Path) -> None:
    _write_book(tmp_path, "aa bb. cc dd. ee ff. gg hh. z.")
    policy = SizePolicy(min_tokens=5, max_tokens=10, split_target_tokens=6, overlap_tokens=0)
    body = [
        c
        for c in chunk_book(tmp_path, policy=policy, min_content_tokens=1).chunks
        if c.content_role is ContentRole.BODY
    ]
    # The tiny trailing "z." merges into the previous child instead of standing alone.
    assert body[-1].source_text.strip().endswith("z.")
    assert "gg hh." in body[-1].source_text
    assert all(c.source_text.strip() != "z." for c in body)
