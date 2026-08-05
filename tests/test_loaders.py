from __future__ import annotations

from pathlib import Path

from shamela_rag.data.loaders import iter_pages, iter_toc, load_book_metadata

FIXTURE = Path(__file__).parent / "fixtures" / "book_1021"


def test_iter_pages_reads_valid_and_skips_blank_and_malformed() -> None:
    pages = list(iter_pages(FIXTURE))
    # Two valid records; the blank line and the truncated last line are skipped.
    assert len(pages) == 2
    assert pages[0]["book_id"] == 1021
    assert pages[0]["footnotes"] is None
    assert pages[1]["footnotes"] == "حاشية المحقق"


def test_iter_toc_reads_entries() -> None:
    toc = list(iter_toc(FIXTURE))
    assert len(toc) == 2
    assert toc[0]["shamela_title_id"] == 37


def test_load_book_metadata() -> None:
    md = load_book_metadata(FIXTURE)
    assert md["book_id"] == 1021
    assert md["book_type_label"] == "كتاب"
    assert "betaka_text" in md
