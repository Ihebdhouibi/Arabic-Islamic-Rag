from __future__ import annotations

from pathlib import Path

from shamela_rag.data.models import Book, Page, TocEntry, load_book, load_pages, load_toc

FIXTURE = Path(__file__).parent / "fixtures" / "book_1021"


def test_load_pages_returns_models() -> None:
    pages = list(load_pages(FIXTURE))
    assert len(pages) == 2
    assert isinstance(pages[0], Page)
    assert pages[0].book_id == 1021
    assert pages[0].footnotes is None
    assert pages[1].footnotes == "حاشية المحقق"


def test_load_toc_returns_models() -> None:
    toc = list(load_toc(FIXTURE))
    assert all(isinstance(t, TocEntry) for t in toc)
    assert toc[0].shamela_title_id == 37


def test_load_book_returns_model() -> None:
    book = load_book(FIXTURE)
    assert isinstance(book, Book)
    assert book.book_id == 1021
    assert book.main_author_death_hijri == 630
    assert book.book_type_label == "كتاب"
