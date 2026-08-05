"""Typed domain models for the raw corpus records, layered on top of the streaming loaders.

Fields mirror the verified Shamela schema. Models ignore unknown fields (``book_metadata.json`` in
particular carries many more), and distinguish the internal ``page_id`` / ``title_id`` from the
original ``shamela_page_id`` / ``shamela_title_id`` (the latter is what inline ``toc-N`` markup
points to).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from shamela_rag.data import loaders


class Page(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page_id: int
    book_id: int
    shamela_page_id: int | None = None
    part: str | None = None
    page_num: int | None = None
    sequence_num: int | None = None
    body: str
    footnotes: str | None = None
    hints: str | None = None


class TocEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title_id: int
    book_id: int
    page_id: int
    parent_id: int | None = None
    shamela_title_id: int  # the id inline `toc-N` markup resolves to
    title_text: str


class Book(BaseModel):
    model_config = ConfigDict(extra="ignore")

    book_id: int
    shamela_id: int | None = None
    title_ar: str | None = None
    book_type_label: str | None = None
    category_id: int | None = None
    category_name_ar: str | None = None
    main_author_id: int | None = None
    main_author_name_ar: str | None = None
    main_author_death_hijri: int | None = None
    betaka_text: str | None = None


def load_pages(book_dir: Path) -> Iterator[Page]:
    for row in loaders.iter_pages(book_dir):
        yield Page.model_validate(row)


def load_toc(book_dir: Path) -> Iterator[TocEntry]:
    for row in loaders.iter_toc(book_dir):
        yield TocEntry.model_validate(row)


def load_book(book_dir: Path) -> Book:
    return Book.model_validate(loaders.load_book_metadata(book_dir))
