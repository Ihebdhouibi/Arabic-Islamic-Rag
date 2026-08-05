"""SQLAlchemy ORM models.

A chunk keeps its verbatim ``source_text`` separate from the normalized ``retrieval_text`` (so the
corpus text can round-trip unchanged) and from its ``context_header`` (so the dense-input prefix can
be tuned independently). Book-level metadata lives on ``Book``; section structure/provenance on
``Section``; and per-chunk placement/offsets on ``Chunk``.
"""

from __future__ import annotations

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Book(Base):
    __tablename__ = "books"

    # Internal Shamela book_id (matches pages.jsonl / book_metadata.json), not auto-generated.
    book_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    title_ar: Mapped[str | None] = mapped_column(Text)
    author_name_ar: Mapped[str | None] = mapped_column(Text)
    author_death_hijri: Mapped[int | None] = mapped_column()
    category_id: Mapped[int | None] = mapped_column()
    book_type_label: Mapped[str | None] = mapped_column(String(64))


class Section(Base):
    __tablename__ = "sections"

    id: Mapped[int] = mapped_column(primary_key=True)
    book_id: Mapped[int] = mapped_column(
        ForeignKey("books.book_id", ondelete="CASCADE"), index=True
    )
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("sections.id", ondelete="CASCADE"))
    shamela_title_id: Mapped[int | None] = mapped_column()
    title_text: Mapped[str | None] = mapped_column(Text)
    title_trail: Mapped[str | None] = mapped_column(Text)  # root -> leaf, ' > '-joined
    depth: Mapped[int | None] = mapped_column()
    path_source: Mapped[str | None] = mapped_column(String(32))  # explicit_parent | derived_order
    boundary_source: Mapped[str | None] = mapped_column(String(32))
    confidence: Mapped[str | None] = mapped_column(String(16))  # high | medium | low
    start_page_id: Mapped[int | None] = mapped_column()
    end_page_id: Mapped[int | None] = mapped_column()


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    book_id: Mapped[int] = mapped_column(
        ForeignKey("books.book_id", ondelete="CASCADE"), index=True
    )
    section_id: Mapped[int | None] = mapped_column(ForeignKey("sections.id", ondelete="SET NULL"))
    content_role: Mapped[str] = mapped_column(String(16))  # body | footnote
    source_text: Mapped[str] = mapped_column(Text)  # verbatim; never mutated
    retrieval_text: Mapped[str | None] = mapped_column(Text)  # normalized for indexing
    context_header: Mapped[str | None] = mapped_column(Text)  # stored apart from source_text
    part: Mapped[str | None] = mapped_column(String(16))
    start_page_id: Mapped[int | None] = mapped_column()
    end_page_id: Mapped[int | None] = mapped_column()
    start_offset: Mapped[int | None] = mapped_column()  # char offset into the book source stream
    end_offset: Mapped[int | None] = mapped_column()
    token_count: Mapped[int | None] = mapped_column()
