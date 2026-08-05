"""Compact Arabic context header prepended to each embedding child for attribution.

Only the most useful fields go in the header (book, author + era, TOC path, content role); the full
bibliographic ``betaka_text`` is deliberately never prepended. The death-year sentinel ``99999``
means "unknown" and is omitted from the text (stored as null in metadata). The header is produced
separately from the child text so how much of it goes into the dense input can be A/B tested (M6):
callers combine them via ``embedding_input`` with ``include_header`` as the toggle.
"""

from __future__ import annotations

from dataclasses import dataclass

from shamela_rag.chunking.content_roles import ContentRole
from shamela_rag.data.models import Book

DEATH_YEAR_UNKNOWN = 99999


@dataclass(frozen=True)
class HeaderContext:
    book_title: str | None
    author_name: str | None
    author_death_hijri: int | None
    trail: tuple[str, ...]
    content_role: ContentRole


def normalize_death_year(value: int | None) -> int | None:
    return None if value is None or value == DEATH_YEAR_UNKNOWN else value


def _role_label(role: ContentRole) -> str:
    return "متن" if role is ContentRole.BODY else "حاشية"


def build_context_header(context: HeaderContext) -> str:
    lines: list[str] = []
    if context.book_title:
        lines.append(f"الكتاب: {context.book_title}")
    if context.author_name:
        death = normalize_death_year(context.author_death_hijri)
        if death is not None:
            lines.append(f"المؤلف: {context.author_name} (ت {death} هـ)")
        else:
            lines.append(f"المؤلف: {context.author_name}")
    if context.trail:
        lines.append(f"المسار: {' > '.join(context.trail)}")
    lines.append(f"نوع المحتوى: {_role_label(context.content_role)}")
    return "\n".join(lines)


def context_from(book: Book, trail: tuple[str, ...], role: ContentRole) -> HeaderContext:
    return HeaderContext(
        book_title=book.title_ar,
        author_name=book.main_author_name_ar,
        author_death_hijri=book.main_author_death_hijri,
        trail=trail,
        content_role=role,
    )


def embedding_input(header: str, text: str, *, include_header: bool = True) -> str:
    return f"{header}\n\n{text}" if include_header and header else text
