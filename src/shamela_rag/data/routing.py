"""Genre routing hook: maps a book's ``category_id`` to a retrieval/chunking route.

The general module currently uses the structural (TOC-anchored) chunking path for *every* route;
this map exists so the specialized paths (hadith takhrij, tafsir-by-verse, fiqh) can branch on it
later without reworking ingestion. Categories not listed fall back to ``GENERAL``.
"""

from __future__ import annotations

from enum import StrEnum

from shamela_rag.data.models import Book


class Route(StrEnum):
    GENERAL = "general"
    HADITH = "hadith"
    TAFSIR = "tafsir"
    FIQH = "fiqh"
    BIOGRAPHY = "biography"
    POETRY = "poetry"


# Category ids per the corpus taxonomy (see docs/technical_docs/07 §4).
_CATEGORY_ROUTES: dict[int, Route] = {
    3: Route.TAFSIR,  # التفسير
    6: Route.HADITH,  # كتب السنة
    7: Route.HADITH,  # شروح الحديث
    14: Route.FIQH,  # الفقه الحنفي
    15: Route.FIQH,  # الفقه المالكي
    16: Route.FIQH,  # الفقه الشافعي
    17: Route.FIQH,  # الفقه الحنبلي
    18: Route.FIQH,  # الفقه العام
    19: Route.FIQH,  # مسائل فقهية
    26: Route.BIOGRAPHY,  # التراجم والطبقات
    27: Route.BIOGRAPHY,  # الأنساب
    34: Route.POETRY,  # الشعر ودواوينه
}


def route_for_category(category_id: int | None) -> Route:
    if category_id is None:
        return Route.GENERAL
    return _CATEGORY_ROUTES.get(category_id, Route.GENERAL)


def route_for_book(book: Book) -> Route:
    return route_for_category(book.category_id)
