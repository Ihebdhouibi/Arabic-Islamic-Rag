"""Separate a page's body from its footnotes into role-tagged content units.

Body text and footnotes must never be concatenated: footnotes hold editor/muhaqqiq commentary,
manuscript variants, and citations that are not the original author's words. Each unit keeps only
page-level linkage (the footnote-marker-to-body-position relationship is unreliable, so we do not
fabricate an exact attachment). The ``FOOTNOTE`` role travels with the unit so downstream never
attributes a footnote to the book's author.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum

from shamela_rag.data.models import Page


class ContentRole(StrEnum):
    BODY = "body"
    FOOTNOTE = "footnote"


@dataclass(frozen=True)
class ContentUnit:
    page_id: int  # page-level linkage; a footnote links to the body of the same page
    content_role: ContentRole
    text: str


def split_page_content(page: Page) -> list[ContentUnit]:
    units = [ContentUnit(page.page_id, ContentRole.BODY, page.body)]
    if page.footnotes and page.footnotes.strip():
        units.append(ContentUnit(page.page_id, ContentRole.FOOTNOTE, page.footnotes))
    return units


def iter_content_units(pages: Iterable[Page]) -> Iterator[ContentUnit]:
    for page in pages:
        yield from split_page_content(page)
