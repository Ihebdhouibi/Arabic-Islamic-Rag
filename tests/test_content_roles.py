from __future__ import annotations

from pathlib import Path

from shamela_rag.chunking.content_roles import (
    ContentRole,
    iter_content_units,
    split_page_content,
)
from shamela_rag.data.models import Page, load_pages

FIXTURE = Path(__file__).parent / "fixtures" / "book_1021"


def _page(page_id: int, body: str, footnotes: str | None) -> Page:
    return Page(page_id=page_id, book_id=1, body=body, footnotes=footnotes)


def test_body_only_page_yields_one_body_unit() -> None:
    units = split_page_content(_page(1, "متن", None))
    assert [u.content_role for u in units] == [ContentRole.BODY]


def test_body_and_footnote_are_separate_never_concatenated() -> None:
    units = split_page_content(_page(1, "متن", "حاشية"))
    assert [u.content_role for u in units] == [ContentRole.BODY, ContentRole.FOOTNOTE]
    assert all(u.page_id == 1 for u in units)
    assert units[0].text == "متن"
    assert units[1].text == "حاشية"


def test_blank_footnotes_are_skipped() -> None:
    units = split_page_content(_page(1, "متن", "   "))
    assert [u.content_role for u in units] == [ContentRole.BODY]


def test_fixture_pages_have_expected_roles() -> None:
    roles = {(u.page_id, u.content_role) for u in iter_content_units(load_pages(FIXTURE))}
    assert (954912, ContentRole.BODY) in roles
    assert (954949, ContentRole.BODY) in roles
    assert (954949, ContentRole.FOOTNOTE) in roles
    assert (954912, ContentRole.FOOTNOTE) not in roles
