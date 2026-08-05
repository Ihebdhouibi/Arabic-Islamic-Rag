from __future__ import annotations

from pathlib import Path

import pytest

from shamela_rag.chunking.title_spans import parse_title_spans
from shamela_rag.data.models import load_pages

FIXTURE = Path(__file__).parent / "fixtures" / "book_1021"


@pytest.mark.parametrize(
    "body",
    [
        "<span data-type='title' id=toc-7>x</span>",
        '<span data-type="title" id="toc-7">x</span>',
        "<span id=toc-7 data-type='title'>x</span>",
    ],
)
def test_tolerates_quoting_and_attribute_order(body: str) -> None:
    spans = parse_title_spans(body)
    assert len(spans) == 1
    assert spans[0].shamela_title_id == 7


def test_captures_multiple_spans_including_ids_absent_from_toc() -> None:
    body = (
        "<span data-type='title' id=toc-37>باب الهمزة</span>نص"
        "<span data-type='title' id=toc-38>فرع</span>مزيد"
        "<span data-type='title' id=toc-39>عنوان</span>"
    )
    assert [s.shamela_title_id for s in parse_title_spans(body)] == [37, 38, 39]


def test_records_offsets_and_inner_text() -> None:
    body = "abc<span data-type='title' id=toc-5>عنوان</span>xyz"
    spans = parse_title_spans(body)
    assert len(spans) == 1
    assert spans[0].start == 3
    assert body[spans[0].start : spans[0].end].endswith("</span>")
    assert spans[0].title_text == "عنوان"


def test_ignores_non_title_spans() -> None:
    assert parse_title_spans("<span data-type='footnote' id=fn-1>x</span>") == []


def test_parses_fixture_pages() -> None:
    pages = {p.page_id: p for p in load_pages(FIXTURE)}
    p2 = parse_title_spans(pages[954949].body)
    assert p2[0].shamela_title_id == 37
    assert p2[0].title_text == "باب الهمزة"
    # toc-1 on page 1 is absent from the fixture toc.jsonl but still captured from inline markup.
    p1 = parse_title_spans(pages[954912].body)
    assert p1[0].shamela_title_id == 1
