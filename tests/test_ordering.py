from __future__ import annotations

from pathlib import Path

from shamela_rag.data.models import Page
from shamela_rag.data.ordering import build_source_stream, load_source_stream, order_pages

FIXTURE = Path(__file__).parent / "fixtures" / "book_1021"


def _page(page_id: int, body: str, sequence_num: int) -> Page:
    return Page(page_id=page_id, book_id=1, body=body, sequence_num=sequence_num)


def test_order_is_by_page_id_not_sequence_num() -> None:
    # Both pages share sequence_num=1; ordering must still be deterministic by page_id.
    pages = [_page(20, "second", 1), _page(10, "first", 1)]
    ordered = order_pages(pages)
    assert [p.page_id for p in ordered] == [10, 20]


def test_spans_map_back_to_verbatim_body() -> None:
    pages = [_page(20, "beta", 1), _page(10, "alpha", 1)]
    stream = build_source_stream(pages, separator="\n")
    assert [s.page.page_id for s in stream.spans] == [10, 20]
    for span in stream.spans:
        assert stream.text[span.start : span.end] == span.page.body
    assert stream.page_at(0).page.page_id == 10  # type: ignore[union-attr]
    assert stream.page_at(stream.spans[1].start).page.page_id == 20  # type: ignore[union-attr]


def test_load_source_stream_from_fixture() -> None:
    stream = load_source_stream(FIXTURE)
    assert [s.page.page_id for s in stream.spans] == [954912, 954949]
    for span in stream.spans:
        assert stream.text[span.start : span.end] == span.page.body
