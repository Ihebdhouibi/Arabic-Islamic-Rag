from __future__ import annotations

from shamela_rag.chunking.boundaries import (
    Boundary,
    BoundarySource,
    Confidence,
    confidence_counts,
    detect_page_boundaries,
)
from shamela_rag.data.models import TocEntry


def _toc(shamela_title_id: int, title_text: str, page_id: int = 100) -> TocEntry:
    return TocEntry(
        title_id=shamela_title_id + 1000,
        book_id=1,
        page_id=page_id,
        shamela_title_id=shamela_title_id,
        title_text=title_text,
    )


def test_inline_toc_is_high_confidence() -> None:
    body = "<span data-type='title' id=toc-37>باب الهمزة</span>نص"
    boundaries = detect_page_boundaries(body, [])
    assert len(boundaries) == 1
    assert boundaries[0].source is BoundarySource.INLINE_TOC
    assert boundaries[0].confidence is Confidence.HIGH
    assert boundaries[0].shamela_title_id == 37


def test_inline_title_without_id_is_medium() -> None:
    body = "<span data-type='title'>عنوان بلا معرف</span>نص"
    boundaries = detect_page_boundaries(body, [])
    assert [b.source for b in boundaries] == [BoundarySource.INLINE_TITLE]
    assert boundaries[0].confidence is Confidence.MEDIUM


def test_recovered_title_matches_toc_text_in_body() -> None:
    body = "مقدمة قصيرة\rباب الطهارة ثم المتن"
    boundaries = detect_page_boundaries(body, [_toc(5, "باب الطهارة")])
    assert [b.source for b in boundaries] == [BoundarySource.RECOVERED_TITLE]
    assert boundaries[0].offset == body.find("باب الطهارة")
    assert boundaries[0].shamela_title_id == 5


def test_single_unmatched_toc_entry_falls_back_to_page_start() -> None:
    boundaries = detect_page_boundaries("متن بلا عنوان مطابق", [_toc(9, "عنوان غير موجود")])
    assert [b.source for b in boundaries] == [BoundarySource.TOC_PAGE_FALLBACK]
    assert boundaries[0].offset == 0
    assert boundaries[0].confidence is Confidence.LOW


def test_multiple_unmatched_toc_entries_are_ambiguous() -> None:
    boundaries = detect_page_boundaries(
        "متن", [_toc(1, "عنوان أ غير موجود"), _toc(2, "عنوان ب غير موجود")]
    )
    assert [b.source for b in boundaries] == [BoundarySource.AMBIGUOUS_TOC_PAGE]


def test_no_toc_uses_paragraph_fallback() -> None:
    body = "فقرة اولى\n\nفقرة ثانية"
    boundaries = detect_page_boundaries(body, [])
    assert {b.source for b in boundaries} == {BoundarySource.PARAGRAPH_FALLBACK}
    assert [b.offset for b in boundaries] == [0, body.index("فقرة ثانية")]


def test_confidence_counts() -> None:
    boundaries = [
        Boundary(0, BoundarySource.INLINE_TOC, Confidence.HIGH),
        Boundary(5, BoundarySource.RECOVERED_TITLE, Confidence.MEDIUM),
        Boundary(9, BoundarySource.PARAGRAPH_FALLBACK, Confidence.LOW),
    ]
    counts = confidence_counts(boundaries)
    assert counts[Confidence.HIGH] == 1
    assert counts[Confidence.MEDIUM] == 1
    assert counts[Confidence.LOW] == 1
