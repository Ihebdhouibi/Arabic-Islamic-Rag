from __future__ import annotations

from shamela_rag.chunking.sections import PathSource, build_sections
from shamela_rag.data.models import TocEntry


def _entry(title_id: int, text: str, page_id: int, parent_id: int | None = None) -> TocEntry:
    return TocEntry(
        title_id=title_id,
        book_id=1,
        page_id=page_id,
        parent_id=parent_id,
        shamela_title_id=title_id,
        title_text=text,
    )


def test_derived_trail_for_null_parent_leaf() -> None:
    harf = _entry(1, "حرف الباء الموحدة", 10)
    bab = _entry(2, "باب الباء والألف", 11, parent_id=1)
    baqum = _entry(3, "باقوم", 12)  # parent_id null, a leaf under bab
    by_text = {s.title_text: s for s in build_sections([harf, bab, baqum])}

    assert by_text["باقوم"].trail == ("حرف الباء الموحدة", "باب الباء والألف", "باقوم")
    assert by_text["باقوم"].path_source is PathSource.DERIVED_ORDER
    assert by_text["باب الباء والألف"].path_source is PathSource.EXPLICIT_PARENT
    assert by_text["حرف الباء الموحدة"].trail == ("حرف الباء الموحدة",)


def test_top_level_siblings_are_not_falsely_nested() -> None:
    bab1 = _entry(1, "باب الهمزة", 38)
    bab2 = _entry(2, "باب الحاء", 50)
    by_text = {s.title_text: s for s in build_sections([bab1, bab2])}

    assert by_text["باب الهمزة"].trail == ("باب الهمزة",)
    assert by_text["باب الحاء"].trail == ("باب الحاء",)
    assert by_text["باب الحاء"].path_source is PathSource.EXPLICIT_PARENT


def test_page_ranges_follow_document_order() -> None:
    by_text = {s.title_text: s for s in build_sections([_entry(1, "A", 10), _entry(2, "B", 20)])}
    assert by_text["A"].start_page_id == 10
    assert by_text["A"].end_page_id == 20
    assert by_text["B"].end_page_id is None
