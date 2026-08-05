from __future__ import annotations

from pathlib import Path

from shamela_rag.chunking.content_roles import ContentRole
from shamela_rag.chunking.context_header import (
    HeaderContext,
    build_context_header,
    context_from,
    embedding_input,
    normalize_death_year,
)
from shamela_rag.data.models import load_book

FIXTURE = Path(__file__).parent / "fixtures" / "book_1021"


def _ctx(death: int | None, role: ContentRole = ContentRole.BODY) -> HeaderContext:
    return HeaderContext("كتاب", "مؤلف", death, ("حرف الألف", "باب"), role)


def test_known_death_year_is_shown() -> None:
    assert "المؤلف: مؤلف (ت 630 هـ)" in build_context_header(_ctx(630))


def test_unknown_death_year_sentinel_is_omitted() -> None:
    header = build_context_header(_ctx(99999))
    assert "المؤلف: مؤلف" in header
    assert "ت 99999" not in header
    assert "(ت" not in header


def test_none_death_year_is_omitted() -> None:
    assert "(ت" not in build_context_header(_ctx(None))


def test_header_fields_and_role_label() -> None:
    header = build_context_header(_ctx(630, ContentRole.FOOTNOTE))
    assert "الكتاب: كتاب" in header
    assert "المسار: حرف الألف > باب" in header
    assert "نوع المحتوى: حاشية" in header


def test_normalize_death_year() -> None:
    assert normalize_death_year(99999) is None
    assert normalize_death_year(None) is None
    assert normalize_death_year(630) == 630


def test_embedding_input_toggle() -> None:
    assert embedding_input("H", "متن") == "H\n\nمتن"
    assert embedding_input("H", "متن", include_header=False) == "متن"


def test_context_from_fixture_book() -> None:
    header = build_context_header(
        context_from(load_book(FIXTURE), ("باب الهمزة",), ContentRole.BODY)
    )
    assert "الكتاب: أسد الغابة" in header
    assert "المؤلف: عز الدين ابن الأثير (ت 630 هـ)" in header
    assert "المسار: باب الهمزة" in header
    assert "نوع المحتوى: متن" in header
