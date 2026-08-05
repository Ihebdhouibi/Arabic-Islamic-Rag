from __future__ import annotations

import pytest

from shamela_rag.text.normalization import (
    normalize_alef,
    normalize_alef_maksura,
    normalize_for_display,
    normalize_for_index,
    normalize_ta_marbuta,
    strip_diacritics,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("الصَّلَاة", "الصلاة"),
        ("مُحَمَّد", "محمد"),
    ],
)
def test_strip_diacritics(raw: str, expected: str) -> None:
    assert strip_diacritics(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("أحمد", "احمد"), ("إسلام", "اسلام"), ("آدم", "ادم"), ("ٱلله", "الله")],
)
def test_normalize_alef(raw: str, expected: str) -> None:
    assert normalize_alef(raw) == expected


def test_normalize_ta_marbuta() -> None:
    assert normalize_ta_marbuta("صلاة") == "صلاه"


def test_normalize_alef_maksura() -> None:
    assert normalize_alef_maksura("على") == "علي"


@pytest.mark.parametrize(
    ("diacritized", "plain"),
    [("الصَّلَاةِ", "الصلاة"), ("مُحَمَّدٌ", "محمد")],
)
def test_index_folds_diacritic_variants(diacritized: str, plain: str) -> None:
    assert normalize_for_index(diacritized) == normalize_for_index(plain)


def test_inflections_normalize_consistently() -> None:
    assert normalize_for_index("الصلاة") == "الصلاه"
    assert normalize_for_index("للصلاة") == "للصلاه"
    assert normalize_for_index("صلاته") == "صلاته"


def test_display_preserves_diacritics_and_normalizes_newlines() -> None:
    assert normalize_for_display("سطر\rآخر") == "سطر\nآخر"
    assert "\u064f" in normalize_for_display("مُحَمَّد")
