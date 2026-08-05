from __future__ import annotations

from pathlib import Path

from shamela_rag.data.models import load_book
from shamela_rag.data.routing import Route, route_for_book, route_for_category

FIXTURE = Path(__file__).parent / "fixtures" / "book_1021"


def test_known_categories_route() -> None:
    assert route_for_category(3) is Route.TAFSIR
    assert route_for_category(6) is Route.HADITH
    assert route_for_category(16) is Route.FIQH
    assert route_for_category(26) is Route.BIOGRAPHY
    assert route_for_category(34) is Route.POETRY


def test_unknown_and_none_fall_back_to_general() -> None:
    assert route_for_category(999) is Route.GENERAL
    assert route_for_category(None) is Route.GENERAL


def test_route_for_book_uses_category() -> None:
    # Fixture book_1021 is category 26 (التراجم والطبقات).
    assert route_for_book(load_book(FIXTURE)) is Route.BIOGRAPHY
