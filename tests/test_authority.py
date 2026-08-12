from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from shamela_rag.chunking.context_header import DEATH_YEAR_UNKNOWN
from shamela_rag.data.models import Book, load_book
from shamela_rag.retrieval.authority import (
    PRINTED_BOOK_TYPE,
    TRANSCRIPT_BOOK_TYPE,
    adjusted_authority_score,
    apply_authority_boost,
)
from shamela_rag.retrieval.rerank import RerankedChunk

FIXTURE_BOOK = Path(__file__).parent / "fixtures" / "book_1021"
CORPUS_ROOT = Path(os.environ.get("SHAMELA_CORPUS_ROOT", "Shamela4_Full_DB"))
_LOCAL_PAIR_LIMIT = 8


def _chunk(
    chunk_id: int,
    score: float,
    *,
    book_type_label: str | None = None,
    author_death_hijri: int | None = None,
    text: str = "نص",
) -> RerankedChunk:
    payload: dict[str, object] = {}
    if book_type_label is not None:
        payload["book_type_label"] = book_type_label
    if author_death_hijri is not None:
        payload["author_death_hijri"] = author_death_hijri
    return RerankedChunk(chunk_id=chunk_id, score=score, text=text, payload=payload)


def _sample_corpus_books(root: Path, *, label: str, limit: int) -> list[tuple[Path, Book]]:
    found: list[tuple[Path, Book]] = []
    for meta_path in sorted(root.rglob("book_metadata.json")):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("book_type_label") != label:
            continue
        book_dir = meta_path.parent
        try:
            book = load_book(book_dir)
        except (OSError, ValueError):
            continue
        found.append((book_dir, book))
        if len(found) >= limit:
            break
    return found


def test_printed_outranks_transcript_when_scores_equal() -> None:
    ranked = apply_authority_boost(
        [
            _chunk(1, 0.5, book_type_label=TRANSCRIPT_BOOK_TYPE),
            _chunk(2, 0.5, book_type_label=PRINTED_BOOK_TYPE),
        ]
    )
    assert [c.chunk_id for c in ranked] == [2, 1]
    assert ranked[0].score > ranked[1].score


def test_adjusted_score_helpers() -> None:
    assert adjusted_authority_score(1.0, PRINTED_BOOK_TYPE) == pytest.approx(1.1)
    assert adjusted_authority_score(1.0, TRANSCRIPT_BOOK_TYPE) == pytest.approx(0.9)
    assert adjusted_authority_score(1.0, "رسالة") == pytest.approx(1.0)
    assert adjusted_authority_score(1.0, None) == pytest.approx(1.0)


def test_empty_input() -> None:
    assert apply_authority_boost([]) == []


def test_death_hijri_orders_debate_history_mode() -> None:
    ranked = apply_authority_boost(
        [
            _chunk(1, 1.0, book_type_label=PRINTED_BOOK_TYPE, author_death_hijri=700),
            _chunk(2, 1.0, book_type_label=PRINTED_BOOK_TYPE, author_death_hijri=200),
        ],
        order_by_death_hijri=True,
    )
    assert [c.chunk_id for c in ranked] == [2, 1]


def test_unknown_death_year_99999_sorts_after_known() -> None:
    ranked = apply_authority_boost(
        [
            _chunk(
                1,
                1.0,
                book_type_label=PRINTED_BOOK_TYPE,
                author_death_hijri=DEATH_YEAR_UNKNOWN,
            ),
            _chunk(2, 1.0, book_type_label=PRINTED_BOOK_TYPE, author_death_hijri=300),
        ],
        order_by_death_hijri=True,
    )
    assert [c.chunk_id for c in ranked] == [2, 1]


def test_preserves_text_and_payload() -> None:
    original = _chunk(7, 0.8, book_type_label=PRINTED_BOOK_TYPE, text="متن")
    ranked = apply_authority_boost([original])
    assert ranked[0].text == "متن"
    assert ranked[0].payload["book_type_label"] == PRINTED_BOOK_TYPE


def test_github_fixture_book_is_printed_and_beats_transcript() -> None:
    book = load_book(FIXTURE_BOOK)
    assert book.book_type_label == PRINTED_BOOK_TYPE
    ranked = apply_authority_boost(
        [
            _chunk(
                1,
                0.55,
                book_type_label=TRANSCRIPT_BOOK_TYPE,
                author_death_hijri=800,
                text="درس مفرغ",
            ),
            _chunk(
                2,
                0.55,
                book_type_label=book.book_type_label,
                author_death_hijri=book.main_author_death_hijri,
                text=book.title_ar or "fixture",
            ),
        ]
    )
    assert ranked[0].chunk_id == 2
    assert ranked[0].payload["book_type_label"] == PRINTED_BOOK_TYPE


@pytest.mark.skipif(not CORPUS_ROOT.is_dir(), reason="local Shamela corpus not present")
def test_local_shamela_many_printed_vs_transcript_pairs() -> None:
    printed_books = _sample_corpus_books(
        CORPUS_ROOT, label=PRINTED_BOOK_TYPE, limit=_LOCAL_PAIR_LIMIT
    )
    transcript_books = _sample_corpus_books(
        CORPUS_ROOT, label=TRANSCRIPT_BOOK_TYPE, limit=_LOCAL_PAIR_LIMIT
    )
    assert len(printed_books) >= 3
    assert len(transcript_books) >= 3

    pair_count = min(len(printed_books), len(transcript_books))
    for index in range(pair_count):
        printed_dir, printed = printed_books[index]
        transcript_dir, transcript = transcript_books[index]
        assert printed.book_type_label == PRINTED_BOOK_TYPE, printed_dir
        assert transcript.book_type_label == TRANSCRIPT_BOOK_TYPE, transcript_dir

        ranked = apply_authority_boost(
            [
                _chunk(
                    1,
                    0.7,
                    book_type_label=transcript.book_type_label,
                    author_death_hijri=transcript.main_author_death_hijri,
                    text=transcript.title_ar or transcript_dir.name,
                ),
                _chunk(
                    2,
                    0.7,
                    book_type_label=printed.book_type_label,
                    author_death_hijri=printed.main_author_death_hijri,
                    text=printed.title_ar or printed_dir.name,
                ),
            ]
        )
        assert ranked[0].chunk_id == 2, (printed_dir.name, transcript_dir.name)
        assert ranked[0].score > ranked[1].score


@pytest.mark.skipif(not CORPUS_ROOT.is_dir(), reason="local Shamela corpus not present")
def test_local_shamela_death_hijri_ordering_on_printed_books() -> None:
    printed_books = _sample_corpus_books(CORPUS_ROOT, label=PRINTED_BOOK_TYPE, limit=20)
    known = [
        (path, book)
        for path, book in printed_books
        if book.main_author_death_hijri not in (None, DEATH_YEAR_UNKNOWN)
    ]
    assert len(known) >= 2

    known_sorted = sorted(known, key=lambda item: item[1].main_author_death_hijri or 0)
    earlier_path, earlier = known_sorted[0]
    later_path, later = known_sorted[-1]
    assert earlier.main_author_death_hijri is not None
    assert later.main_author_death_hijri is not None
    assert earlier.main_author_death_hijri < later.main_author_death_hijri

    ranked = apply_authority_boost(
        [
            _chunk(
                1,
                1.0,
                book_type_label=later.book_type_label,
                author_death_hijri=later.main_author_death_hijri,
                text=later.title_ar or later_path.name,
            ),
            _chunk(
                2,
                1.0,
                book_type_label=earlier.book_type_label,
                author_death_hijri=earlier.main_author_death_hijri,
                text=earlier.title_ar or earlier_path.name,
            ),
            _chunk(
                3,
                1.0,
                book_type_label=PRINTED_BOOK_TYPE,
                author_death_hijri=DEATH_YEAR_UNKNOWN,
                text="unknown-death",
            ),
        ],
        order_by_death_hijri=True,
    )
    assert [c.chunk_id for c in ranked] == [2, 1, 3]
