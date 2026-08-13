from __future__ import annotations

import json
from pathlib import Path

import pytest

from shamela_rag import cli
from shamela_rag.data.discovery import BookLocation
from shamela_rag.embeddings.bm25 import Bm25Encoder
from shamela_rag.ingestion.bm25_fit import fit_corpus_bm25


def _write_book(directory: Path, *, book_id: int, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "book_metadata.json").write_text(
        json.dumps({"book_id": book_id, "title_ar": "كتاب"}), encoding="utf-8"
    )
    (directory / "toc.jsonl").write_text("", encoding="utf-8")
    (directory / "pages.jsonl").write_text(
        json.dumps({"page_id": 1, "book_id": book_id, "body": body}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return directory


_BODY_A = "قال الشافعي رحمه الله العلم نور وطلب العلم فريضة على كل مسلم ومسلمة في كل حال"
_BODY_B = "قال مالك في الموطأ عن نافع عن ابن عمر في أحكام الطهارة والوضوء والصلاة"


def test_fit_corpus_bm25_covers_all_books(tmp_path: Path) -> None:
    book_a = _write_book(tmp_path / "a", book_id=1, body=_BODY_A)
    book_b = _write_book(tmp_path / "b", book_id=2, body=_BODY_B)

    encoder = fit_corpus_bm25([book_a, book_b])

    assert encoder.is_fitted
    # Vocabulary spans distinctive terms from both books.
    assert encoder.term_id("الشافعي") is not None
    assert encoder.term_id("مالك") is not None
    assert encoder.encode_query("الشافعي").indices


def test_fit_corpus_bm25_empty_raises(tmp_path: Path) -> None:
    empty = _write_book(tmp_path / "empty", book_id=3, body="")
    with pytest.raises(ValueError, match="empty corpus"):
        fit_corpus_bm25([empty])


def test_build_bm25_cli_writes_loadable_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    book_a = _write_book(tmp_path / "a", book_id=1, body=_BODY_A)
    book_b = _write_book(tmp_path / "b", book_id=2, body=_BODY_B)
    locations = [
        BookLocation(book_dir=book_a, book_id=1, category_id=1, has_all_files=True),
        BookLocation(book_dir=book_b, book_id=2, category_id=1, has_all_files=True),
    ]
    monkeypatch.setattr(cli, "iter_valid_books", lambda _root: locations)
    output = tmp_path / "bm25.json"

    rc = cli.main(["build-bm25", "--all", "--output", str(output)])

    assert rc == 0
    assert output.is_file()
    loaded = Bm25Encoder.load(output)
    assert loaded.term_id("مالك") is not None


def test_build_bm25_cli_no_books_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "iter_valid_books", lambda _root: [])
    rc = cli.main(["build-bm25", "--all", "--output", str(tmp_path / "x.json")])
    assert rc == 1
