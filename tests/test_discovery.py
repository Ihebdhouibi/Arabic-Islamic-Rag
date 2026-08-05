from __future__ import annotations

from pathlib import Path

from shamela_rag.data.discovery import discover_books, iter_valid_books

_ALL_FILES = ("pages.jsonl", "toc.jsonl", "book_metadata.json")


def _make_book(category_dir: Path, name: str, files: tuple[str, ...]) -> None:
    book_dir = category_dir / name
    book_dir.mkdir(parents=True)
    for filename in files:
        (book_dir / filename).write_text("{}", encoding="utf-8")


def test_discovers_books_and_flags_missing_files(tmp_path: Path) -> None:
    category = tmp_path / "26__biography"
    category.mkdir()
    _make_book(category, "1021__book", _ALL_FILES)
    _make_book(category, "999__incomplete", ("pages.jsonl", "book_metadata.json"))
    (category / "notabook").mkdir()  # no id prefix -> skipped
    (tmp_path / "docs").mkdir()  # not a category -> skipped
    (tmp_path / "README.md").write_text("x", encoding="utf-8")

    by_id = {loc.book_id: loc for loc in discover_books(tmp_path)}
    assert set(by_id) == {1021, 999}
    assert by_id[1021].category_id == 26
    assert by_id[1021].has_all_files is True
    assert by_id[999].has_all_files is False


def test_iter_valid_books_filters_incomplete(tmp_path: Path) -> None:
    category = tmp_path / "03__tafsir"
    category.mkdir()
    _make_book(category, "10__ok", _ALL_FILES)
    _make_book(category, "11__bad", ("pages.jsonl",))
    assert [loc.book_id for loc in iter_valid_books(tmp_path)] == [10]
