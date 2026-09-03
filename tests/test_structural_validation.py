from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from shamela_rag import cli
from shamela_rag.eval.structural import (
    Severity,
    aggregate_category_audit,
    format_category_audit_report,
    format_report,
    recommend_category,
    stratified_book_locations,
    validate_book,
    validate_category_audit,
    validate_corpus,
)

FIXTURE = Path(__file__).parent / "fixtures" / "book_1021"


def _write_book(
    directory: Path,
    *,
    body: str,
    footnotes: str | None = None,
    toc: list[dict[str, object]] | None = None,
    book_id: int = 1,
    category_id: int | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "book_id": book_id,
        "title_ar": "كتاب",
        "main_author_name_ar": "مؤلف",
    }
    if category_id is not None:
        metadata["category_id"] = category_id
    (directory / "book_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    toc_lines = ""
    if toc:
        toc_lines = "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in toc)
    (directory / "toc.jsonl").write_text(toc_lines, encoding="utf-8")
    page = {"page_id": 1, "book_id": book_id, "body": body, "footnotes": footnotes}
    (directory / "pages.jsonl").write_text(
        json.dumps(page, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return directory


def _write_corpus_book(
    corpus_root: Path,
    *,
    category_id: int,
    book_id: int,
    body: str,
    footnotes: str | None = None,
    toc: list[dict[str, object]] | None = None,
) -> Path:
    book_dir = corpus_root / f"{category_id}__cat" / f"{book_id}__book"
    return _write_book(
        book_dir,
        body=body,
        footnotes=footnotes,
        toc=toc,
        book_id=book_id,
        category_id=category_id,
    )


def test_fixture_book_passes_structural_validation() -> None:
    report = validate_book(FIXTURE, min_content_tokens=1)
    assert report.ok, format_report(report)
    assert report.chunk_count > 0
    assert report.boundary_source_counts.get("inline_toc", 0) > 0
    assert all(f.severity is Severity.SOFT for f in report.findings) or not report.findings


def test_inline_toc_book_passes(tmp_path: Path) -> None:
    body = (
        "<span data-type='title' id=toc-7>باب الهمزة</span>\n"
        "هذا نص طويل فيه كلمات كافية حتى لا يصنف كتنقل فقط. "
        "ونزيد جملا أخرى حتى يتجاوز الحد الأدنى للمحتوى بوضوح."
    )
    book_dir = _write_book(
        tmp_path / "book",
        body=body,
        footnotes="حاشية المحقق على الصفحة.",
        toc=[
            {
                "title_id": 100,
                "book_id": 1,
                "shamela_title_id": 7,
                "title_text": "باب الهمزة",
                "page_id": 1,
                "parent_id": None,
            }
        ],
    )
    report = validate_book(book_dir, min_content_tokens=1)
    assert report.ok, format_report(report)
    assert report.chunk_count >= 2


def test_coverage_detects_uncategorized_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from shamela_rag.eval import structural as structural_mod

    body = (
        "<span data-type='title' id=toc-1>عنوان</span>\n"
        "محتوى حقيقي يجب أن يظهر في القطع وإلا فشل الغطاء."
    )
    book_dir = _write_book(tmp_path / "gap", body=body)
    real_chunk_book = structural_mod.chunk_book

    def truncated_chunk_book(path: Path, **kwargs: Any) -> Any:
        result = real_chunk_book(path, **kwargs)
        return structural_mod.ChunkingResult([], result.stats)

    monkeypatch.setattr(structural_mod, "chunk_book", truncated_chunk_book)
    report = validate_book(book_dir, min_content_tokens=1)
    assert not report.ok
    assert any(f.check == "coverage" and f.severity is Severity.HARD for f in report.findings)


def test_validate_corpus_respects_limit(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    for book_id in (1, 2, 3):
        cat = root / "1__cat" / f"{book_id}__book"
        _write_book(
            cat,
            book_id=book_id,
            body=(
                f"<span data-type='title' id=toc-{book_id}>عنوان</span>\n"
                "نص كاف حتى لا يكون تنقليا فقط مع كلمات إضافية هنا."
            ),
        )
    report = validate_corpus(root, limit=2, min_content_tokens=1)
    assert len(report.books) == 2
    assert report.ok, format_report(report)


def test_stratified_sampling_selects_one_book_per_category(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    for category_id, book_id in ((1, 10), (1, 11), (2, 20), (3, 30)):
        _write_corpus_book(
            root,
            category_id=category_id,
            book_id=book_id,
            body=(
                f"<span data-type='title' id=toc-{book_id}>عنوان</span>\n"
                "نص كاف حتى لا يكون تنقليا فقط مع كلمات إضافية هنا."
            ),
        )
    locations = stratified_book_locations(root, books_per_category=1)
    assert {loc.category_id for loc in locations} == {1, 2, 3}
    assert len(locations) == 3


def test_category_audit_aggregates_boundary_sources(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _write_corpus_book(
        root,
        category_id=1,
        book_id=101,
        body=(
            "<span data-type='title' id=toc-1>باب</span>\n"
            "نص كاف حتى لا يكون تنقليا فقط مع كلمات إضافية هنا."
        ),
        toc=[
            {
                "title_id": 1,
                "book_id": 101,
                "shamela_title_id": 1,
                "title_text": "باب",
                "page_id": 1,
                "parent_id": None,
            }
        ],
    )
    _write_corpus_book(
        root,
        category_id=2,
        book_id=202,
        body="فقرة أولى بدون عناوين.\n\nفقرة ثانية فيها كلمات كافية حتى لا تكون تنقلية.",
        toc=[],
    )
    _, audit = validate_category_audit(root, books_per_category=1, min_content_tokens=1)
    assert audit.category_count == 2
    by_id = {row.category_id: row for row in audit.rows}
    assert by_id[1].boundary_source_counts.get("inline_toc", 0) > 0
    assert by_id[2].boundary_source_counts.get("paragraph_fallback", 0) > 0
    assert "inline_toc" in recommend_category(by_id[1])
    assert "weak ladder rungs" in recommend_category(by_id[2])


def test_format_category_audit_report_includes_table(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _write_corpus_book(
        root,
        category_id=26,
        book_id=1021,
        body=(
            "<span data-type='title' id=toc-1>باب</span>\n"
            "نص كاف حتى لا يكون تنقليا فقط مع كلمات إضافية هنا."
        ),
    )
    _, audit = validate_category_audit(root, books_per_category=1, min_content_tokens=1)
    markdown = format_category_audit_report(audit)
    assert "| category_id | books | boundaries |" in markdown
    assert "## Recommendations" in markdown
    assert "category 26" in markdown


def test_aggregate_category_audit_sums_books(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    for book_id in (1, 2):
        _write_corpus_book(
            root,
            category_id=5,
            book_id=book_id,
            body=(
                f"<span data-type='title' id=toc-{book_id}>عنوان</span>\n"
                "نص كاف حتى لا يكون تنقليا فقط مع كلمات إضافية هنا."
            ),
        )
    corpus = validate_corpus(
        root, stratified=True, books_per_category=2, limit=None, min_content_tokens=1
    )
    audit = aggregate_category_audit(corpus, books_per_category=2)
    assert audit.rows[0].book_count == 2


def test_cli_validate_structure_book_dir() -> None:
    assert cli.main(["validate-structure", "--book-dir", str(FIXTURE)]) == 0


def test_cli_validate_structure_parser_defaults() -> None:
    args = cli.build_parser().parse_args(["validate-structure", "--corpus-root", "."])
    assert args.command == "validate-structure"
    assert args.limit == 20
    assert args.book_dir is None


def test_cli_audit_structure(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _write_corpus_book(
        root,
        category_id=1,
        book_id=1,
        body=(
            "<span data-type='title' id=toc-1>عنوان</span>\n"
            "نص كاف حتى لا يكون تنقليا فقط مع كلمات إضافية هنا."
        ),
    )
    output = tmp_path / "audit.md"
    assert (
        cli.main(
            [
                "audit-structure",
                "--corpus-root",
                str(root),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.is_file()
    assert "Structural chunking per-category audit" in output.read_text(encoding="utf-8")
