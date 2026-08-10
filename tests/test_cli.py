from __future__ import annotations

from pathlib import Path

import pytest

from shamela_rag import cli
from shamela_rag.data.discovery import BookLocation
from shamela_rag.ingestion.pipeline import BookIngestSummary


def _loc(book_id: int, category_id: int = 1) -> BookLocation:
    return BookLocation(
        book_dir=Path("."), book_id=book_id, category_id=category_id, has_all_files=True
    )


class _FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    def ingest_book(self, location: BookLocation, *, dry_run: bool = False) -> BookIngestSummary:
        self.calls.append((location.book_id, dry_run))
        points = 0 if dry_run else 3
        return BookIngestSummary(location.book_id, 1, 3, points, dry_run=dry_run)


def test_build_parser_reads_ingest_flags() -> None:
    args = cli.build_parser().parse_args(["ingest", "--book", "5", "--dry-run"])
    assert args.command == "ingest"
    assert args.book == 5
    assert args.dry_run is True
    assert args.category is None
    assert args.all is False


def test_ingest_requires_a_target() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["ingest"])


def test_targets_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["ingest", "--book", "1", "--all"])


def test_run_ingest_selects_single_book(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "iter_valid_books", lambda _root: [_loc(1), _loc(2), _loc(3)])
    args = cli.build_parser().parse_args(["ingest", "--book", "2"])
    service = _FakeService()

    assert cli.run_ingest(args, service) == 0  # type: ignore[arg-type]
    assert service.calls == [(2, False)]


def test_run_ingest_selects_category(monkeypatch: pytest.MonkeyPatch) -> None:
    locations = [_loc(1, category_id=1), _loc(2, category_id=2), _loc(3, category_id=1)]
    monkeypatch.setattr(cli, "iter_valid_books", lambda _root: locations)
    args = cli.build_parser().parse_args(["ingest", "--category", "1"])
    service = _FakeService()

    assert cli.run_ingest(args, service) == 0  # type: ignore[arg-type]
    assert service.calls == [(1, False), (3, False)]


def test_run_ingest_all_respects_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "iter_valid_books", lambda _root: [_loc(1), _loc(2), _loc(3)])
    args = cli.build_parser().parse_args(["ingest", "--all", "--limit", "2"])
    service = _FakeService()

    assert cli.run_ingest(args, service) == 0  # type: ignore[arg-type]
    assert service.calls == [(1, False), (2, False)]


def test_run_ingest_passes_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "iter_valid_books", lambda _root: [_loc(1)])
    args = cli.build_parser().parse_args(["ingest", "--all", "--dry-run"])
    service = _FakeService()

    assert cli.run_ingest(args, service) == 0  # type: ignore[arg-type]
    assert service.calls == [(1, True)]


def test_run_ingest_no_match_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "iter_valid_books", lambda _root: [])
    args = cli.build_parser().parse_args(["ingest", "--book", "99"])
    service = _FakeService()

    assert cli.run_ingest(args, service) == 1  # type: ignore[arg-type]
    assert service.calls == []


def test_main_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_main_dispatches_to_run_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(args: object, service: object) -> int:
        captured["service"] = service
        captured["book"] = args.book  # type: ignore[attr-defined]
        return 0

    monkeypatch.setattr(cli, "_build_service", lambda model: "SERVICE")
    monkeypatch.setattr(cli, "run_ingest", fake_run)

    assert cli.main(["ingest", "--book", "7"]) == 0
    assert captured["service"] == "SERVICE"
    assert captured["book"] == 7
