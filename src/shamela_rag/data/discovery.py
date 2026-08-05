"""Corpus discovery: walk the dataset root and locate each book's files.

Layout is ``<root>/<NN>__<category>/<book_id>__<title>/{pages.jsonl,toc.jsonl,book_metadata.json}``.
Folders whose names don't start with ``<digits>__`` (docs/, _meta/, README, ...) are skipped, and a
book missing any of the three files is reported with ``has_all_files=False`` rather than dropped.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

_PREFIX = re.compile(r"^(\d+)__")
_REQUIRED_FILES = ("pages.jsonl", "toc.jsonl", "book_metadata.json")


@dataclass(frozen=True)
class BookLocation:
    book_dir: Path
    book_id: int
    category_id: int | None
    has_all_files: bool


def _parse_id(name: str) -> int | None:
    match = _PREFIX.match(name)
    return int(match.group(1)) if match else None


def discover_books(corpus_root: Path) -> Iterator[BookLocation]:
    for category_dir in sorted(p for p in corpus_root.iterdir() if p.is_dir()):
        category_id = _parse_id(category_dir.name)
        if category_id is None:
            continue
        for book_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            book_id = _parse_id(book_dir.name)
            if book_id is None:
                continue
            has_all = all((book_dir / name).is_file() for name in _REQUIRED_FILES)
            yield BookLocation(
                book_dir=book_dir,
                book_id=book_id,
                category_id=category_id,
                has_all_files=has_all,
            )


def iter_valid_books(corpus_root: Path) -> Iterator[BookLocation]:
    for location in discover_books(corpus_root):
        if location.has_all_files:
            yield location
