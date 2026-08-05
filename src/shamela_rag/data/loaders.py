"""Streaming readers for the raw per-book Shamela files.

Each book directory holds ``pages.jsonl``, ``toc.jsonl`` and ``book_metadata.json``. Readers stream
line by line (files can be very large), skip blank lines, and tolerate the occasional truncated or
malformed line by logging and continuing rather than aborting the whole book.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from shamela_rag.logging_config import get_logger

logger = get_logger(__name__)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at %s:%d", path, lineno)
                continue
            if isinstance(obj, dict):
                yield obj
            else:
                logger.warning("Skipping non-object JSON at %s:%d", path, lineno)


def iter_pages(book_dir: Path) -> Iterator[dict[str, Any]]:
    return iter_jsonl(book_dir / "pages.jsonl")


def iter_toc(book_dir: Path) -> Iterator[dict[str, Any]]:
    return iter_jsonl(book_dir / "toc.jsonl")


def load_book_metadata(book_dir: Path) -> dict[str, Any]:
    with (book_dir / "book_metadata.json").open(encoding="utf-8") as fh:
        data: Any = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"book_metadata.json is not a JSON object: {book_dir}")
    return data
