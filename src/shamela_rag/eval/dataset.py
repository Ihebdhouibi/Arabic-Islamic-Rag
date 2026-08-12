"""Golden evaluation dataset loader (M6-01 staging format).

Each JSONL line is ``{"id", "use_case", "query", "expected_sources": [{"book_title",
"internal_book_id", "shamela_page_id", "confidence"}]}``. Examples with no expected sources are
adversarial ("should surface nothing"). Parsing is tolerant: malformed lines/sources are skipped.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shamela_rag.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class GoldenSource:
    book_id: int
    shamela_page_id: int | None
    confidence: str
    book_title: str


@dataclass(frozen=True)
class GoldenExample:
    example_id: str
    query: str
    sources: tuple[GoldenSource, ...]

    @property
    def is_adversarial(self) -> bool:
        return not self.sources

    @property
    def relevant_book_ids(self) -> set[int]:
        return {source.book_id for source in self.sources}


def _parse_source(obj: dict[str, Any]) -> GoldenSource | None:
    book_id = obj.get("internal_book_id")
    if not isinstance(book_id, int):
        return None
    page = obj.get("shamela_page_id")
    return GoldenSource(
        book_id=book_id,
        shamela_page_id=page if isinstance(page, int) else None,
        confidence=str(obj.get("confidence", "")),
        book_title=str(obj.get("book_title", "")),
    )


def _parse_example(obj: dict[str, Any]) -> GoldenExample | None:
    example_id = obj.get("id")
    query = obj.get("query")
    if not isinstance(example_id, str) or not isinstance(query, str) or not query.strip():
        return None
    raw_sources = obj.get("expected_sources")
    sources: list[GoldenSource] = []
    if isinstance(raw_sources, list):
        for raw in raw_sources:
            if isinstance(raw, dict):
                parsed = _parse_source(raw)
                if parsed is not None:
                    sources.append(parsed)
    return GoldenExample(example_id=example_id, query=query, sources=tuple(sources))


def iter_golden_examples(path: Path) -> Iterator[GoldenExample]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed golden line %s:%d", path, line_number)
                continue
            example = _parse_example(obj) if isinstance(obj, dict) else None
            if example is None:
                logger.warning("Skipping invalid golden example %s:%d", path, line_number)
                continue
            yield example


def load_golden_dataset(path: Path) -> list[GoldenExample]:
    return list(iter_golden_examples(path))
