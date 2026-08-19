"""Arabic root-dictionary loader (inflected form → root(s)).

Loads ``_meta/root_dictionary.jsonl`` (~1.95M lines of
``{"token": "<form>", "roots": ["...", ...]}``) into an in-memory lookup.
This module is the lookup only; the gated sparse expansion field that uses it
lives in ``shamela_rag.embeddings.root_field``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from shamela_rag.data.loaders import iter_jsonl
from shamela_rag.logging_config import get_logger

logger = get_logger(__name__)


class RootDictionary:
    """In-memory inflected-form → root(s) lookup."""

    def __init__(self, mapping: Mapping[str, Sequence[str]]) -> None:
        self._by_token: dict[str, tuple[str, ...]] = {
            token: tuple(roots) for token, roots in mapping.items()
        }

    def __len__(self) -> int:
        return len(self._by_token)

    def __contains__(self, token: object) -> bool:
        return isinstance(token, str) and token in self._by_token

    def lookup(self, token: str) -> tuple[str, ...]:
        """Return root(s) for ``token``, or an empty tuple if unknown."""
        return self._by_token.get(token, ())

    def items(self) -> Iterator[tuple[str, tuple[str, ...]]]:
        return iter(self._by_token.items())


def _parse_entry(obj: dict[str, Any]) -> tuple[str, tuple[str, ...]] | None:
    token = obj.get("token")
    roots = obj.get("roots")
    if not isinstance(token, str) or not token:
        return None
    if not isinstance(roots, list):
        return None
    cleaned: list[str] = []
    for root in roots:
        if isinstance(root, str) and root:
            cleaned.append(root)
    return token, tuple(cleaned)


def iter_root_entries(path: Path) -> Iterator[tuple[str, tuple[str, ...]]]:
    """Yield ``(token, roots)`` from a root-dictionary JSONL file."""
    for obj in iter_jsonl(path):
        parsed = _parse_entry(obj)
        if parsed is None:
            logger.warning("Skipping invalid root-dictionary entry in %s", path)
            continue
        yield parsed


def load_root_dictionary(path: Path) -> RootDictionary:
    """Load the full JSONL into memory.

    Duplicate tokens keep the first seen mapping (later lines are ignored).
    """
    mapping: dict[str, tuple[str, ...]] = {}
    skipped_dupes = 0
    for token, roots in iter_root_entries(path):
        if token in mapping:
            skipped_dupes += 1
            continue
        mapping[token] = roots
    if skipped_dupes:
        logger.info("Ignored %d duplicate token rows in %s", skipped_dupes, path)
    logger.info("Loaded root dictionary from %s (%d tokens)", path, len(mapping))
    return RootDictionary(mapping)
