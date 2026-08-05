"""Build the structural section tree and each section's context trail.

Most TOC entries nest via ``parent_id``; some leaf entries (e.g. an individual biography like
``باقوم``) carry ``parent_id = null`` yet clearly belong under the preceding heading. For those we
*derive* the trail from the nearest structural heading seen in document order and mark
``path_source = DERIVED_ORDER`` (vs ``EXPLICIT_PARENT``), so a derived nesting is never mistaken for
one the source TOC actually encoded.

An entry is treated as *structural* (something others nest under) if it has an explicit resolvable
parent or is itself referenced as a ``parent_id`` by another entry. That distinguishes a top-level
``باب`` (whose siblings must not be nested under it) from a leaf under a deeper heading.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from shamela_rag.data.models import TocEntry


class PathSource(StrEnum):
    EXPLICIT_PARENT = "explicit_parent"
    DERIVED_ORDER = "derived_order"


@dataclass(frozen=True)
class Section:
    shamela_title_id: int
    title_text: str
    trail: tuple[str, ...]  # root -> leaf, including this entry
    path_source: PathSource
    depth: int
    start_page_id: int
    end_page_id: int | None


def build_sections(entries: Sequence[TocEntry]) -> list[Section]:
    ordered = sorted(entries, key=lambda e: (e.page_id, e.title_id))
    by_title_id = {e.title_id: e for e in entries}
    referenced = {e.parent_id for e in entries if e.parent_id is not None}

    trails: dict[int, tuple[str, ...]] = {}
    sources: dict[int, PathSource] = {}
    context_title_id: int | None = None

    for entry in ordered:
        has_explicit_parent = entry.parent_id is not None and entry.parent_id in by_title_id
        is_structural = has_explicit_parent or entry.title_id in referenced
        is_derived_leaf = (
            entry.parent_id is None
            and entry.title_id not in referenced
            and context_title_id is not None
        )

        if has_explicit_parent:
            assert entry.parent_id is not None
            parent = by_title_id[entry.parent_id]
            parent_trail = trails.get(entry.parent_id, (parent.title_text,))
            trail = (*parent_trail, entry.title_text)
            source = PathSource.EXPLICIT_PARENT
        elif is_derived_leaf:
            assert context_title_id is not None
            trail = (*trails[context_title_id], entry.title_text)
            source = PathSource.DERIVED_ORDER
        else:
            # Top-level heading, or a dangling parent_id: treat as a root.
            trail = (entry.title_text,)
            source = PathSource.EXPLICIT_PARENT

        trails[entry.title_id] = trail
        sources[entry.title_id] = source
        if is_structural:
            context_title_id = entry.title_id

    sections: list[Section] = []
    for index, entry in enumerate(ordered):
        next_page = ordered[index + 1].page_id if index + 1 < len(ordered) else None
        trail = trails[entry.title_id]
        sections.append(
            Section(
                shamela_title_id=entry.shamela_title_id,
                title_text=entry.title_text,
                trail=trail,
                path_source=sources[entry.title_id],
                depth=len(trail) - 1,
                start_page_id=entry.page_id,
                end_page_id=next_page,
            )
        )
    return sections
