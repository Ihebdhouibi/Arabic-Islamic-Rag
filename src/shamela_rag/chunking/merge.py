"""Conditionally merge sub-minimum discursive fragments with an adjacent sibling.

A very short fragment is merged into its neighbour ONLY when every condition holds: neither piece is
a named entity (a named entry must stay atomic), both share the same structural parent and content
role, the combined result stays under ``max_tokens``, and source order is preserved (guaranteed by
the left-to-right pass). Original child offset spans are retained on the merged fragment so the
source mapping survives the merge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from shamela_rag.chunking.content_roles import ContentRole
from shamela_rag.chunking.sizing import SizePolicy
from shamela_rag.chunking.tokens import count_tokens


@dataclass(frozen=True)
class Fragment:
    text: str
    parent_key: str
    content_role: ContentRole
    is_named_entity: bool
    spans: tuple[tuple[int, int], ...]  # original child offset spans, preserved across merges


def _is_short(fragment: Fragment, policy: SizePolicy) -> bool:
    return count_tokens(fragment.text) < policy.min_tokens


def _can_merge(left: Fragment, right: Fragment, policy: SizePolicy) -> bool:
    if left.is_named_entity or right.is_named_entity:
        return False
    if left.parent_key != right.parent_key or left.content_role != right.content_role:
        return False
    if not (_is_short(left, policy) or _is_short(right, policy)):
        return False
    return count_tokens(left.text) + count_tokens(right.text) <= policy.max_tokens


def _merge(left: Fragment, right: Fragment) -> Fragment:
    return Fragment(
        text=f"{left.text} {right.text}",
        parent_key=left.parent_key,
        content_role=left.content_role,
        is_named_entity=False,
        spans=left.spans + right.spans,
    )


def merge_short_fragments(fragments: Sequence[Fragment], policy: SizePolicy) -> list[Fragment]:
    merged: list[Fragment] = []
    for fragment in fragments:
        if merged and _can_merge(merged[-1], fragment, policy):
            merged[-1] = _merge(merged[-1], fragment)
        else:
            merged.append(fragment)
    return merged
