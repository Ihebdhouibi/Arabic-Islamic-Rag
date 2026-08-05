from __future__ import annotations

from shamela_rag.chunking.content_roles import ContentRole
from shamela_rag.chunking.merge import Fragment, merge_short_fragments
from shamela_rag.chunking.sizing import SizePolicy

POLICY = SizePolicy(min_tokens=5, max_tokens=10)


def _frag(
    text: str,
    parent: str = "p1",
    role: ContentRole = ContentRole.BODY,
    entity: bool = False,
    spans: tuple[tuple[int, int], ...] = ((0, 1),),
) -> Fragment:
    return Fragment(
        text=text, parent_key=parent, content_role=role, is_named_entity=entity, spans=spans
    )


def test_two_short_siblings_merge_and_preserve_spans() -> None:
    merged = merge_short_fragments(
        [_frag("a b", spans=((0, 3),)), _frag("c d", spans=((3, 6),))], POLICY
    )
    assert len(merged) == 1
    assert merged[0].text == "a b c d"
    assert merged[0].spans == ((0, 3), (3, 6))


def test_named_entity_never_merges() -> None:
    merged = merge_short_fragments([_frag("a b", entity=True), _frag("c d")], POLICY)
    assert len(merged) == 2


def test_different_parent_does_not_merge() -> None:
    merged = merge_short_fragments([_frag("a b", parent="p1"), _frag("c d", parent="p2")], POLICY)
    assert len(merged) == 2


def test_different_content_role_does_not_merge() -> None:
    merged = merge_short_fragments(
        [_frag("a b", role=ContentRole.BODY), _frag("c d", role=ContentRole.FOOTNOTE)], POLICY
    )
    assert len(merged) == 2


def test_merge_blocked_when_result_exceeds_max() -> None:
    tight = SizePolicy(min_tokens=5, max_tokens=3)
    assert len(merge_short_fragments([_frag("a b"), _frag("c d")], tight)) == 2


def test_two_full_size_fragments_do_not_merge() -> None:
    merged = merge_short_fragments([_frag("a b c d e"), _frag("f g h i j")], POLICY)
    assert len(merged) == 2


def test_chained_short_fragments_merge_in_order() -> None:
    merged = merge_short_fragments([_frag("a b"), _frag("c d"), _frag("e f")], POLICY)
    assert len(merged) == 1
    assert merged[0].text == "a b c d e f"
