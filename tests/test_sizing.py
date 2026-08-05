from __future__ import annotations

from shamela_rag.chunking.sizing import SizePolicy, split_section
from shamela_rag.chunking.tokens import count_tokens


def test_section_at_or_under_max_is_single_child() -> None:
    assert split_section("نص قصير", SizePolicy(max_tokens=100)) == ["نص قصير"]


def test_empty_text_yields_no_children() -> None:
    assert split_section("   \r\r ") == []


def test_oversized_section_splits_into_bounded_children() -> None:
    text = " ".join(f"كلمة{i}" for i in range(40))  # 40 tokens, no punctuation
    policy = SizePolicy(max_tokens=10, split_target_tokens=6, overlap_tokens=0)
    children = split_section(text, policy)
    assert len(children) > 1
    assert all(count_tokens(child) <= policy.max_tokens for child in children)


def test_overlap_repeats_trailing_content_within_section() -> None:
    text = "aa bb. cc dd. ee ff. gg hh. ii jj. kk ll."
    policy = SizePolicy(min_tokens=2, max_tokens=10, split_target_tokens=6, overlap_tokens=3)
    children = split_section(text, policy)
    assert children == [
        "aa bb. cc dd.",
        "cc dd. ee ff.",
        "ee ff. gg hh.",
        "gg hh. ii jj.",
        "ii jj. kk ll.",
    ]


def test_thresholds_are_configurable() -> None:
    text = "aa bb cc dd ee ff"
    assert split_section(text, SizePolicy(max_tokens=100)) == [text]
    forced = split_section(text, SizePolicy(max_tokens=3, split_target_tokens=2, overlap_tokens=0))
    assert len(forced) > 1


def test_no_overlap_when_overlap_tokens_zero() -> None:
    text = "aa bb. cc dd. ee ff. gg hh."
    policy = SizePolicy(max_tokens=10, split_target_tokens=6, overlap_tokens=0)
    children = split_section(text, policy)
    assert children == ["aa bb. cc dd.", "ee ff. gg hh."]
