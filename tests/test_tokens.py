from __future__ import annotations

from shamela_rag.chunking.tokens import HeuristicTokenCounter, count_tokens


def test_counts_words_and_punctuation() -> None:
    assert count_tokens("باب الهمزة") == 2
    assert count_tokens("الصلاة") == 1
    assert count_tokens("hello, world") == 3
    assert count_tokens("قال الشيخ: الحمد لله") == 5
    assert count_tokens("") == 0


def test_is_deterministic() -> None:
    text = "قال الشيخ الإمام الحافظ"
    assert count_tokens(text) == count_tokens(text)


def test_counter_satisfies_protocol() -> None:
    counter = HeuristicTokenCounter()
    assert counter.count("a b c") == 3
