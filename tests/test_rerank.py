from __future__ import annotations

import os
from collections.abc import Sequence

import pytest

from shamela_rag.retrieval import rerank as rerank_mod
from shamela_rag.retrieval.rerank import (
    CrossEncoderReranker,
    LexicalOverlapReranker,
    RerankCandidate,
    Reranker,
)

_CANDIDATES = [
    RerankCandidate(1, "قال الشافعي في الرسالة إن القياس أصل", {"book_id": 10}),
    RerankCandidate(2, "باب الطهارة والوضوء وأحكام المياه", {"book_id": 20}),
    RerankCandidate(3, "رأي الشافعي في القياس والاجتهاد", {"book_id": 10}),
]


def test_lexical_reranker_orders_by_overlap() -> None:
    reranker = LexicalOverlapReranker()
    ranked = reranker.rerank("رأي الشافعي في القياس", _CANDIDATES)
    assert ranked[0].chunk_id == 3  # most shared tokens
    assert ranked[-1].chunk_id == 2  # unrelated passage
    assert ranked[0].score >= ranked[1].score >= ranked[2].score


def test_rerank_top_k_truncates() -> None:
    reranker = LexicalOverlapReranker()
    ranked = reranker.rerank("الشافعي القياس", _CANDIDATES, top_k=2)
    assert len(ranked) == 2


def test_rerank_empty_candidates() -> None:
    assert LexicalOverlapReranker().rerank("الشافعي", []) == []


def test_rerank_preserves_text_and_payload() -> None:
    ranked = LexicalOverlapReranker().rerank("الطهارة", _CANDIDATES)
    top = next(chunk for chunk in ranked if chunk.chunk_id == 2)
    assert top.text == _CANDIDATES[1].text
    assert top.payload == {"book_id": 20}


class _MiscountingReranker(Reranker):
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [1.0]  # wrong length on purpose


def test_rerank_rejects_score_length_mismatch() -> None:
    with pytest.raises(ValueError, match="scores for"):
        _MiscountingReranker().rerank("q", _CANDIDATES)


class _ConstantTieReranker(Reranker):
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [1.0] * len(passages)


def test_rerank_ties_break_by_chunk_id() -> None:
    ranked = _ConstantTieReranker().rerank("q", _CANDIDATES)
    assert [chunk.chunk_id for chunk in ranked] == [1, 2, 3]


def test_cross_encoder_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        CrossEncoderReranker(batch_size=0)


def test_cross_encoder_import_error_mentions_optional_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> object:
        raise ImportError(
            'CrossEncoderReranker requires optional deps: pip install "shamela-rag[rerank]"'
        )

    monkeypatch.setattr(rerank_mod, "_load_cross_encoder", _boom)
    with pytest.raises(ImportError, match=r"shamela-rag\[rerank\]"):
        CrossEncoderReranker()


def test_cross_encoder_reranks_when_weights_available() -> None:
    """Integration: skipped unless SHAMELA_RUN_RERANK_INTEGRATION=1 and deps/weights exist."""
    if os.environ.get("SHAMELA_RUN_RERANK_INTEGRATION") != "1":
        pytest.skip("set SHAMELA_RUN_RERANK_INTEGRATION=1 to run reranker weight test")
    pytest.importorskip("sentence_transformers")

    try:
        reranker = CrossEncoderReranker(device="cpu", batch_size=4)
    except Exception as exc:  # noqa: BLE001 - absent weights must skip, not fail CI
        pytest.skip(f"reranker weights unavailable: {exc}")

    ranked = reranker.rerank("ما رأي الشافعي في القياس", _CANDIDATES, top_k=1)
    assert ranked[0].chunk_id in {1, 3}  # a Shafiʿi/qiyas passage, not the unrelated one
