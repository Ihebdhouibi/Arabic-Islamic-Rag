from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from shamela_rag.eval.dataset import load_golden_dataset
from shamela_rag.eval.harness import book_ids_from_hits, evaluate_retrieval

_GOLDEN = Path(__file__).parent / "fixtures" / "golden_sample.jsonl"


def test_load_golden_dataset_skips_malformed_and_empty() -> None:
    examples = load_golden_dataset(_GOLDEN)

    # g-001, g-002, g-003 are valid; the bad JSON line and empty-query line are dropped.
    ids = [example.example_id for example in examples]
    assert ids == ["g-001", "g-002", "g-003"]

    first = examples[0]
    assert first.relevant_book_ids == {10, 20}
    assert not first.is_adversarial
    assert examples[2].is_adversarial


def test_evaluate_retrieval_scores_against_golden() -> None:
    examples = load_golden_dataset(_GOLDEN)

    # Perfect for g-001 (10 first), miss for g-002, adversarial g-003 returns noise.
    canned: dict[str, list[int]] = {
        examples[0].query: [10, 99, 20],
        examples[1].query: [77, 88],
        examples[2].query: [1, 2, 3],
    }

    def retrieve(query: str) -> Sequence[int]:
        return canned.get(query, [])

    report = evaluate_retrieval(examples, retrieve, ks=(1, 10))

    assert report.aggregate.num_queries == 3
    assert report.aggregate.num_adversarial == 1
    # g-001 hit@1 = 1, g-002 hit@1 = 0 -> mean over the 2 labeled = 0.5
    assert report.aggregate.hit[1] == 0.5
    assert report.aggregate.recall[10] == 0.5  # g-001 recall=1.0, g-002 recall=0.0 -> mean 0.5
    assert report.aggregate.mean_latency_ms is not None
    assert len(report.per_query) == 3


def test_book_ids_from_hits() -> None:
    class _Hit:
        def __init__(self, book_id: int) -> None:
            self.book_id = book_id

    assert book_ids_from_hits([_Hit(3), _Hit(7), object()]) == [3, 7]
