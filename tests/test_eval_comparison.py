from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from shamela_rag.eval.comparison import RunConfig, format_comparison, run_comparison
from shamela_rag.eval.dataset import load_golden_dataset

_GOLDEN = Path(__file__).parent / "fixtures" / "golden_sample.jsonl"


def _canned(mapping: dict[str, list[int]]):
    def retrieve(query: str) -> Sequence[int]:
        return mapping.get(query, [])

    return retrieve


def test_run_comparison_scores_each_config() -> None:
    dataset = load_golden_dataset(_GOLDEN)
    q0, q1 = dataset[0].query, dataset[1].query

    good = RunConfig("good", _canned({q0: [10, 20], q1: [30]}))  # both labeled queries perfect
    weak = RunConfig("weak", _canned({q0: [99], q1: [99]}))  # both miss

    report = run_comparison([good, weak], dataset, ks=(1, 10))

    assert set(report.results) == {"good", "weak"}
    assert report.results["good"].aggregate.recall[10] == 1.0
    assert report.results["weak"].aggregate.recall[10] == 0.0


def test_format_comparison_renders_table() -> None:
    dataset = load_golden_dataset(_GOLDEN)
    report = run_comparison([RunConfig("bge", _canned({}))], dataset, ks=(10, 100))

    table = format_comparison(report)
    assert "config" in table
    assert "recall@10" in table and "recall@100" in table
    assert "ndcg@10" in table
    assert "bge" in table


def test_run_comparison_requires_configs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        run_comparison([], load_golden_dataset(_GOLDEN))


def test_run_comparison_rejects_duplicate_names() -> None:
    dataset = load_golden_dataset(_GOLDEN)
    with pytest.raises(ValueError, match="unique"):
        run_comparison([RunConfig("x", _canned({})), RunConfig("x", _canned({}))], dataset)
