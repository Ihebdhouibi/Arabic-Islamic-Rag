"""Evaluation helpers: structural validation and retrieval metrics."""

from __future__ import annotations

from shamela_rag.eval.comparison import (
    ComparisonReport,
    RunConfig,
    format_comparison,
    run_comparison,
)
from shamela_rag.eval.dataset import (
    GoldenExample,
    GoldenSource,
    load_golden_dataset,
)
from shamela_rag.eval.harness import EvalReport, book_ids_from_hits, evaluate_retrieval
from shamela_rag.eval.metrics import (
    AggregateScore,
    QueryScore,
    aggregate,
    dcg_at_k,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_query,
)
from shamela_rag.eval.qwen_quant import (
    QuantComparisonReport,
    QuantVariantMetrics,
    QuantVariantSpec,
    build_recommendation,
    default_variant_specs,
    format_quant_table,
    run_qwen_quant_comparison,
    write_quant_artifacts,
)

__all__ = [
    "AggregateScore",
    "ComparisonReport",
    "EvalReport",
    "GoldenExample",
    "GoldenSource",
    "QuantComparisonReport",
    "QuantVariantMetrics",
    "QuantVariantSpec",
    "QueryScore",
    "RunConfig",
    "aggregate",
    "book_ids_from_hits",
    "build_recommendation",
    "dcg_at_k",
    "default_variant_specs",
    "evaluate_retrieval",
    "format_comparison",
    "format_quant_table",
    "hit_at_k",
    "load_golden_dataset",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "run_comparison",
    "run_qwen_quant_comparison",
    "score_query",
    "write_quant_artifacts",
]
