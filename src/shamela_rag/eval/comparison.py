"""Compare retrieval configurations on the golden set (M6-03 / M6-04 tooling).

Each ``RunConfig`` names a retrieval configuration and supplies a ``retrieve`` callable
(query -> ranked book ids); ``run_comparison`` scores every config over the same golden set and
``format_comparison`` renders one table. Model-agnostic: the dense-model comparison and the
chunk-size sweep both feed their own retrieve callables here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from shamela_rag.eval.dataset import GoldenExample
from shamela_rag.eval.harness import DEFAULT_KS, EvalReport, RetrieveFn, evaluate_retrieval


@dataclass(frozen=True)
class RunConfig:
    name: str
    retrieve: RetrieveFn


@dataclass(frozen=True)
class ComparisonReport:
    ks: tuple[int, ...]
    results: dict[str, EvalReport]


def run_comparison(
    configs: Sequence[RunConfig],
    dataset: Sequence[GoldenExample],
    *,
    ks: Sequence[int] = DEFAULT_KS,
) -> ComparisonReport:
    if not configs:
        raise ValueError("configs must be non-empty")
    names = [config.name for config in configs]
    if len(set(names)) != len(names):
        raise ValueError("config names must be unique")

    results = {
        config.name: evaluate_retrieval(dataset, config.retrieve, ks=ks) for config in configs
    }
    return ComparisonReport(ks=tuple(ks), results=results)


def _format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_comparison(report: ComparisonReport) -> str:
    ks = report.ks
    headers = (
        ["config"] + [f"recall@{k}" for k in ks] + ["mrr"] + [f"ndcg@{k}" for k in ks] + ["mean_ms"]
    )
    rows: list[list[str]] = []
    for name, result in report.results.items():
        agg = result.aggregate
        rows.append(
            [name]
            + [_format_float(agg.recall[k]) for k in ks]
            + [_format_float(agg.mrr)]
            + [_format_float(agg.ndcg[k]) for k in ks]
            + [_format_float(agg.mean_latency_ms)]
        )

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows)) if rows else len(headers[col])
        for col in range(len(headers))
    ]
    lines = ["  ".join(cell.ljust(widths[col]) for col, cell in enumerate(headers))]
    lines += ["  ".join(cell.ljust(widths[col]) for col, cell in enumerate(row)) for row in rows]
    return "\n".join(lines) + "\n"
