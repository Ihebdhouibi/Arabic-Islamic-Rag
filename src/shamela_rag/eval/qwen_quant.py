"""Qwen3 quantization footprint + quality comparison (issue #135)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from shamela_rag.embeddings.qwen import (
    QWEN3_EMBEDDING_MODEL_ID,
    QuantizationMode,
    Qwen3EmbeddingProvider,
)
from shamela_rag.eval.comparison import (
    ComparisonReport,
    RunConfig,
    format_comparison,
    run_comparison,
)
from shamela_rag.eval.dataset import GoldenExample, load_golden_dataset

ProgressFn = Callable[[str], None]

_BASELINE_NAMES: frozenset[str] = frozenset({"fp16-baseline", "gguf-q8-baseline"})


class EvalChunkRow(TypedDict):
    chunk_id: str
    book_id: int
    text: str


DEFAULT_PROBE_TEXTS: tuple[str, ...] = (
    "باب الهمزة",
    "ما هي الصلاة",
    "What is the capital of the Abbasid caliphate?",
    "روى البخاري عن عائشة رضي الله عنها",
    "The jurisprudential difference between wudu and ghusl",
)


@dataclass(frozen=True)
class QuantVariantSpec:
    name: str
    quantization: QuantizationMode | None = None
    gguf_path: Path | None = None
    device: str | None = None
    gguf_n_ctx: int = 512


@dataclass
class QuantVariantMetrics:
    name: str
    quantization: str
    load_seconds: float
    peak_rss_mb: float | None
    peak_vram_mb: float | None
    mean_embed_ms: float
    mean_cosine_vs_baseline: float | None
    embed_count: int
    error: str | None = None


@dataclass
class QuantComparisonReport:
    model_id: str
    variants: list[QuantVariantMetrics]
    retrieval: ComparisonReport | None = None
    recommendation: str = ""
    probe_texts: list[str] = field(default_factory=list)


def _rss_mb() -> float | None:
    try:
        import resource

        getrusage = getattr(resource, "getrusage", None)
        rusage_self = getattr(resource, "RUSAGE_SELF", None)
        if getrusage is None or rusage_self is None:
            raise AttributeError("resource.getrusage unavailable")
        usage = getrusage(rusage_self).ru_maxrss
        if usage > 10_000_000:
            return float(usage / (1024 * 1024))
        return float(usage / 1024)
    except (ImportError, AttributeError, ValueError):
        pass
    try:
        import importlib

        psutil = importlib.import_module("psutil")
        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except Exception:  # noqa: BLE001 - optional probe
        return None


def _vram_mb() -> float | None:
    try:
        import importlib

        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:  # noqa: BLE001 - optional probe
        return None


def _reset_cuda_peak() -> None:
    try:
        import importlib

        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"cosine length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


def _mean_cosine(baseline: Sequence[Sequence[float]], other: Sequence[Sequence[float]]) -> float:
    if len(baseline) != len(other):
        raise ValueError("vector lists must have equal length")
    if not baseline:
        return 0.0
    return float(sum(_cosine(a, b) for a, b in zip(baseline, other, strict=True)) / len(baseline))


def default_variant_specs(
    *,
    include_fp16: bool = True,
    include_int8: bool = True,
    include_int4: bool = True,
    gguf_path: Path | None = None,
    gguf_baseline_path: Path | None = None,
    device: str | None = None,
    gguf_n_ctx: int = 512,
) -> list[QuantVariantSpec]:
    specs: list[QuantVariantSpec] = []
    if include_fp16:
        specs.append(QuantVariantSpec(name="fp16-baseline", quantization=None, device=device))
    if include_int8:
        specs.append(QuantVariantSpec(name="int8", quantization="int8", device=device))
    if include_int4:
        specs.append(QuantVariantSpec(name="int4", quantization="int4", device=device))
    if gguf_baseline_path is not None:
        specs.append(
            QuantVariantSpec(
                name="gguf-q8-baseline",
                quantization="gguf",
                gguf_path=gguf_baseline_path,
                device=device,
                gguf_n_ctx=gguf_n_ctx,
            )
        )
    if gguf_path is not None:
        specs.append(
            QuantVariantSpec(
                name="gguf",
                quantization="gguf",
                gguf_path=gguf_path,
                device=device,
                gguf_n_ctx=gguf_n_ctx,
            )
        )
    if not specs:
        raise ValueError("at least one variant must be enabled")
    return specs


def _baseline_first(specs: Sequence[QuantVariantSpec]) -> list[QuantVariantSpec]:
    """Run baseline-named specs before candidates so cosine vs baseline is always computed."""
    baselines = [spec for spec in specs if spec.name in _BASELINE_NAMES]
    candidates = [spec for spec in specs if spec.name not in _BASELINE_NAMES]
    return baselines + candidates


def _build_provider(spec: QuantVariantSpec) -> Qwen3EmbeddingProvider:
    return Qwen3EmbeddingProvider(
        device=None if spec.quantization in ("int8", "int4") else spec.device,
        quantization=spec.quantization,
        gguf_path=spec.gguf_path,
        batch_size=8 if spec.quantization else 32,
        gguf_n_ctx=spec.gguf_n_ctx,
    )


def measure_variant(
    spec: QuantVariantSpec,
    texts: Sequence[str],
    *,
    baseline_vectors: Sequence[Sequence[float]] | None = None,
    progress: ProgressFn | None = None,
) -> tuple[QuantVariantMetrics, list[list[float]] | None]:
    log = progress or (lambda _m: None)
    log(f"loading {spec.name} (quantization={spec.quantization!r})")
    _reset_cuda_peak()
    t0 = time.perf_counter()
    try:
        provider = _build_provider(spec)
    except Exception as exc:  # noqa: BLE001 - record and continue other variants
        return (
            QuantVariantMetrics(
                name=spec.name,
                quantization=str(spec.quantization or "none"),
                load_seconds=time.perf_counter() - t0,
                peak_rss_mb=_rss_mb(),
                peak_vram_mb=_vram_mb(),
                mean_embed_ms=0.0,
                mean_cosine_vs_baseline=None,
                embed_count=0,
                error=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )

    load_s = time.perf_counter() - t0
    log(f"{spec.name} loaded in {load_s:.1f}s (dims={provider.dims})")
    try:
        t1 = time.perf_counter()
        texts_list = list(texts)
        vectors: list[list[float]] = []
        total = len(texts_list)
        report_every = max(1, total // 10) if total >= 10 else 1
        # Per-text embeds (not batch_size) so GGUF/ST mean_embed_ms stay comparable.
        for index, text in enumerate(texts_list, start=1):
            vectors.append(provider.embed_documents([text])[0])
            if index == total or index % report_every == 0:
                log(f"{spec.name} embedded {index}/{total} texts")
        elapsed_ms = (time.perf_counter() - t1) * 1000.0
        mean_ms = elapsed_ms / max(1, len(texts_list))
        cosine = _mean_cosine(baseline_vectors, vectors) if baseline_vectors is not None else None
        metrics = QuantVariantMetrics(
            name=spec.name,
            quantization=str(spec.quantization or "none"),
            load_seconds=load_s,
            peak_rss_mb=_rss_mb(),
            peak_vram_mb=_vram_mb(),
            mean_embed_ms=mean_ms,
            mean_cosine_vs_baseline=cosine,
            embed_count=len(texts_list),
        )
        return metrics, vectors
    finally:
        provider.close()
        log(f"{spec.name} released")


def _rank_books_by_dense(
    query_vector: Sequence[float],
    chunk_vectors: Sequence[Sequence[float]],
    book_ids: Sequence[int],
    *,
    limit: int,
) -> list[int]:
    scored = [
        (_cosine(query_vector, vec), book)
        for vec, book in zip(chunk_vectors, book_ids, strict=True)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    ranked: list[int] = []
    seen: set[int] = set()
    for _, book in scored:
        if book in seen:
            continue
        seen.add(book)
        ranked.append(book)
        if len(ranked) >= limit:
            break
    return ranked


def run_quant_retrieval(
    dataset: Sequence[GoldenExample],
    _chunk_texts: Sequence[str],
    chunk_book_ids: Sequence[int],
    dense_by_variant: Mapping[str, Sequence[Sequence[float]]],
    query_vectors_by_variant: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    candidate_limit: int = 100,
    ks: Sequence[int] = (10, 100),
) -> ComparisonReport:
    configs: list[RunConfig] = []
    for name, doc_vectors in dense_by_variant.items():
        qmap = query_vectors_by_variant[name]

        def _retrieve(
            query: str,
            *,
            _docs: Sequence[Sequence[float]] = doc_vectors,
            _books: Sequence[int] = chunk_book_ids,
            _qmap: Mapping[str, Sequence[float]] = qmap,
        ) -> list[int]:
            return _rank_books_by_dense(_qmap[query], _docs, _books, limit=candidate_limit)

        configs.append(RunConfig(name, _retrieve))
    return run_comparison(configs, dataset, ks=ks)


def build_recommendation(variants: Sequence[QuantVariantMetrics]) -> str:
    ok = [v for v in variants if v.error is None]
    if not ok:
        return (
            "No quantized variant loaded successfully. Prefer BGE-M3 for constrained hosts "
            "until a working int8/GGUF path is verified on target hardware."
        )
    baseline = next(
        (v for v in ok if v.name in _BASELINE_NAMES),
        None,
    )
    if baseline is None:
        gguf = next((v for v in ok if v.name == "gguf"), None)
        if gguf is not None:
            return (
                f"fp16 baseline unavailable on this host; GGUF ({gguf.quantization}) loaded in "
                f"{gguf.load_seconds:.1f}s, ~{gguf.peak_rss_mb!s} MB RSS, "
                f"{gguf.mean_embed_ms:.1f} ms/text. Use GGUF for CPU/16GB M6-03 trials; "
                "confirm quality vs fp16 on a GPU box before production."
            )
        only = ok[0]
        return (
            f"Only {only.name} ran (no fp16 baseline on this host). "
            "Re-measure against fp16 on a GPU / ≥32GB machine before locking ADR-002."
        )
    candidates = [v for v in ok if v.name != baseline.name]
    if not candidates:
        return (
            f"Only {baseline.name} ran. Re-run with int8 (CUDA + bitsandbytes) or GGUF before "
            "choosing a quantized path for M6-03 / production."
        )

    def _acceptable(v: QuantVariantMetrics) -> bool:
        return v.mean_cosine_vs_baseline is not None and v.mean_cosine_vs_baseline >= 0.95

    good = [v for v in candidates if _acceptable(v)]
    if not good:
        unverified = [v for v in candidates if v.mean_cosine_vs_baseline is None]
        if unverified and all(v.mean_cosine_vs_baseline is None for v in candidates):
            names = ", ".join(v.name for v in unverified)
            return (
                f"Candidates ran without cosine vs {baseline.name} ({names}); "
                "not auto-recommendable. Re-run so the baseline is measured first."
            )
        best = max(
            candidates,
            key=lambda v: v.mean_cosine_vs_baseline or 0.0,
        )
        return (
            f"No variant retained >=0.95 mean cosine vs {baseline.name}. "
            f"Closest was {best.name} "
            f"(cosine={best.mean_cosine_vs_baseline!s}). "
            "Do not use quantized Qwen for M6-03 until quality recovers or GGUF is validated."
        )

    winner = min(
        good,
        key=lambda v: (
            v.peak_vram_mb if v.peak_vram_mb is not None else 1e12,
            v.peak_rss_mb if v.peak_rss_mb is not None else 1e12,
            v.load_seconds,
        ),
    )
    cos = (
        f"{winner.mean_cosine_vs_baseline:.4f}"
        if winner.mean_cosine_vs_baseline is not None
        else "n/a"
    )
    vs_label = "fp16" if baseline.name == "fp16-baseline" else baseline.name
    return (
        f"Recommend {winner.name} ({winner.quantization}) for M6-03 / constrained hosts: "
        f"mean cosine vs {vs_label}={cos}, "
        f"load={winner.load_seconds:.1f}s, "
        f"VRAM~{winner.peak_vram_mb!s} MB, RSS~{winner.peak_rss_mb!s} MB. "
        "Confirm with golden-set Recall@100 / MRR / nDCG before full-corpus embed "
        "(and vs real fp16 on a GPU box if baseline was GGUF-Q8)."
    )


def run_qwen_quant_comparison(
    specs: Sequence[QuantVariantSpec],
    *,
    probe_texts: Sequence[str] | None = None,
    chunks_path: Path | None = None,
    corpus_root: Path | None = None,
    golden_path: Path | None = None,
    chunk_cache_path: Path | None = None,
    force_rechunk: bool = False,
    candidate_limit: int = 100,
    max_chunks: int | None = 256,
    progress: ProgressFn | None = None,
) -> QuantComparisonReport:
    log = progress or (lambda _m: None)
    texts: list[str]
    book_ids: list[int] | None = None
    dataset: list[GoldenExample] | None = None

    if golden_path is not None:
        dataset = list(load_golden_dataset(golden_path))
        log(f"loaded {len(dataset)} golden queries from {golden_path}")

    if chunks_path is not None:
        texts, book_ids = _load_chunk_texts(chunks_path, max_chunks=max_chunks)
        log(f"loaded {len(texts)} chunks from {chunks_path}")
    elif corpus_root is not None:
        if dataset is None:
            raise ValueError("--corpus-root for retrieval requires --golden")
        cache = chunk_cache_path or Path("artifacts/qwen-quant/eval_chunks.jsonl")
        if cache.is_file() and not force_rechunk:
            log(f"reusing chunk cache {cache} (pass --force-rechunk to rebuild)")
            rows = _load_eval_chunk_rows(cache)
        else:
            rows = build_golden_eval_chunks(dataset, corpus_root, progress=log)
            save_eval_chunks_jsonl(cache, rows)
            log(f"wrote {len(rows)} chunks to {cache}")
        if max_chunks is not None and len(rows) > max_chunks:
            rows = subsample_eval_chunks(rows, max_chunks=max_chunks)
            log(f"using {len(rows)} chunks (max_chunks={max_chunks})")
        texts = [str(row["text"]) for row in rows]
        book_ids = [int(row["book_id"]) for row in rows]
    else:
        texts = list(probe_texts or DEFAULT_PROBE_TEXTS)
        log(f"using {len(texts)} probe texts (no --chunks / --corpus-root)")

    metrics_rows: list[QuantVariantMetrics] = []
    dense_by_variant: dict[str, list[list[float]]] = {}
    baseline_vectors: list[list[float]] | None = None
    ordered_specs = _baseline_first(specs)

    for spec in ordered_specs:
        row, vectors = measure_variant(
            spec,
            texts,
            baseline_vectors=baseline_vectors,
            progress=log,
        )
        metrics_rows.append(row)
        if vectors is None:
            continue
        if baseline_vectors is None and spec.name in _BASELINE_NAMES:
            baseline_vectors = vectors
        dense_by_variant[spec.name] = vectors

    retrieval: ComparisonReport | None = None
    if dataset is not None and book_ids is not None and dense_by_variant:
        log("embedding golden queries per successful variant")
        query_maps: dict[str, dict[str, list[float]]] = {}
        for spec in ordered_specs:
            if spec.name not in dense_by_variant:
                continue
            provider = None
            try:
                provider = _build_provider(spec)
                query_maps[spec.name] = {
                    example.query: provider.embed_query(example.query) for example in dataset
                }
            except Exception as exc:  # noqa: BLE001
                log(f"skip retrieval for {spec.name}: {exc}")
            finally:
                if provider is not None:
                    provider.close()
        if query_maps:
            retrieval = run_quant_retrieval(
                dataset,
                texts,
                book_ids,
                dense_by_variant={k: dense_by_variant[k] for k in query_maps},
                query_vectors_by_variant=query_maps,
                candidate_limit=candidate_limit,
            )

    recommendation = build_recommendation(metrics_rows)
    return QuantComparisonReport(
        model_id=QWEN3_EMBEDDING_MODEL_ID,
        variants=metrics_rows,
        retrieval=retrieval,
        recommendation=recommendation,
        probe_texts=list(texts) if chunks_path is None else [],
    )


def save_eval_chunks_jsonl(path: Path, rows: Sequence[EvalChunkRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def subsample_eval_chunks(rows: Sequence[EvalChunkRow], *, max_chunks: int) -> list[EvalChunkRow]:
    if max_chunks <= 0:
        raise ValueError(f"max_chunks must be positive, got {max_chunks}")
    ordered = list(rows)
    if len(ordered) <= max_chunks:
        return ordered
    by_book: dict[int, list[EvalChunkRow]] = {}
    for row in ordered:
        by_book.setdefault(int(row["book_id"]), []).append(row)
    out: list[EvalChunkRow] = []
    book_ids = sorted(by_book)
    while len(out) < max_chunks:
        added = False
        for book_id in book_ids:
            bucket = by_book[book_id]
            if not bucket:
                continue
            out.append(bucket.pop(0))
            added = True
            if len(out) >= max_chunks:
                break
        if not added:
            break
    return out


def build_golden_eval_chunks(
    dataset: Sequence[GoldenExample],
    corpus_root: Path,
    *,
    progress: ProgressFn | None = None,
) -> list[EvalChunkRow]:
    from shamela_rag.chunking.orchestrator import chunk_book
    from shamela_rag.data.discovery import iter_valid_books
    from shamela_rag.ingestion.pipeline import dense_input

    log = progress or (lambda _m: None)
    book_ids = sorted({book_id for example in dataset for book_id in example.relevant_book_ids})
    by_id = {
        location.book_id: location
        for location in iter_valid_books(corpus_root)
        if location.has_all_files
    }
    rows: list[EvalChunkRow] = []
    for index, book_id in enumerate(book_ids, start=1):
        location = by_id.get(book_id)
        if location is None:
            log(f"skip book {book_id}: not found under {corpus_root}")
            continue
        log(f"chunking golden book {book_id} ({index}/{len(book_ids)})")
        result = chunk_book(location.book_dir)
        for chunk_index, chunk in enumerate(result.chunks):
            rows.append(
                {
                    "chunk_id": f"{book_id}:{chunk_index}",
                    "book_id": book_id,
                    "text": dense_input(chunk),
                }
            )
    if not rows:
        raise ValueError(f"no chunks built from golden books under {corpus_root}")
    return rows


def _load_eval_chunk_rows(path: Path) -> list[EvalChunkRow]:
    rows: list[EvalChunkRow] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            book_id = raw.get("book_id")
            text = raw.get("text")
            chunk_id = raw.get("chunk_id", "")
            if book_id is None or text is None:
                raise ValueError(f"chunk row missing book_id/text in {path}")
            rows.append(
                {
                    "chunk_id": str(chunk_id),
                    "book_id": int(book_id),
                    "text": str(text),
                }
            )
    if not rows:
        raise ValueError(f"no chunks found in {path}")
    return rows


def _load_chunk_texts(path: Path, *, max_chunks: int | None) -> tuple[list[str], list[int]]:
    rows = _load_eval_chunk_rows(path)
    if max_chunks is not None and len(rows) > max_chunks:
        rows = subsample_eval_chunks(rows, max_chunks=max_chunks)
    texts = [str(row["text"]) for row in rows]
    books = [int(row["book_id"]) for row in rows]
    return texts, books


def format_quant_table(report: QuantComparisonReport) -> str:
    headers = [
        "variant",
        "quant",
        "load_s",
        "rss_mb",
        "vram_mb",
        "ms/text",
        "cos_vs_fp16",
        "error",
    ]
    rows: list[list[str]] = []
    for v in report.variants:
        rows.append(
            [
                v.name,
                v.quantization,
                f"{v.load_seconds:.1f}",
                "n/a" if v.peak_rss_mb is None else f"{v.peak_rss_mb:.0f}",
                "n/a" if v.peak_vram_mb is None else f"{v.peak_vram_mb:.0f}",
                f"{v.mean_embed_ms:.1f}",
                "n/a" if v.mean_cosine_vs_baseline is None else f"{v.mean_cosine_vs_baseline:.4f}",
                v.error or "",
            ]
        )
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines += ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    lines.append("")
    lines.append("Recommendation:")
    lines.append(report.recommendation)
    if report.retrieval is not None:
        lines.append("")
        lines.append("Dense retrieval (shared chunks):")
        lines.append(format_comparison(report.retrieval).rstrip("\n"))
    return "\n".join(lines) + "\n"


def write_quant_artifacts(output_dir: Path, report: QuantComparisonReport) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_table.md").write_text(
        "# Qwen3 quantization comparison\n\n```\n"
        + format_quant_table(report)
        + "```\n\n"
        + "## Recommendation\n\n"
        + report.recommendation
        + "\n",
        encoding="utf-8",
    )
    payload: dict[str, Any] = {
        "model_id": report.model_id,
        "recommendation": report.recommendation,
        "variants": [asdict(v) for v in report.variants],
        "probe_texts": report.probe_texts,
    }
    if report.retrieval is not None:
        payload["retrieval"] = {
            name: {
                "mrr": result.aggregate.mrr,
                "recall": dict(result.aggregate.recall),
                "ndcg": dict(result.aggregate.ndcg),
                "mean_latency_ms": result.aggregate.mean_latency_ms,
            }
            for name, result in report.retrieval.results.items()
        }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
