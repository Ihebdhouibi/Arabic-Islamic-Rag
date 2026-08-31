"""M6-03 dense-model comparison: dense-only, hybrid BM25, and BGE sparse ablation."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from shamela_rag.embeddings.bge_m3 import BgeM3EmbeddingProvider, SparseEmbedding
from shamela_rag.embeddings.bm25 import Bm25Encoder, SparseVector
from shamela_rag.embeddings.provider import EmbeddingProvider
from shamela_rag.eval.comparison import (
    ComparisonReport,
    RunConfig,
    format_comparison,
    run_comparison,
)
from shamela_rag.eval.dataset import GoldenExample, load_golden_dataset
from shamela_rag.eval.qwen_quant import _cosine, run_quant_retrieval
from shamela_rag.retrieval.fusion import reciprocal_rank_fusion
from shamela_rag.retrieval.results import RetrievedChunk

ProgressFn = Callable[[str], None]


def _log(progress: ProgressFn | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _texts_fingerprint(texts: Sequence[str]) -> str:
    """Content hash of the exact chunk set a cache was built from.

    A count-only check (``len(cached) == len(texts)``) would silently accept a cache built from a
    different ``--chunks`` file (or a regenerated one) that happens to subsample to the same count
    -- pairing stale vectors with the current texts/book_ids with no error. Hashing the actual
    texts closes that: any change to the underlying chunk set changes the fingerprint.
    """
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _sparse_dot(
    query: SparseVector | SparseEmbedding,
    document: SparseVector | SparseEmbedding,
) -> float:
    weights = dict(zip(document.indices, document.values, strict=True))
    return float(
        sum(
            weights.get(idx, 0.0) * value
            for idx, value in zip(query.indices, query.values, strict=True)
        )
    )


def _sparse_to_dict(sparse: SparseVector | SparseEmbedding) -> dict[str, list[int] | list[float]]:
    return {"indices": list(sparse.indices), "values": [float(v) for v in sparse.values]}


def _sparse_from_dict(raw: Mapping[str, Any]) -> SparseVector:
    indices = raw.get("indices")
    values = raw.get("values")
    if not isinstance(indices, list) or not isinstance(values, list):
        raise ValueError("sparse cache entry must have indices and values lists")
    return SparseVector(
        indices=[int(x) for x in indices],
        values=[float(x) for x in values],
    )


def _load_chunk_texts(path: Path, *, max_chunks: int | None) -> tuple[list[str], list[int]]:
    """Load eval chunk JSONL; optionally round-robin subsample without keeping unused rows."""
    if max_chunks is None:
        from shamela_rag.eval.qwen_quant import _load_chunk_texts as _full

        return _full(path, max_chunks=None)

    by_book: dict[int, list[str]] = {}
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            book_id = raw.get("book_id")
            text = raw.get("text")
            if book_id is None or text is None:
                continue
            by_book.setdefault(int(book_id), []).append(str(text))
            total += 1
    if not by_book:
        raise ValueError(f"no chunks found in {path}")
    if total <= max_chunks:
        texts: list[str] = []
        books: list[int] = []
        for book_id, bucket in sorted(by_book.items()):
            texts.extend(bucket)
            books.extend([book_id] * len(bucket))
        return texts, books

    out_texts: list[str] = []
    out_books: list[int] = []
    book_ids = sorted(by_book)
    while len(out_texts) < max_chunks:
        added = False
        for book_id in book_ids:
            bucket = by_book[book_id]
            if not bucket:
                continue
            out_texts.append(bucket.pop(0))
            out_books.append(book_id)
            added = True
            if len(out_texts) >= max_chunks:
                break
        if not added:
            break
    return out_texts, out_books


def _embed_texts_with_progress(
    provider: EmbeddingProvider,
    texts: Sequence[str],
    *,
    label: str,
    batch_size: int,
    progress: ProgressFn | None,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    total = len(texts)
    report_every = max(1, total // 20) if total >= 20 else 1
    t0 = time.perf_counter()
    for start in range(0, total, batch_size):
        batch = list(texts[start : start + batch_size])
        vectors.extend(provider.embed_documents(batch))
        done = min(start + batch_size, total)
        if done == total or done <= batch_size or done % report_every < batch_size:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            _log(
                progress,
                f"{label}: embedded {done}/{total} texts ({elapsed:.1f}s, {rate:.1f} texts/s)",
            )
    return vectors


def _embed_queries_with_progress(
    provider: EmbeddingProvider,
    dataset: Sequence[GoldenExample],
    *,
    label: str,
    progress: ProgressFn | None,
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    total = len(dataset)
    t0 = time.perf_counter()
    for index, example in enumerate(dataset, start=1):
        out[example.query] = provider.embed_query(example.query)
        if index == total or index == 1 or index % max(1, total // 10) == 0:
            _log(
                progress,
                f"{label}: embedded queries {index}/{total} ({time.perf_counter() - t0:.1f}s)",
            )
    return out


def _cache_path(output_dir: Path, model_name: str, kind: str) -> Path:
    safe = model_name.replace("/", "_")
    return output_dir / f"{safe}_{kind}.json"


def _save_vector_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _load_doc_cache(path: Path, *, fingerprint: str) -> list[list[float]] | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("fingerprint") != fingerprint:
        return None
    vectors = raw.get("vectors")
    if not isinstance(vectors, list):
        return None
    return [[float(x) for x in row] for row in vectors]


def _load_query_cache(path: Path, *, fingerprint: str) -> dict[str, list[float]] | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("fingerprint") != fingerprint:
        return None
    mapping = raw.get("queries")
    if not isinstance(mapping, dict):
        return None
    return {str(k): [float(x) for x in v] for k, v in mapping.items()}


def _write_stage_artifacts(
    output_dir: Path,
    *,
    stage: str,
    golden_path: Path,
    chunks_path: Path,
    max_chunks: int,
    chunk_count: int,
    book_coverage: int,
    candidate_limit: int,
    models: Sequence[str],
    report: ComparisonReport,
    extra: Mapping[str, Any] | None = None,
) -> str:
    table = format_comparison(report)
    stem = stage.replace("-", "_")
    (output_dir / f"{stem}_table.md").write_text(
        f"# M6-03 {stage}\n\n```\n{table}```\n",
        encoding="utf-8",
    )
    metrics: dict[str, Any] = {
        "stage": stage,
        "golden": str(golden_path),
        "chunks": str(chunks_path),
        "max_chunks": max_chunks,
        "chunk_count": chunk_count,
        "book_coverage": book_coverage,
        "candidate_limit": candidate_limit,
        "models": list(models),
        "results": {
            name: {
                "mrr": result.aggregate.mrr,
                "recall": {str(k): v for k, v in result.aggregate.recall.items()},
                "ndcg": {str(k): v for k, v in result.aggregate.ndcg.items()},
                "mean_latency_ms": result.aggregate.mean_latency_ms,
            }
            for name, result in report.results.items()
        },
    }
    if extra:
        metrics.update(dict(extra))
    (output_dir / f"{stem}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return table


def _prepare_dense_caches(
    *,
    models: dict[str, EmbeddingProvider],
    dataset: Sequence[GoldenExample],
    texts: Sequence[str],
    output_dir: Path,
    batch_size: int,
    force_reembed: bool,
    progress: ProgressFn | None,
    require_cache: bool = False,
) -> tuple[dict[str, list[list[float]]], dict[str, dict[str, list[float]]]]:
    dense_by_model: dict[str, list[list[float]]] = {}
    query_maps: dict[str, dict[str, list[float]]] = {}
    fingerprint = _texts_fingerprint(texts)

    for name, provider in models.items():
        _log(progress, f"=== model={name} dims={provider.dims} ===")
        doc_cache = _cache_path(output_dir, name, "docs")
        query_cache = _cache_path(output_dir, name, "queries")

        doc_vectors = None if force_reembed else _load_doc_cache(doc_cache, fingerprint=fingerprint)
        if doc_vectors is not None and len(doc_vectors) == len(texts):
            _log(progress, f"{name}: reusing doc cache {doc_cache}")
        elif require_cache:
            raise FileNotFoundError(
                f"missing or stale dense doc cache for {name}: {doc_cache} "
                "(run stage dense-only first, or --force-reembed if the chunk set changed)"
            )
        else:
            _log(progress, f"{name}: embedding {len(texts)} documents...")
            doc_vectors = _embed_texts_with_progress(
                provider,
                texts,
                label=f"{name}/docs",
                batch_size=batch_size,
                progress=progress,
            )
            _save_vector_cache(
                doc_cache,
                {
                    "model": name,
                    "count": len(doc_vectors),
                    "fingerprint": fingerprint,
                    "vectors": doc_vectors,
                },
            )
            _log(progress, f"{name}: wrote doc cache {doc_cache}")

        qmap = None if force_reembed else _load_query_cache(query_cache, fingerprint=fingerprint)
        if qmap is not None and all(example.query in qmap for example in dataset):
            _log(progress, f"{name}: reusing query cache {query_cache}")
        elif require_cache:
            raise FileNotFoundError(
                f"missing or stale dense query cache for {name}: {query_cache} "
                "(run stage dense-only first, or --force-reembed if the golden set changed)"
            )
        else:
            _log(progress, f"{name}: embedding {len(dataset)} golden queries...")
            qmap = _embed_queries_with_progress(
                provider, dataset, label=f"{name}/queries", progress=progress
            )
            _save_vector_cache(
                query_cache, {"model": name, "fingerprint": fingerprint, "queries": qmap}
            )
            _log(progress, f"{name}: wrote query cache {query_cache}")

        dense_by_model[name] = doc_vectors
        query_maps[name] = qmap
    return dense_by_model, query_maps


def _rank_chunk_ids_by_dense(
    query_vector: Sequence[float],
    chunk_vectors: Sequence[Sequence[float]],
    *,
    limit: int,
) -> list[RetrievedChunk]:
    scored = [(index, _cosine(query_vector, vec)) for index, vec in enumerate(chunk_vectors)]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [
        RetrievedChunk(chunk_id=index, score=score, payload={}) for index, score in scored[:limit]
    ]


def _rank_chunk_ids_by_sparse(
    query: SparseVector | SparseEmbedding,
    doc_sparse: Sequence[SparseVector | SparseEmbedding],
    *,
    limit: int,
) -> list[RetrievedChunk]:
    """Rank by sparse dot product. Zero lexical overlap means no signal, not "everything ties at
    index order" -- returning the latter would feed RRF fusion index-position noise disguised as
    real sparse hits, for exactly the paraphrase/morphological-variant queries where zero overlap
    is expected and common."""
    scored = [(index, _sparse_dot(query, doc)) for index, doc in enumerate(doc_sparse)]
    positive = [(index, score) for index, score in scored if score > 0.0]
    if not positive:
        return []
    positive.sort(key=lambda item: (-item[1], item[0]))
    return [
        RetrievedChunk(chunk_id=index, score=score, payload={}) for index, score in positive[:limit]
    ]


def _books_from_fused(
    fused_chunk_ids: Sequence[int],
    book_ids: Sequence[int],
    *,
    limit: int,
) -> list[int]:
    ranked: list[int] = []
    seen: set[int] = set()
    for chunk_index in fused_chunk_ids:
        book = book_ids[chunk_index]
        if book in seen:
            continue
        seen.add(book)
        ranked.append(book)
        if len(ranked) >= limit:
            break
    return ranked


def run_dense_only_comparison(
    *,
    models: dict[str, EmbeddingProvider],
    golden_path: Path,
    chunks_path: Path,
    output_dir: Path,
    max_chunks: int = 2048,
    candidate_limit: int = 100,
    batch_size: int = 16,
    force_reembed: bool = False,
    progress: ProgressFn | None = None,
) -> ComparisonReport:
    """Embed shared chunks + golden queries per model; score dense-only retrieval."""
    if not models:
        raise ValueError("models must be non-empty")
    if max_chunks <= 0:
        raise ValueError(f"max_chunks must be positive, got {max_chunks}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = list(load_golden_dataset(golden_path))
    _log(progress, f"loaded {len(dataset)} golden queries from {golden_path}")

    _log(
        progress,
        f"loading chunks from {chunks_path} (round-robin subsample to {max_chunks})...",
    )
    texts, book_ids = _load_chunk_texts(chunks_path, max_chunks=max_chunks)
    unique_books = len(set(book_ids))
    _log(
        progress,
        f"using {len(texts)} chunks across {unique_books} books (max_chunks={max_chunks})",
    )

    dense_by_model, query_maps = _prepare_dense_caches(
        models=models,
        dataset=dataset,
        texts=texts,
        output_dir=output_dir,
        batch_size=batch_size,
        force_reembed=force_reembed,
        progress=progress,
    )

    _log(progress, "scoring dense-only retrieval (Recall@k / MRR / nDCG)...")
    report = run_quant_retrieval(
        dataset,
        texts,
        book_ids,
        dense_by_variant=dense_by_model,
        query_vectors_by_variant=query_maps,
        candidate_limit=candidate_limit,
        ks=(10, 100),
    )
    table = _write_stage_artifacts(
        output_dir,
        stage="dense-only",
        golden_path=golden_path,
        chunks_path=chunks_path,
        max_chunks=max_chunks,
        chunk_count=len(texts),
        book_coverage=unique_books,
        candidate_limit=candidate_limit,
        models=list(models),
        report=report,
    )
    _log(progress, f"wrote {output_dir / 'dense_only_table.md'}")
    _log(progress, "\n" + table.rstrip("\n"))
    return report


def run_hybrid_bm25_comparison(
    *,
    models: dict[str, EmbeddingProvider],
    golden_path: Path,
    chunks_path: Path,
    output_dir: Path,
    max_chunks: int = 2048,
    candidate_limit: int = 100,
    fusion_pool: int = 200,
    rrf_k: int = 60,
    batch_size: int = 16,
    force_reembed: bool = False,
    require_dense_cache: bool = True,
    progress: ProgressFn | None = None,
) -> ComparisonReport:
    """Stage 2: each dense model + the same surface BM25, fused with RRF (reranker off)."""
    if not models:
        raise ValueError("models must be non-empty")
    if max_chunks <= 0:
        raise ValueError(f"max_chunks must be positive, got {max_chunks}")
    if fusion_pool <= 0:
        raise ValueError(f"fusion_pool must be positive, got {fusion_pool}")
    if rrf_k <= 0:
        raise ValueError(f"rrf_k must be positive, got {rrf_k}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = list(load_golden_dataset(golden_path))
    _log(progress, f"loaded {len(dataset)} golden queries from {golden_path}")

    _log(
        progress,
        f"loading chunks from {chunks_path} (round-robin subsample to {max_chunks})...",
    )
    texts, book_ids = _load_chunk_texts(chunks_path, max_chunks=max_chunks)
    unique_books = len(set(book_ids))
    _log(
        progress,
        f"using {len(texts)} chunks across {unique_books} books",
    )

    dense_by_model, query_maps = _prepare_dense_caches(
        models=models,
        dataset=dataset,
        texts=texts,
        output_dir=output_dir,
        batch_size=batch_size,
        force_reembed=force_reembed,
        progress=progress,
        require_cache=require_dense_cache and not force_reembed,
    )

    _log(progress, "fitting shared surface BM25 on eval chunks...")
    t0 = time.perf_counter()
    bm25 = Bm25Encoder().fit(texts)
    doc_sparse = [bm25.encode_document(text) for text in texts]
    _log(
        progress,
        f"BM25 ready in {time.perf_counter() - t0:.1f}s "
        f"(vocab={bm25.vocabulary_size}, docs={len(doc_sparse)})",
    )
    bm25_path = output_dir / "hybrid_bm25_state.json"
    bm25.save(bm25_path)
    _log(progress, f"wrote {bm25_path}")

    configs: list[RunConfig] = []
    for name, doc_vectors in dense_by_model.items():
        qmap = query_maps[name]

        def _retrieve(
            query: str,
            *,
            _docs: Sequence[Sequence[float]] = doc_vectors,
            _books: Sequence[int] = book_ids,
            _qmap: Mapping[str, Sequence[float]] = qmap,
            _doc_sparse: Sequence[SparseVector] = doc_sparse,
            _bm25: Bm25Encoder = bm25,
        ) -> list[int]:
            dense_list = _rank_chunk_ids_by_dense(_qmap[query], _docs, limit=fusion_pool)
            sparse_list = _rank_chunk_ids_by_sparse(
                _bm25.encode_query(query), _doc_sparse, limit=fusion_pool
            )
            fused = reciprocal_rank_fusion([dense_list, sparse_list], k=rrf_k, limit=fusion_pool)
            return _books_from_fused([hit.chunk_id for hit in fused], _books, limit=candidate_limit)

        configs.append(RunConfig(f"{name}+bm25", _retrieve))

    _log(progress, "scoring hybrid dense+BM25 retrieval (Recall@k / MRR / nDCG)...")
    report = run_comparison(configs, dataset, ks=(10, 100))
    table = _write_stage_artifacts(
        output_dir,
        stage="hybrid-bm25",
        golden_path=golden_path,
        chunks_path=chunks_path,
        max_chunks=max_chunks,
        chunk_count=len(texts),
        book_coverage=unique_books,
        candidate_limit=candidate_limit,
        models=[c.name for c in configs],
        report=report,
        extra={"fusion_pool": fusion_pool, "rrf_k": rrf_k, "shared_bm25": True},
    )
    _log(progress, f"wrote {output_dir / 'hybrid_bm25_table.md'}")
    _log(progress, "\n" + table.rstrip("\n"))
    return report


def _sparse_doc_cache_path(output_dir: Path, model_name: str) -> Path:
    safe = model_name.replace("/", "_")
    return output_dir / f"{safe}_sparse_docs.json"


def _sparse_query_cache_path(output_dir: Path, model_name: str) -> Path:
    safe = model_name.replace("/", "_")
    return output_dir / f"{safe}_sparse_queries.json"


def _load_sparse_doc_cache(path: Path, *, fingerprint: str) -> list[SparseVector] | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("fingerprint") != fingerprint:
        return None
    vectors = raw.get("vectors")
    if not isinstance(vectors, list):
        return None
    return [_sparse_from_dict(entry) for entry in vectors]


def _load_sparse_query_cache(path: Path, *, fingerprint: str) -> dict[str, SparseVector] | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("fingerprint") != fingerprint:
        return None
    mapping = raw.get("queries")
    if not isinstance(mapping, dict):
        return None
    return {str(k): _sparse_from_dict(v) for k, v in mapping.items()}


def _embed_bge_sparse_docs_with_progress(
    provider: BgeM3EmbeddingProvider,
    texts: Sequence[str],
    *,
    label: str,
    batch_size: int,
    progress: ProgressFn | None,
) -> list[SparseVector]:
    vectors: list[SparseVector] = []
    total = len(texts)
    report_every = max(1, total // 20) if total >= 20 else 1
    t0 = time.perf_counter()
    for start in range(0, total, batch_size):
        batch = list(texts[start : start + batch_size])
        for sparse in provider.embed_documents_sparse(batch):
            vectors.append(SparseVector(indices=list(sparse.indices), values=list(sparse.values)))
        done = min(start + batch_size, total)
        if done == total or done <= batch_size or done % report_every < batch_size:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            _log(
                progress,
                f"{label}: encoded sparse {done}/{total} texts "
                f"({elapsed:.1f}s, {rate:.1f} texts/s)",
            )
    return vectors


def _embed_bge_sparse_queries_with_progress(
    provider: BgeM3EmbeddingProvider,
    dataset: Sequence[GoldenExample],
    *,
    label: str,
    progress: ProgressFn | None,
) -> dict[str, SparseVector]:
    out: dict[str, SparseVector] = {}
    total = len(dataset)
    t0 = time.perf_counter()
    for index, example in enumerate(dataset, start=1):
        sparse = provider.embed_query_sparse(example.query)
        out[example.query] = SparseVector(indices=list(sparse.indices), values=list(sparse.values))
        if index == total or index == 1 or index % max(1, total // 10) == 0:
            _log(
                progress,
                f"{label}: encoded sparse queries {index}/{total} "
                f"({time.perf_counter() - t0:.1f}s)",
            )
    return out


def _prepare_bge_sparse_caches(
    *,
    provider: BgeM3EmbeddingProvider,
    model_name: str,
    dataset: Sequence[GoldenExample],
    texts: Sequence[str],
    output_dir: Path,
    batch_size: int,
    force_reembed: bool,
    progress: ProgressFn | None,
) -> tuple[list[SparseVector], dict[str, SparseVector]]:
    doc_cache = _sparse_doc_cache_path(output_dir, model_name)
    query_cache = _sparse_query_cache_path(output_dir, model_name)
    fingerprint = _texts_fingerprint(texts)

    doc_sparse = (
        None if force_reembed else _load_sparse_doc_cache(doc_cache, fingerprint=fingerprint)
    )
    if doc_sparse is not None and len(doc_sparse) == len(texts):
        _log(progress, f"{model_name}: reusing learned-sparse doc cache {doc_cache}")
    else:
        _log(progress, f"{model_name}: encoding {len(texts)} documents (learned-sparse)...")
        doc_sparse = _embed_bge_sparse_docs_with_progress(
            provider,
            texts,
            label=f"{model_name}/sparse-docs",
            batch_size=batch_size,
            progress=progress,
        )
        _save_vector_cache(
            doc_cache,
            {
                "model": model_name,
                "count": len(doc_sparse),
                "fingerprint": fingerprint,
                "vectors": [_sparse_to_dict(v) for v in doc_sparse],
            },
        )
        _log(progress, f"{model_name}: wrote learned-sparse doc cache {doc_cache}")

    qmap = None if force_reembed else _load_sparse_query_cache(query_cache, fingerprint=fingerprint)
    if qmap is not None and all(example.query in qmap for example in dataset):
        _log(progress, f"{model_name}: reusing learned-sparse query cache {query_cache}")
    else:
        _log(progress, f"{model_name}: encoding {len(dataset)} queries with BGE learned-sparse...")
        qmap = _embed_bge_sparse_queries_with_progress(
            provider, dataset, label=f"{model_name}/sparse-queries", progress=progress
        )
        _save_vector_cache(
            query_cache,
            {
                "model": model_name,
                "fingerprint": fingerprint,
                "queries": {k: _sparse_to_dict(v) for k, v in qmap.items()},
            },
        )
        _log(progress, f"{model_name}: wrote learned-sparse query cache {query_cache}")

    return doc_sparse, qmap


def run_bge_sparse_ablation(
    *,
    dense_provider: EmbeddingProvider,
    sparse_provider: BgeM3EmbeddingProvider,
    dense_model_name: str = "bge-m3",
    golden_path: Path,
    chunks_path: Path,
    output_dir: Path,
    max_chunks: int = 2048,
    candidate_limit: int = 100,
    fusion_pool: int = 200,
    rrf_k: int = 60,
    dense_batch_size: int = 16,
    sparse_batch_size: int = 8,
    force_reembed: bool = False,
    require_dense_cache: bool = True,
    progress: ProgressFn | None = None,
) -> ComparisonReport:
    """Stage 3: BGE dense + BM25 vs + learned-sparse vs + both (RRF, reranker off)."""
    if max_chunks <= 0:
        raise ValueError(f"max_chunks must be positive, got {max_chunks}")
    if fusion_pool <= 0:
        raise ValueError(f"fusion_pool must be positive, got {fusion_pool}")
    if rrf_k <= 0:
        raise ValueError(f"rrf_k must be positive, got {rrf_k}")
    if not sparse_provider.sparse_enabled:
        raise ValueError("sparse_provider must have enable_sparse=True")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = list(load_golden_dataset(golden_path))
    _log(progress, f"loaded {len(dataset)} golden queries from {golden_path}")

    _log(
        progress,
        f"loading chunks from {chunks_path} (round-robin subsample to {max_chunks})...",
    )
    texts, book_ids = _load_chunk_texts(chunks_path, max_chunks=max_chunks)
    unique_books = len(set(book_ids))
    _log(
        progress,
        f"using {len(texts)} chunks across {unique_books} books",
    )

    dense_by_model, query_maps = _prepare_dense_caches(
        models={dense_model_name: dense_provider},
        dataset=dataset,
        texts=texts,
        output_dir=output_dir,
        batch_size=dense_batch_size,
        force_reembed=force_reembed,
        progress=progress,
        require_cache=require_dense_cache and not force_reembed,
    )
    doc_dense = dense_by_model[dense_model_name]
    dense_queries = query_maps[dense_model_name]

    doc_learned_sparse, learned_query_maps = _prepare_bge_sparse_caches(
        provider=sparse_provider,
        model_name=dense_model_name,
        dataset=dataset,
        texts=texts,
        output_dir=output_dir,
        batch_size=sparse_batch_size,
        force_reembed=force_reembed,
        progress=progress,
    )

    bm25_path = output_dir / "hybrid_bm25_state.json"
    if bm25_path.is_file() and not force_reembed:
        _log(progress, f"loading shared BM25 state from {bm25_path}")
        bm25 = Bm25Encoder.load(bm25_path)
        doc_bm25 = [bm25.encode_document(text) for text in texts]
    else:
        _log(progress, "fitting shared surface BM25 on eval chunks...")
        t0 = time.perf_counter()
        bm25 = Bm25Encoder().fit(texts)
        doc_bm25 = [bm25.encode_document(text) for text in texts]
        _log(
            progress,
            f"BM25 ready in {time.perf_counter() - t0:.1f}s "
            f"(vocab={bm25.vocabulary_size}, docs={len(doc_bm25)})",
        )
        bm25.save(bm25_path)
        _log(progress, f"wrote {bm25_path}")

    def _dense_rank(query: str) -> list[RetrievedChunk]:
        return _rank_chunk_ids_by_dense(dense_queries[query], doc_dense, limit=fusion_pool)

    def _bm25_rank(query: str) -> list[RetrievedChunk]:
        return _rank_chunk_ids_by_sparse(bm25.encode_query(query), doc_bm25, limit=fusion_pool)

    def _learned_rank(query: str) -> list[RetrievedChunk]:
        return _rank_chunk_ids_by_sparse(
            learned_query_maps[query], doc_learned_sparse, limit=fusion_pool
        )

    def _retrieve_bm25(query: str) -> list[int]:
        fused = reciprocal_rank_fusion(
            [_dense_rank(query), _bm25_rank(query)], k=rrf_k, limit=fusion_pool
        )
        return _books_from_fused([hit.chunk_id for hit in fused], book_ids, limit=candidate_limit)

    def _retrieve_learned(query: str) -> list[int]:
        fused = reciprocal_rank_fusion(
            [_dense_rank(query), _learned_rank(query)], k=rrf_k, limit=fusion_pool
        )
        return _books_from_fused([hit.chunk_id for hit in fused], book_ids, limit=candidate_limit)

    def _retrieve_both(query: str) -> list[int]:
        fused = reciprocal_rank_fusion(
            [_dense_rank(query), _bm25_rank(query), _learned_rank(query)],
            k=rrf_k,
            limit=fusion_pool,
        )
        return _books_from_fused([hit.chunk_id for hit in fused], book_ids, limit=candidate_limit)

    configs = [
        RunConfig(f"{dense_model_name}+bm25", _retrieve_bm25),
        RunConfig(f"{dense_model_name}+learned-sparse", _retrieve_learned),
        RunConfig(f"{dense_model_name}+both", _retrieve_both),
    ]

    _log(progress, "scoring BGE sparse ablation (Recall@k / MRR / nDCG)...")
    report = run_comparison(configs, dataset, ks=(10, 100))
    table = _write_stage_artifacts(
        output_dir,
        stage="bge-sparse",
        golden_path=golden_path,
        chunks_path=chunks_path,
        max_chunks=max_chunks,
        chunk_count=len(texts),
        book_coverage=unique_books,
        candidate_limit=candidate_limit,
        models=[c.name for c in configs],
        report=report,
        extra={
            "fusion_pool": fusion_pool,
            "rrf_k": rrf_k,
            "dense_source": dense_model_name,
        },
    )
    _log(progress, f"wrote {output_dir / 'bge_sparse_table.md'}")
    _log(progress, "\n" + table.rstrip("\n"))
    return report
