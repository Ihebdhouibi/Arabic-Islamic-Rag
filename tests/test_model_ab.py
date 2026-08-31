"""Tests for eval/model_ab.py (M6-03 dense-model comparison), including regression tests for
two bugs found in code review before merge: stale count-only cache validation, and a sparse
zero-overlap fallback that fabricated index-ordered "hits" instead of returning no signal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from shamela_rag.embeddings.bge_m3 import SparseEmbedding
from shamela_rag.embeddings.bm25 import SparseVector
from shamela_rag.embeddings.provider import InMemoryEmbeddingProvider
from shamela_rag.eval.dataset import GoldenExample, GoldenSource
from shamela_rag.eval.model_ab import (
    _books_from_fused,
    _load_chunk_texts,
    _prepare_bge_sparse_caches,
    _prepare_dense_caches,
    _rank_chunk_ids_by_dense,
    _rank_chunk_ids_by_sparse,
    _texts_fingerprint,
    run_bge_sparse_ablation,
    run_dense_only_comparison,
    run_hybrid_bm25_comparison,
)


class _FakeSparseProvider:
    """Duck-typed stand-in for BgeM3EmbeddingProvider's learned-sparse interface, so stage-3
    (run_bge_sparse_ablation) is testable without the real (heavy) BGE-M3 weights."""

    sparse_enabled = True

    def embed_documents_sparse(self, texts: list[str]) -> list[SparseEmbedding]:
        return [SparseEmbedding(indices=[len(t) % 7], values=[1.0]) for t in texts]

    def embed_query_sparse(self, text: str) -> SparseEmbedding:
        return SparseEmbedding(indices=[len(text) % 7], values=[1.0])


# --------------------------------------------------------------------------- fingerprint


def test_fingerprint_deterministic_and_content_sensitive() -> None:
    assert _texts_fingerprint(["a", "b"]) == _texts_fingerprint(["a", "b"])
    assert _texts_fingerprint(["a", "b"]) != _texts_fingerprint(["c", "d"])
    # Same total length, different content -> must not collide.
    assert _texts_fingerprint(["ab"]) != _texts_fingerprint(["a", "b"])


# --------------------------------------------------------------------------- sparse ranking


def test_rank_chunk_ids_by_sparse_zero_overlap_returns_empty() -> None:
    """Regression: previously fell back to index-ordered fake hits on zero overlap."""
    query = SparseVector(indices=[1, 2], values=[1.0, 1.0])
    docs = [
        SparseVector(indices=[3, 4], values=[1.0, 1.0]),
        SparseVector(indices=[5, 6], values=[1.0, 1.0]),
    ]
    assert _rank_chunk_ids_by_sparse(query, docs, limit=10) == []


def test_rank_chunk_ids_by_sparse_ranks_by_score_with_index_tiebreak() -> None:
    query = SparseVector(indices=[1, 2], values=[1.0, 1.0])
    docs = [
        SparseVector(indices=[1], values=[0.5]),  # dot = 0.5
        SparseVector(indices=[1, 2], values=[1.0, 1.0]),  # dot = 2.0
        SparseVector(indices=[9], values=[1.0]),  # dot = 0.0 -> excluded
    ]
    result = _rank_chunk_ids_by_sparse(query, docs, limit=10)
    assert [hit.chunk_id for hit in result] == [1, 0]


def test_rank_chunk_ids_by_sparse_respects_limit() -> None:
    query = SparseVector(indices=[1], values=[1.0])
    docs = [SparseVector(indices=[1], values=[1.0]) for _ in range(5)]
    result = _rank_chunk_ids_by_sparse(query, docs, limit=2)
    assert len(result) == 2


def test_rank_chunk_ids_by_sparse_accepts_bge_sparse_embedding() -> None:
    """SparseEmbedding (BGE-M3's type) must work interchangeably with SparseVector."""
    query = SparseEmbedding(indices=[1], values=[1.0])
    docs = [SparseEmbedding(indices=[1], values=[1.0])]
    result = _rank_chunk_ids_by_sparse(query, docs, limit=10)
    assert [hit.chunk_id for hit in result] == [0]


# --------------------------------------------------------------------------- dense ranking


def test_rank_chunk_ids_by_dense_orders_by_cosine() -> None:
    result = _rank_chunk_ids_by_dense([1.0, 0.0], [[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]], limit=10)
    assert [hit.chunk_id for hit in result] == [1, 2, 0]


# --------------------------------------------------------------------------- book dedup


def test_books_from_fused_dedupes_and_respects_limit() -> None:
    # chunk indices 0,1 -> book 5; chunk index 2 -> book 6; chunk index 3 -> book 7
    book_ids = [5, 5, 6, 7]
    result = _books_from_fused([0, 1, 2, 3], book_ids, limit=2)
    assert result == [5, 6]


# --------------------------------------------------------------------------- chunk loading


def test_load_chunk_texts_round_robin_subsample(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    rows = [{"book_id": 1, "text": f"b1-{i}"} for i in range(3)] + [
        {"book_id": 2, "text": f"b2-{i}"} for i in range(3)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    texts, books = _load_chunk_texts(path, max_chunks=4)
    assert len(texts) == 4
    # Round-robin: one from each book alternately, in book order.
    assert books == [1, 2, 1, 2]
    assert texts == ["b1-0", "b2-0", "b1-1", "b2-1"]


def test_load_chunk_texts_returns_all_when_under_cap(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    rows = [{"book_id": 1, "text": "only"}]
    path.write_text(json.dumps(rows[0]), encoding="utf-8")
    texts, books = _load_chunk_texts(path, max_chunks=10)
    assert texts == ["only"]
    assert books == [1]


def test_load_chunk_texts_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no chunks"):
        _load_chunk_texts(path, max_chunks=10)


# --------------------------------------------------------------------------- dense cache staleness


def _dataset(query: str = "q") -> list[GoldenExample]:
    return [
        GoldenExample(
            example_id="1",
            query=query,
            sources=(
                GoldenSource(book_id=1, shamela_page_id=1, confidence="verified", book_title="t"),
            ),
        )
    ]


def test_prepare_dense_caches_reuses_matching_cache(tmp_path: Path) -> None:
    provider = InMemoryEmbeddingProvider(dims=8)
    texts = ["alpha", "beta"]
    dataset = _dataset()

    first, _ = _prepare_dense_caches(
        models={"m": provider},
        dataset=dataset,
        texts=texts,
        output_dir=tmp_path,
        batch_size=8,
        force_reembed=False,
        progress=None,
    )
    # Corrupt the cache file's vectors so we can tell whether the second call reused it verbatim
    # (bit-for-bit) rather than recomputing -- recomputing would restore the correct values.
    cache_path = tmp_path / "m_docs.json"
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    raw["vectors"] = [[999.0] * 8, [999.0] * 8]
    cache_path.write_text(json.dumps(raw), encoding="utf-8")

    second, _ = _prepare_dense_caches(
        models={"m": provider},
        dataset=dataset,
        texts=texts,
        output_dir=tmp_path,
        batch_size=8,
        force_reembed=False,
        progress=None,
    )
    assert second["m"] == [[999.0] * 8, [999.0] * 8]
    assert first["m"] != second["m"]


def test_prepare_dense_caches_rejects_stale_same_length_cache(tmp_path: Path) -> None:
    """Regression: a cache keyed on len(texts) alone would wrongly be reused here, since both
    text sets have the same chunk count but different content."""
    provider = InMemoryEmbeddingProvider(dims=8)
    dataset = _dataset()
    texts_v1 = ["alpha text one", "beta text two"]

    _prepare_dense_caches(
        models={"m": provider},
        dataset=dataset,
        texts=texts_v1,
        output_dir=tmp_path,
        batch_size=8,
        force_reembed=False,
        progress=None,
    )

    texts_v2 = ["gamma text three", "delta text four"]  # same length, different content
    dense_v2, _ = _prepare_dense_caches(
        models={"m": provider},
        dataset=dataset,
        texts=texts_v2,
        output_dir=tmp_path,
        batch_size=8,
        force_reembed=False,
        progress=None,
    )

    assert dense_v2["m"] == provider.embed_documents(texts_v2)


def test_prepare_dense_caches_require_cache_raises_when_missing(tmp_path: Path) -> None:
    provider = InMemoryEmbeddingProvider(dims=8)
    with pytest.raises(FileNotFoundError, match="dense doc cache"):
        _prepare_dense_caches(
            models={"m": provider},
            dataset=_dataset(),
            texts=["a", "b"],
            output_dir=tmp_path,
            batch_size=8,
            force_reembed=False,
            progress=None,
            require_cache=True,
        )


def test_prepare_bge_sparse_caches_rejects_stale_same_length_cache(tmp_path: Path) -> None:
    """Same staleness bug, for the learned-sparse cache path."""
    provider = _FakeSparseProvider()
    dataset = _dataset()
    texts_v1 = ["aa", "bbb"]
    _prepare_bge_sparse_caches(
        provider=provider,  # type: ignore[arg-type]
        model_name="bge-m3",
        dataset=dataset,
        texts=texts_v1,
        output_dir=tmp_path,
        batch_size=8,
        force_reembed=False,
        progress=None,
    )

    texts_v2 = ["cccc", "d"]  # same length (2), different content/lengths
    doc_sparse_v2, _ = _prepare_bge_sparse_caches(
        provider=provider,  # type: ignore[arg-type]
        model_name="bge-m3",
        dataset=dataset,
        texts=texts_v2,
        output_dir=tmp_path,
        batch_size=8,
        force_reembed=False,
        progress=None,
    )
    assert [v.indices for v in doc_sparse_v2] == [[4], [1]]


# --------------------------------------------------------------------------- end-to-end stages


def _write_golden(path: Path) -> None:
    row = {
        "id": "1",
        "query": "q",
        "expected_sources": [
            {
                "internal_book_id": 1,
                "shamela_page_id": 1,
                "confidence": "verified",
                "book_title": "t",
            }
        ],
    }
    path.write_text(json.dumps(row), encoding="utf-8")


def _write_chunks(path: Path) -> None:
    rows = [
        {"book_id": 1, "text": "first book chunk one"},
        {"book_id": 1, "text": "first book chunk two"},
        {"book_id": 2, "text": "second book chunk one"},
        {"book_id": 2, "text": "second book chunk two"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_run_dense_only_comparison_writes_artifacts(tmp_path: Path) -> None:
    golden = tmp_path / "golden.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    _write_golden(golden)
    _write_chunks(chunks)
    output_dir = tmp_path / "out"

    report = run_dense_only_comparison(
        models={
            "bge-m3": InMemoryEmbeddingProvider(dims=8),
            "qwen3": InMemoryEmbeddingProvider(dims=8),
        },
        golden_path=golden,
        chunks_path=chunks,
        output_dir=output_dir,
        max_chunks=4,
        candidate_limit=10,
        batch_size=8,
    )
    assert set(report.results) == {"bge-m3", "qwen3"}
    assert (output_dir / "dense_only_table.md").is_file()
    metrics = json.loads((output_dir / "dense_only_metrics.json").read_text(encoding="utf-8"))
    assert metrics["stage"] == "dense-only"
    assert metrics["chunk_count"] == 4
    assert metrics["book_coverage"] == 2


def test_run_hybrid_bm25_comparison_reuses_dense_cache_from_prior_stage(tmp_path: Path) -> None:
    golden = tmp_path / "golden.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    _write_golden(golden)
    _write_chunks(chunks)
    output_dir = tmp_path / "out"
    models: dict[str, Any] = {"bge-m3": InMemoryEmbeddingProvider(dims=8)}

    run_dense_only_comparison(
        models=models,
        golden_path=golden,
        chunks_path=chunks,
        output_dir=output_dir,
        max_chunks=4,
        candidate_limit=10,
        batch_size=8,
    )
    report = run_hybrid_bm25_comparison(
        models=models,
        golden_path=golden,
        chunks_path=chunks,
        output_dir=output_dir,
        max_chunks=4,
        candidate_limit=10,
        batch_size=8,
        require_dense_cache=True,
    )
    assert "bge-m3+bm25" in report.results
    assert (output_dir / "hybrid_bm25_table.md").is_file()
    assert (output_dir / "hybrid_bm25_state.json").is_file()


def test_run_hybrid_bm25_comparison_requires_dense_cache_when_asked(tmp_path: Path) -> None:
    golden = tmp_path / "golden.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    _write_golden(golden)
    _write_chunks(chunks)
    with pytest.raises(FileNotFoundError):
        run_hybrid_bm25_comparison(
            models={"bge-m3": InMemoryEmbeddingProvider(dims=8)},
            golden_path=golden,
            chunks_path=chunks,
            output_dir=tmp_path / "out2",
            max_chunks=4,
            require_dense_cache=True,
        )


def test_run_bge_sparse_ablation_writes_artifacts(tmp_path: Path) -> None:
    golden = tmp_path / "golden.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    _write_golden(golden)
    _write_chunks(chunks)
    output_dir = tmp_path / "out"
    dense = InMemoryEmbeddingProvider(dims=8)

    # Stage 3 expects the dense cache from a prior dense-only run when require_dense_cache=True
    # (the default) -- populate it first, same as the CLI's real "both" / sequential-stage flow.
    run_dense_only_comparison(
        models={"bge-m3": dense},
        golden_path=golden,
        chunks_path=chunks,
        output_dir=output_dir,
        max_chunks=4,
        batch_size=8,
    )

    report = run_bge_sparse_ablation(
        dense_provider=dense,
        sparse_provider=_FakeSparseProvider(),  # type: ignore[arg-type]
        golden_path=golden,
        chunks_path=chunks,
        output_dir=output_dir,
        max_chunks=4,
        candidate_limit=10,
        dense_batch_size=8,
        sparse_batch_size=8,
    )
    assert set(report.results) == {"bge-m3+bm25", "bge-m3+learned-sparse", "bge-m3+both"}
    assert (output_dir / "bge_sparse_table.md").is_file()
    metrics = json.loads((output_dir / "bge_sparse_metrics.json").read_text(encoding="utf-8"))
    assert metrics["dense_source"] == "bge-m3"


def test_run_bge_sparse_ablation_rejects_provider_without_sparse_enabled(tmp_path: Path) -> None:
    class _NoSparse:
        sparse_enabled = False

    golden = tmp_path / "golden.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    _write_golden(golden)
    _write_chunks(chunks)
    with pytest.raises(ValueError, match="enable_sparse"):
        run_bge_sparse_ablation(
            dense_provider=InMemoryEmbeddingProvider(dims=8),
            sparse_provider=_NoSparse(),  # type: ignore[arg-type]
            golden_path=golden,
            chunks_path=chunks,
            output_dir=tmp_path / "out3",
            max_chunks=4,
            require_dense_cache=False,
        )
