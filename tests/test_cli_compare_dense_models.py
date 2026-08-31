"""CLI tests for `compare-dense-models` (M6-03)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shamela_rag import cli
from shamela_rag.embeddings.provider import InMemoryEmbeddingProvider


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


def test_run_compare_dense_models_missing_chunks_file(tmp_path: Path) -> None:
    golden = tmp_path / "golden.jsonl"
    _write_golden(golden)
    args = cli.build_parser().parse_args(
        [
            "compare-dense-models",
            "--output-dir",
            str(tmp_path / "out"),
            "--golden",
            str(golden),
            "--chunks",
            str(tmp_path / "missing_chunks.jsonl"),
        ]
    )
    assert cli.run_compare_dense_models(args) == 1


def test_run_compare_dense_models_missing_golden_file(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    _write_chunks(chunks)
    args = cli.build_parser().parse_args(
        [
            "compare-dense-models",
            "--output-dir",
            str(tmp_path / "out"),
            "--golden",
            str(tmp_path / "missing_golden.jsonl"),
            "--chunks",
            str(chunks),
        ]
    )
    assert cli.run_compare_dense_models(args) == 1


def test_run_compare_dense_models_dense_only_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    golden = tmp_path / "golden.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    _write_golden(golden)
    _write_chunks(chunks)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        "shamela_rag.factory.build_embedder", lambda model: InMemoryEmbeddingProvider(dims=8)
    )

    args = cli.build_parser().parse_args(
        [
            "compare-dense-models",
            "--output-dir",
            str(output_dir),
            "--golden",
            str(golden),
            "--chunks",
            str(chunks),
            "--max-chunks",
            "4",
            "--stage",
            "dense-only",
        ]
    )
    assert cli.run_compare_dense_models(args) == 0
    assert (output_dir / "dense_only_table.md").is_file()


def test_run_compare_dense_models_bge_sparse_stage_skips_qwen3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--stage bge-sparse must not require a working qwen3 backend at all."""
    golden = tmp_path / "golden.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    _write_golden(golden)
    _write_chunks(chunks)

    requested: list[str] = []

    def _fake_build_embedder(model: str) -> InMemoryEmbeddingProvider:
        requested.append(model)
        if model == "qwen3":
            raise RuntimeError("qwen3 should not be requested for bge-sparse stage")
        return InMemoryEmbeddingProvider(dims=8)

    monkeypatch.setattr("shamela_rag.factory.build_embedder", _fake_build_embedder)
    monkeypatch.setattr(
        "shamela_rag.embeddings.bge_m3.BgeM3EmbeddingProvider",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "shamela_rag.eval.model_ab.run_bge_sparse_ablation",
        lambda **_kwargs: __import__(
            "shamela_rag.eval.comparison", fromlist=["ComparisonReport"]
        ).ComparisonReport(ks=(10, 100), results={}),
    )

    args = cli.build_parser().parse_args(
        [
            "compare-dense-models",
            "--output-dir",
            str(tmp_path / "out"),
            "--golden",
            str(golden),
            "--chunks",
            str(chunks),
            "--max-chunks",
            "4",
            "--stage",
            "bge-sparse",
        ]
    )
    assert cli.run_compare_dense_models(args) == 0
    assert requested == ["bge-m3"]


def test_run_compare_dense_models_bge_sparse_load_failure_returns_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    golden = tmp_path / "golden.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    _write_golden(golden)
    _write_chunks(chunks)

    monkeypatch.setattr(
        "shamela_rag.factory.build_embedder", lambda model: InMemoryEmbeddingProvider(dims=8)
    )

    def _boom(**_kwargs: object) -> object:
        raise ImportError("requires shamela-rag[bge]")

    monkeypatch.setattr("shamela_rag.embeddings.bge_m3.BgeM3EmbeddingProvider", _boom)

    args = cli.build_parser().parse_args(
        [
            "compare-dense-models",
            "--output-dir",
            str(tmp_path / "out"),
            "--golden",
            str(golden),
            "--chunks",
            str(chunks),
            "--stage",
            "bge-sparse",
        ]
    )
    assert cli.run_compare_dense_models(args) == 1


def test_run_compare_dense_models_embedder_build_failure_returns_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    golden = tmp_path / "golden.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    _write_golden(golden)
    _write_chunks(chunks)

    def _boom(model: str) -> InMemoryEmbeddingProvider:
        raise RuntimeError("no backend configured")

    monkeypatch.setattr("shamela_rag.factory.build_embedder", _boom)

    args = cli.build_parser().parse_args(
        [
            "compare-dense-models",
            "--output-dir",
            str(tmp_path / "out"),
            "--golden",
            str(golden),
            "--chunks",
            str(chunks),
        ]
    )
    assert cli.run_compare_dense_models(args) == 1


def test_main_dispatches_compare_dense_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "run_compare_dense_models", lambda _args: 0)
    assert (
        cli.main(
            [
                "compare-dense-models",
                "--output-dir",
                str(tmp_path),
                "--chunks",
                str(tmp_path / "c.jsonl"),
            ]
        )
        == 0
    )
