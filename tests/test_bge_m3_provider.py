from __future__ import annotations

import os
from typing import Any

import pytest

from shamela_rag.embeddings import bge_m3 as bge_mod
from shamela_rag.embeddings.bge_m3 import (
    BGE_M3_DIMS,
    BGE_M3_MODEL_ID,
    BgeM3EmbeddingProvider,
    SparseEmbedding,
    lexical_weights_to_sparse,
)


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))


class _FakeModel:
    tokenizer = _FakeTokenizer()


def test_bge_constants() -> None:
    assert BGE_M3_MODEL_ID == "BAAI/bge-m3"
    assert BGE_M3_DIMS == 1024


def test_lexical_weights_to_sparse_converts_and_drops_zeros() -> None:
    sparse = lexical_weights_to_sparse({"5": 0.5, "9": 0.0, "12": 0.25})
    assert isinstance(sparse, SparseEmbedding)
    assert sparse.indices == [5, 12]
    assert sparse.values == [0.5, 0.25]


def test_bge_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        BgeM3EmbeddingProvider(batch_size=0)


def test_bge_rejects_non_positive_max_length() -> None:
    with pytest.raises(ValueError, match="max_length"):
        BgeM3EmbeddingProvider(max_length=0)


def test_bge_import_error_mentions_optional_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> object:
        raise ImportError(
            'BgeM3EmbeddingProvider requires optional deps: pip install "shamela-rag[bge]"'
        )

    monkeypatch.setattr(bge_mod, "_load_bge_m3", _boom)
    with pytest.raises(ImportError, match=r"shamela-rag\[bge\]"):
        BgeM3EmbeddingProvider()


def test_bge_sparse_disabled_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_loader(*_args: object, **_kwargs: object) -> Any:
        return _FakeModel()

    monkeypatch.setattr(bge_mod, "_load_bge_m3", _fake_loader)
    provider = BgeM3EmbeddingProvider(enable_sparse=False)
    assert provider.dims == BGE_M3_DIMS
    assert provider.query_instruction is None
    assert provider.sparse_enabled is False
    with pytest.raises(RuntimeError, match="sparse output is disabled"):
        provider.embed_query_sparse("نص")
    with pytest.raises(RuntimeError, match="sparse output is disabled"):
        provider.embed_documents_sparse(["نص"])


def test_bge_embeds_dense_and_sparse_when_weights_available() -> None:
    """Integration: skipped unless SHAMELA_RUN_BGE_INTEGRATION=1 and deps/weights exist."""
    if os.environ.get("SHAMELA_RUN_BGE_INTEGRATION") != "1":
        pytest.skip("set SHAMELA_RUN_BGE_INTEGRATION=1 to run BGE-M3 weight test")
    pytest.importorskip("FlagEmbedding")

    try:
        provider = BgeM3EmbeddingProvider(device="cpu", batch_size=2)
    except Exception as exc:  # noqa: BLE001 - absent weights must skip, not fail CI
        pytest.skip(f"BGE-M3 weights unavailable: {exc}")

    assert provider.dims == BGE_M3_DIMS

    vectors = provider.embed_documents(["باب الهمزة", "الصلاة"])
    assert len(vectors) == 2
    assert all(len(v) == provider.dims for v in vectors)

    query_vector = provider.embed_query("ما هي الصلاة")
    assert len(query_vector) == provider.dims

    sparse = provider.embed_query_sparse("ما هي الصلاة")
    assert len(sparse.indices) == len(sparse.values)
    assert len(sparse.indices) > 0
    assert provider.tokenizer.count("hello world") > 0
