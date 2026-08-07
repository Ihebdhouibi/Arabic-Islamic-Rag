from __future__ import annotations

import os

import pytest

from shamela_rag.embeddings import qwen as qwen_mod
from shamela_rag.embeddings.qwen import (
    DEFAULT_EMBEDDING_DIMS,
    DEFAULT_TASK_DESCRIPTION,
    QWEN3_EMBEDDING_MODEL_ID,
    Qwen3EmbeddingProvider,
    format_qwen_query,
)


def test_format_qwen_query_matches_official_template() -> None:
    assert (
        format_qwen_query("Given a web search query, retrieve relevant passages", "ما هي الصلاة")
        == "Instruct: Given a web search query, retrieve relevant passages\nQuery:ما هي الصلاة"
    )


def test_qwen_constants() -> None:
    assert QWEN3_EMBEDDING_MODEL_ID == "Qwen/Qwen3-Embedding-8B"
    assert DEFAULT_EMBEDDING_DIMS == 4096
    assert "retrieve relevant passages" in DEFAULT_TASK_DESCRIPTION


def test_qwen_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        Qwen3EmbeddingProvider(batch_size=0)


def test_qwen_import_error_mentions_optional_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> object:
        raise ImportError(
            'Qwen3EmbeddingProvider requires optional deps: pip install "shamela-rag[qwen]"'
        )

    monkeypatch.setattr(qwen_mod, "_load_sentence_transformer", _boom)
    with pytest.raises(ImportError, match=r"shamela-rag\[qwen\]"):
        Qwen3EmbeddingProvider()


def test_qwen_embeds_batch_when_weights_available() -> None:
    """Integration: skipped unless SHAMELA_RUN_QWEN_INTEGRATION=1 and deps/weights exist."""
    if os.environ.get("SHAMELA_RUN_QWEN_INTEGRATION") != "1":
        pytest.skip("set SHAMELA_RUN_QWEN_INTEGRATION=1 to run Qwen weight test")
    pytest.importorskip("sentence_transformers")

    try:
        provider = Qwen3EmbeddingProvider(device="cpu", batch_size=2)
    except Exception as exc:  # noqa: BLE001 - absent weights must skip, not fail CI
        pytest.skip(f"Qwen weights unavailable: {exc}")

    assert provider.dims == DEFAULT_EMBEDDING_DIMS
    assert provider.query_instruction == f"Instruct: {DEFAULT_TASK_DESCRIPTION}\nQuery:"

    vectors = provider.embed_documents(["باب الهمزة", "الصلاة"])
    assert len(vectors) == 2
    assert all(len(v) == provider.dims for v in vectors)

    query_vector = provider.embed_query("ما هي الصلاة")
    assert len(query_vector) == provider.dims
    assert provider.tokenizer.count("hello") > 0
