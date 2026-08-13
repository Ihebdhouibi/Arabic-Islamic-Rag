"""Tests for the Streamlit demo wiring (M7-04)."""

from __future__ import annotations

from pathlib import Path

import pytest

from shamela_rag.config import Settings, get_settings
from shamela_rag.demo.wiring import build_general_qa_service
from shamela_rag.embeddings.bm25 import Bm25Encoder
from shamela_rag.embeddings.provider import InMemoryEmbeddingProvider
from shamela_rag.generation.provider import InMemoryGenerationProvider
from shamela_rag.generation.service import GeneralQAService
from shamela_rag.retrieval.rerank import LexicalOverlapReranker


def test_build_general_qa_service_requires_bm25_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing_bm25.json"
    base = get_settings()
    monkeypatch.setattr(
        "shamela_rag.demo.wiring.get_settings",
        lambda: Settings(
            **{**base.model_dump(), "bm25_state_path": missing},
        ),
    )

    with pytest.raises(FileNotFoundError, match="BM25 state not found"):
        build_general_qa_service(generation_provider=InMemoryGenerationProvider())


def test_build_general_qa_service_with_injected_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "bm25.json"
    Bm25Encoder().fit(["الشافعي العلم", "مالك"]).save(state_path)
    base = get_settings()
    monkeypatch.setattr(
        "shamela_rag.demo.wiring.get_settings",
        lambda: Settings(
            **{**base.model_dump(), "bm25_state_path": state_path},
        ),
    )

    qa = build_general_qa_service(
        generation_provider=InMemoryGenerationProvider(),
        reranker=LexicalOverlapReranker(),
        embedder=InMemoryEmbeddingProvider(dims=8),
    )
    assert isinstance(qa, GeneralQAService)
