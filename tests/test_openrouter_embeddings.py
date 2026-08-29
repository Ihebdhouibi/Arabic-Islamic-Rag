"""Unit tests for OpenRouter embedding provider (no live network)."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from shamela_rag.embeddings.openrouter import (
    OPENROUTER_BGE_M3,
    OPENROUTER_QWEN3_EMBEDDING_8B,
    OpenRouterEmbeddingProvider,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], *, status: int = 200) -> None:
        self._raw = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_a: object) -> None:
        return None


def test_openrouter_embed_documents_posts_batch_and_orders_by_index() -> None:
    seen: dict[str, Any] = {}

    def opener(request: object, timeout: float = 0) -> _FakeResponse:
        seen["timeout"] = timeout
        assert hasattr(request, "full_url")
        assert request.full_url.endswith("/embeddings")  # type: ignore[attr-defined]
        headers = dict(request.header_items())  # type: ignore[attr-defined]
        assert headers["Authorization"] == "Bearer test-key"
        body = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        seen["body"] = body
        return _FakeResponse(
            {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            }
        )

    provider = OpenRouterEmbeddingProvider(
        OPENROUTER_BGE_M3,
        api_key="test-key",
        dims=2,
        batch_size=8,
        opener=opener,
    )
    vectors = provider.embed_documents(["a", "b"])
    assert seen["body"]["model"] == OPENROUTER_BGE_M3
    assert seen["body"]["input"] == ["a", "b"]
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert provider.dims == 2
    assert provider.query_instruction is None


def test_openrouter_embed_query_applies_qwen_instruction() -> None:
    seen: dict[str, Any] = {}

    def opener(request: object, timeout: float = 0) -> _FakeResponse:
        seen["body"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        return _FakeResponse({"data": [{"index": 0, "embedding": [1.0] * 4}]})

    provider = OpenRouterEmbeddingProvider(
        OPENROUTER_QWEN3_EMBEDDING_8B,
        api_key="k",
        dims=4,
        opener=opener,
    )
    vec = provider.embed_query("hello")
    assert len(vec) == 4
    assert seen["body"]["input"][0].startswith("Instruct:")
    assert "Query:hello" in seen["body"]["input"][0]
    assert provider.query_instruction is not None


def test_openrouter_rejects_empty_key_or_model() -> None:
    with pytest.raises(ValueError, match="api_key"):
        OpenRouterEmbeddingProvider(OPENROUTER_BGE_M3, api_key=" ")
    with pytest.raises(ValueError, match="model"):
        OpenRouterEmbeddingProvider("  ", api_key="k", dims=2)
    with pytest.raises(ValueError, match="dims"):
        OpenRouterEmbeddingProvider("unknown/model", api_key="k")


def test_openrouter_surfaces_http_error() -> None:
    import urllib.error

    def opener(request: object, timeout: float = 0) -> Any:
        raise urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/embeddings",
            code=401,
            msg="Unauthorized",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(json.dumps({"error": {"message": "bad key"}}).encode()),
        )

    provider = OpenRouterEmbeddingProvider(
        OPENROUTER_BGE_M3, api_key="k", dims=2, opener=opener
    )
    with pytest.raises(ConnectionError, match="bad key"):
        provider.embed_documents(["x"])


def test_build_embedder_openrouter_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from shamela_rag import factory
    from shamela_rag.config import Settings

    monkeypatch.setattr(
        factory,
        "get_settings",
        lambda: Settings(embedding_backend="openrouter", embedding_api_key=""),
    )
    with pytest.raises(ValueError, match="SHAMELA_EMBEDDING_API_KEY"):
        factory.build_embedder("bge-m3")


def test_build_embedder_openrouter_maps_model_names(monkeypatch: pytest.MonkeyPatch) -> None:
    from shamela_rag import factory
    from shamela_rag.config import Settings
    from shamela_rag.embeddings.openrouter import OpenRouterEmbeddingProvider

    monkeypatch.setattr(
        factory,
        "get_settings",
        lambda: Settings(
            embedding_backend="openrouter",
            embedding_api_key="secret",
            embedding_api_base_url="https://openrouter.ai/api/v1",
        ),
    )
    qwen = factory.build_embedder("qwen3")
    bge = factory.build_embedder("bge-m3")
    assert isinstance(qwen, OpenRouterEmbeddingProvider)
    assert isinstance(bge, OpenRouterEmbeddingProvider)
    assert qwen.model == OPENROUTER_QWEN3_EMBEDDING_8B
    assert bge.model == OPENROUTER_BGE_M3
